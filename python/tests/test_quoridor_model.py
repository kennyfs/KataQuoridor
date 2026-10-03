import os
import sys
import tempfile
import pytest
import torch
import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import modelconfigs
from katago.train.load_model import load_model_state_dict
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train.data_processing_pytorch import (
    apply_symmetry_quoridor,
    apply_symmetry_policy_quoridor,
    apply_symmetry_value_targets_quoridor,
)
from export_model_pytorch import export_quoridor_onnx, QuoridorOnnxExportWrapper

V1_CONFIGS = ["b2c64_quoridor", "tf2_b4c192_quoridor"]
V2_CONFIGS = ["b2c64_quoridor_v2", "tf2_b4c192_quoridor_v2"]
V3_CONFIGS = ["b2c64_quoridor_v3", "tf2_b4c192_quoridor_v3"]
# Input channel counts by Quoridor I/O version (QuoridorNN in cpp/neuralnet/quoridornn.h).
NUM_SPATIAL = {1: 17, 2: 19, 3: 21}
NUM_GLOBAL = {1: 15, 2: 17, 3: 19}

METRICS_ARGS = dict(
    soft_policy_weight_scale=1.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
    value_loss_scale=1.5, td_value_loss_scales=[0.2, 0.2, 0.2, 0.2], seki_loss_scale=1.0,
    variance_time_loss_scale=1.0, main_loss_scale=1.0, intermediate_loss_scale=0.25,
)


def make_inputs(B, io_version, seed=0):
    g = torch.Generator().manual_seed(seed)
    spatial = torch.randn(B, NUM_SPATIAL[io_version], 9, 9, generator=g)
    spatial[:, 0, :, :] = 1.0  # channel 0 is the all-ones on-board mask in real rows
    glob = torch.randn(B, NUM_GLOBAL[io_version], generator=g)
    return spatial, glob


def test_quoridor_config_properties():
    for names, io_version in ((V1_CONFIGS, 1), (V2_CONFIGS, 2), (V3_CONFIGS, 3)):
        for name in names:
            assert name in modelconfigs.config_of_name
            cfg = modelconfigs.config_of_name[name]
            assert modelconfigs.is_quoridor(cfg)
            assert modelconfigs.get_quoridor_io_version(cfg) == io_version
            assert modelconfigs.get_num_bin_input_features(cfg) == NUM_SPATIAL[io_version]
            assert modelconfigs.get_num_global_input_features(cfg) == NUM_GLOBAL[io_version]
    assert modelconfigs.QUORIDOR_TRAINING_IO_VERSION == 3


@pytest.mark.parametrize("name", V1_CONFIGS + V2_CONFIGS + V3_CONFIGS)
def test_quoridor_forward_shapes(name):
    B = 2
    cfg = modelconfigs.base_config_of_name[name]
    io_version = modelconfigs.get_quoridor_io_version(cfg)
    spatial, glob = make_inputs(B, io_version)
    model = Model(cfg, pos_len=9)
    model.initialize()
    model.eval()
    with torch.no_grad():
        post = model.postprocess_output(model(spatial, glob))

    assert len(post) == 1
    (
        policy,
        value,
        td_value,
        variance_time,
        utility_score,
        utility_score_stdev,
        st_value_error,
        st_score_error,
        trajectory,
        wall_graph,
        lead,
        remaining_turns,
    ) = post[0]

    assert policy.shape == (B, 18, 9, 9)
    assert value.shape == (B, 2)
    assert td_value.shape == (B, 4, 2)
    assert variance_time.shape == (B,)
    assert utility_score.shape == (B,)
    assert lead.shape == (B,)
    assert remaining_turns.shape == (B,)
    for t in (utility_score_stdev, st_value_error, st_score_error):
        assert t.shape == (B,)
        assert (t > 0).all()
    assert trajectory.shape == (B, 2, 9, 9)
    assert wall_graph.shape == (B, 2, 9, 9)
    if io_version == 1:
        # One margin head serves as both the score and the lead; no remaining-plies head.
        assert torch.equal(lead, utility_score)
        assert (remaining_turns == 0).all()
        assert not hasattr(model.value_head, "linear_lead")
    else:
        assert not torch.equal(lead, utility_score)


def outcome_batch(B, io_version):
    """A batch with every target of the Quoridor loss set, as written by TrainingWriteBuffers::addRow."""
    spatial, glob = make_inputs(B, io_version, seed=1)
    batch = {
        "binaryInputNCHW": spatial,
        "globalInputNC": glob,
        "policyTargetsNCMove": torch.zeros(B, 2, 243),
        "globalTargetsNC": torch.zeros(B, 80),
        "valueTargetsNCHW": torch.zeros(B, 4, 9, 9),
    }
    batch["policyTargetsNCMove"][:, :, 0] = 1.0  # legal pawn move
    g = batch["globalTargetsNC"]
    g[:, 0] = 1.0        # win
    g[:, 4:18:4] = 1.0   # td wins
    g[:, 15] = 3.0       # short-term score
    g[:, 20] = 12.5      # final utility score u
    g[:, 21] = 0.5       # final lead s
    g[:, 23] = 60.0      # plies left
    g[:, 25] = 1.0       # global weight
    g[:, 26] = 1.0       # p0 weight
    g[:, 27] = 1.0       # outcome weight (u, remaining plies, trajectory / walls)
    g[:, 28] = 1.0       # p1 weight
    g[:, 29] = 1.0       # lead weight
    return batch


def run_metrics(model, batch, is_training=True, include_model_norms=False):
    metrics = Metrics(world_size=1, raw_model=model)
    post = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))
    return metrics.metrics_dict_batchwise(
        raw_model=model, model_output_postprocessed_byheads=post, extra_outputs=None, batch=batch,
        is_training=is_training, include_model_norms=include_model_norms, **METRICS_ARGS,
    )


def test_quoridor_backward_and_gradients():
    B = 2
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor_v3"]
    model = Model(cfg, pos_len=9)
    model.initialize()
    model.train()
    results = run_metrics(model, outcome_batch(B, 3), include_model_norms=True)

    loss = results["loss_sum"]
    assert torch.isfinite(loss).item()
    for key in ("smloss_sum", "leadloss_sum", "rtloss_sum", "sdregloss_sum", "esstloss_sum"):
        assert results[key].item() > 0, key
    assert "gmloss_sum" not in results
    loss.backward()

    # Gradient flow to the trunk and every head
    vh = model.value_head
    for param in (
        model.conv_spatial.weight, model.linear_global.weight, model.policy_head.conv2p.weight,
        vh.linear_value.weight, vh.linear_utility_score.weight, vh.linear_lead.weight,
        vh.linear_remaining_turns.weight, vh.linear_misc.weight, vh.conv_trajectory.weight, vh.conv_wall_graph.weight,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_quoridor_draw_rows_have_no_lead_loss():
    # A draw row (cpp/dataio/trainingwrite.cpp): value 0.5 / 0.5, u = 0 with its normal weight, lead weight 0,
    # plies left to maxPlies.
    torch.manual_seed(0)
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor_v3"]
    model = Model(cfg, pos_len=9)
    model.initialize()
    model.train()

    batch = outcome_batch(2, 3)
    g = batch["globalTargetsNC"]
    g[1, 0:2] = 0.5   # draw
    g[1, 20] = 0.0    # u = 0
    g[1, 21] = 0.0    # no lead ...
    g[1, 29] = 0.0    # ... and no lead weight
    g[1, 23] = 7.0

    draw_only = {k: v[1:2] for k, v in batch.items()}
    results = run_metrics(model, draw_only)
    assert results["leadloss_sum"].item() == 0.0
    assert results["smloss_sum"].item() > 0.0
    assert results["rtloss_sum"].item() > 0.0
    model.zero_grad()
    results["loss_sum"].backward()
    assert model.value_head.linear_lead.weight.grad.abs().sum() == 0
    assert model.value_head.linear_utility_score.weight.grad.abs().sum() > 0

    # With the won row next to it, the lead loss is that of the won row alone.
    won_only = {k: v[0:1] for k, v in batch.items()}
    both = run_metrics(model, batch)["leadloss_sum"].item()
    assert both == pytest.approx(run_metrics(model, won_only)["leadloss_sum"].item(), rel=1e-5)
    assert both > 0


def test_quoridor_v1_models_cannot_train():
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
    model = Model(cfg, pos_len=9)
    model.initialize()
    with pytest.raises(AssertionError, match="only Quoridor I/O v3"):
        run_metrics(model, outcome_batch(1, 1))


def test_quoridor_v1_checkpoint_keys_load():
    # 0.1.0 checkpoints name the score head linear_game_margin.
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
    model = Model(cfg, pos_len=9)
    model.initialize()
    old = {k.replace(".linear_utility_score.", ".linear_game_margin."): v for k, v in model.state_dict().items()}
    assert any(".linear_game_margin." in k for k in old)
    fresh = Model(cfg, pos_len=9)
    fresh.load_state_dict(load_model_state_dict({"model": old}))
    assert torch.equal(fresh.value_head.linear_utility_score.weight, model.value_head.linear_utility_score.weight)


def test_quoridor_shortterm_optimistic_policy_weight_uses_short_horizon():
    # The short-term optimistic policy (exported as the search's optimistic policy channel) is weighted by how
    # much better than expected the short-term TD target turned out: horizon index 2, globalTargetsNC[12:16],
    # whose nowFactor 1/(1 + 81*0.016) is the largest. globalTargetsNC[4:8] is the longest horizon.
    torch.manual_seed(0)
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor_v3"]
    model = Model(cfg, pos_len=9)
    model.initialize()
    model.eval()
    metrics = Metrics(world_size=1, raw_model=model)

    def shortopt_weight(short_win, long_win):
        spatial = torch.zeros(1, NUM_SPATIAL[3], 9, 9)
        spatial[:, 0, :, :] = 1.0
        with torch.no_grad():
            post = model.postprocess_output(model(spatial, torch.zeros(1, NUM_GLOBAL[3])))
        g = torch.zeros(1, 80)
        g[:, 0:2] = 0.5
        g[:, 4:6] = torch.tensor([long_win, 1.0 - long_win])
        g[:, 8:10] = 0.5
        g[:, 12:14] = torch.tensor([short_win, 1.0 - short_win])
        g[:, 16:18] = 0.5
        g[:, 25] = 1.0  # global weight
        g[:, 26] = 1.0  # p0 weight
        g[:, 27] = 1.0  # game finished, not a side position
        batch = {
            "binaryInputNCHW": spatial,
            "policyTargetsNCMove": torch.zeros(1, 2, 243),
            "globalTargetsNC": g,
            "valueTargetsNCHW": torch.zeros(1, 4, 9, 9),
        }
        batch["policyTargetsNCMove"][:, :, 0] = 1.0
        results = metrics.metrics_dict_batchwise(
            raw_model=model, model_output_postprocessed_byheads=post, extra_outputs=None, batch=batch,
            is_training=False, include_model_norms=False, **METRICS_ARGS,
        )
        return float(results["p0soptw_sum"])

    assert shortopt_weight(short_win=1.0, long_win=0.0) > 0.5
    assert shortopt_weight(short_win=0.0, long_win=1.0) < 0.1


@pytest.mark.parametrize("io_version", [1, 2, 3])
def test_quoridor_symmetries(io_version):
    B = 3
    C = NUM_SPATIAL[io_version]
    wall_channels = [14, 15, 16] + ([17, 18] if io_version >= 2 else [])
    # 1. Spatial symmetry (wall-anchor channels are 8x8: row 8 / col 8 are 0)
    spatial = torch.randn(B, C, 9, 9)
    spatial[:, wall_channels, 8, :] = 0.0
    spatial[:, wall_channels, :, 8] = 0.0

    symm_0 = apply_symmetry_quoridor(spatial, 0)
    assert torch.allclose(symm_0, spatial)

    symm_1 = apply_symmetry_quoridor(spatial, 1)
    symm_2 = apply_symmetry_quoridor(symm_1, 1)
    # Applying reflection twice should return the original tensor
    assert torch.allclose(symm_2, spatial, atol=1e-6)
    # Wall-anchor channels mirror c -> 7 - c, pawn-cell channels c -> 8 - c, and E/W blocked (5, 6) swap.
    for ch in wall_channels:
        assert torch.equal(symm_1[:, ch, :8, :8], torch.flip(spatial[:, ch, :8, :8], dims=[-1]))
        assert (symm_1[:, ch, 8, :] == 0).all() and (symm_1[:, ch, :, 8] == 0).all()
    for ch in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13) + ((19, 20) if io_version >= 3 else ()):
        assert torch.equal(symm_1[:, ch], torch.flip(spatial[:, ch], dims=[-1]))
    assert torch.equal(symm_1[:, 5], torch.flip(spatial[:, 6], dims=[-1]))

    # 2. Policy symmetry (6 targets, each 243 slots: 81 pawn + 64 V-wall + 64 H-wall)
    p_planes = torch.randn(B, 6, 3, 9, 9)
    p_planes[:, :, 1, 8, :] = 0.0
    p_planes[:, :, 1, :, 8] = 0.0
    p_planes[:, :, 2, 8, :] = 0.0
    p_planes[:, :, 2, :, 8] = 0.0
    policy = p_planes.view(B, 6, 243)

    p_symm_1 = apply_symmetry_policy_quoridor(policy, 1)
    p_symm_2 = apply_symmetry_policy_quoridor(p_symm_1, 1)
    assert torch.allclose(p_symm_2, policy, atol=1e-6)

    # 3. Value targets symmetry (ch 0..1 trajectory 9x9, ch 2..3 wall graph 8x8)
    val_targets = torch.randn(B, 4, 9, 9)
    val_targets[:, 2:4, 8, :] = 0.0
    val_targets[:, 2:4, :, 8] = 0.0

    vt_symm_1 = apply_symmetry_value_targets_quoridor(val_targets, 1)
    vt_symm_2 = apply_symmetry_value_targets_quoridor(vt_symm_1, 1)
    assert torch.allclose(vt_symm_2, val_targets, atol=1e-6)


@pytest.mark.parametrize("name", ["b2c64_quoridor", "b2c64_quoridor_v2", "b2c64_quoridor_v3"])
def test_quoridor_onnx_export(name):
    cfg = modelconfigs.base_config_of_name[name]
    io_version = modelconfigs.get_quoridor_io_version(cfg)
    model = Model(cfg, pos_len=9)
    model.initialize()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "test_quoridor.onnx")
        export_quoridor_onnx(model, onnx_file, model_name="test_model_export")

        assert os.path.exists(onnx_file)
        assert os.path.getsize(onnx_file) > 0

        proto = onnx.load(onnx_file)
        onnx.checker.check_model(proto)

        meta = {prop.key: prop.value for prop in proto.metadata_props}
        assert meta["katago.metadataVersion"] == "1"
        assert meta["katago.name"] == "test_model_export"
        assert meta["katago.modelVersion"] == "1"
        assert meta["katago.numInputChannels"] == str(NUM_SPATIAL[io_version])
        assert meta["katago.numInputGlobalChannels"] == str(NUM_GLOBAL[io_version])
        assert meta["katago.numPolicyChannels"] == "3"
        assert meta["katago.numValueChannels"] == "2"
        assert meta["katago.numScoreValueChannels"] == "0"
        assert meta["katago.numOwnershipChannels"] == "0"
        assert meta["katago.build.nnXLen"] == "9"
        assert meta["katago.build.nnYLen"] == "9"
        assert meta["katago.build.requireExactNNLen"] == "true"

        in_names = [i.name for i in proto.graph.input]
        out_names = [o.name for o in proto.graph.output]
        assert "InputSpatial" in in_names
        assert "InputGlobal" in in_names
        assert "OutputPolicy" in out_names
        assert "OutputValue" in out_names


def test_quoridor_models_not_per_block_compiled():
    # Upstream's per-block compiled trunk only knows the Go heads; Quoridor must fall back to a single graph.
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model
    for name in ("tf2_b4c192_quoridor_v3", "tf3_b5c256_quoridor_v3"):
        model = Model(modelconfigs.config_of_name[name], 9)
        assert not model.supports_per_block_compile(), name
