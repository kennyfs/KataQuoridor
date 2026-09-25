import os
import sys
import tempfile
import torch
import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train.data_processing_pytorch import (
    apply_symmetry_quoridor,
    apply_symmetry_policy_quoridor,
    apply_symmetry_value_targets_quoridor,
)
from katago.train.export_onnx import export_quoridor_onnx, QuoridorOnnxExportWrapper


def test_quoridor_config_properties():
    for name in ["b2c64_quoridor", "tf3_b4c192_quoridor"]:
        assert name in modelconfigs.config_of_name
        cfg = modelconfigs.config_of_name[name]
        assert modelconfigs.is_quoridor(cfg)
        assert modelconfigs.get_num_bin_input_features(cfg) == 16
        assert modelconfigs.get_num_global_input_features(cfg) == 16


def test_quoridor_forward_shapes():
    B = 2
    spatial = torch.randn(B, 16, 9, 9)
    glob = torch.randn(B, 16)

    # 1. Test b2c64_quoridor
    cfg_b2 = modelconfigs.base_config_of_name["b2c64_quoridor"]
    model_b2 = Model(cfg_b2, pos_len=9)
    model_b2.eval()
    with torch.no_grad():
        out_byheads = model_b2(spatial, glob)
        post = model_b2.postprocess_output(out_byheads)

    assert len(post) == 1
    (
        policy,
        value,
        td_value,
        variance_time,
        game_margin,
        trajectory,
        wall_graph,
    ) = post[0]

    assert policy.shape == (B, 18, 9, 9)
    assert value.shape == (B, 2)
    assert td_value.shape == (B, 4, 2)
    assert variance_time.shape == (B,)
    assert game_margin.shape == (B,)
    assert trajectory.shape == (B, 2, 9, 9)
    assert wall_graph.shape == (B, 2, 9, 9)

    # 2. Test tf3_b4c192_quoridor
    cfg_tf3 = modelconfigs.base_config_of_name["tf3_b4c192_quoridor"]
    model_tf3 = Model(cfg_tf3, pos_len=9)
    model_tf3.eval()
    with torch.inference_mode():
        out_byheads_tf3 = model_tf3(spatial, glob)
        post_tf3 = model_tf3.postprocess_output(out_byheads_tf3)

    assert len(post_tf3) == 1
    (
        p_tf3,
        v_tf3,
        td_tf3,
        vt_tf3,
        gm_tf3,
        tr_tf3,
        wg_tf3,
    ) = post_tf3[0]

    assert p_tf3.shape == (B, 18, 9, 9)
    assert v_tf3.shape == (B, 2)
    assert td_tf3.shape == (B, 4, 2)
    assert vt_tf3.shape == (B,)
    assert gm_tf3.shape == (B,)
    assert tr_tf3.shape == (B, 2, 9, 9)
    assert wg_tf3.shape == (B, 2, 9, 9)


def test_quoridor_backward_and_gradients():
    B = 2
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
    model = Model(cfg, pos_len=9)
    model.train()
    metrics = Metrics(world_size=1, raw_model=model)

    spatial = torch.randn(B, 16, 9, 9)
    glob = torch.randn(B, 16)
    out_byheads = model(spatial, glob)
    post = model.postprocess_output(out_byheads)

    batch = {
        "binaryInputNCHW": spatial,
        "policyTargetsNCMove": torch.zeros(B, 6, 243),
        "globalTargetsNC": torch.zeros(B, 80),
        "valueTargetsNCHW": torch.zeros(B, 4, 9, 9),
    }
    batch["policyTargetsNCMove"][:, :, 0] = 1.0  # legal pawn move
    batch["globalTargetsNC"][:, 0] = 1.0        # win
    batch["globalTargetsNC"][:, 25] = 1.0       # global weight
    batch["globalTargetsNC"][:, 26] = 1.0       # p0 weight
    batch["globalTargetsNC"][:, 28] = 1.0       # p1 weight

    results = metrics.metrics_dict_batchwise(
        raw_model=model,
        model_output_postprocessed_byheads=post,
        extra_outputs=None,
        batch=batch,
        is_training=True,
        soft_policy_weight_scale=1.0,
        disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False,
        value_loss_scale=1.5,
        td_value_loss_scales=[0.2, 0.2, 0.2, 0.2],
        seki_loss_scale=1.0,
        variance_time_loss_scale=1.0,
        main_loss_scale=1.0,
        intermediate_loss_scale=0.25,
        include_model_norms=True,
    )

    loss = results["loss_sum"]
    assert torch.isfinite(loss).item()
    loss.backward()

    # Verify gradient flow to trunk and all heads
    assert model.conv_spatial.weight.grad is not None
    assert torch.isfinite(model.conv_spatial.weight.grad).all()

    assert model.linear_global.weight.grad is not None
    assert torch.isfinite(model.linear_global.weight.grad).all()

    assert model.policy_head.conv2p.weight.grad is not None
    assert torch.isfinite(model.policy_head.conv2p.weight.grad).all()

    assert model.value_head.linear_value.weight.grad is not None
    assert torch.isfinite(model.value_head.linear_value.weight.grad).all()

    assert model.value_head.linear_game_margin.weight.grad is not None
    assert torch.isfinite(model.value_head.linear_game_margin.weight.grad).all()

    assert model.value_head.conv_trajectory.weight.grad is not None
    assert torch.isfinite(model.value_head.conv_trajectory.weight.grad).all()

    assert model.value_head.conv_wall_graph.weight.grad is not None
    assert torch.isfinite(model.value_head.conv_wall_graph.weight.grad).all()


def test_quoridor_symmetries():
    B = 3
    # 1. Spatial symmetry (channels 14 and 15 are 8x8 wall anchors; row 8/col 8 are 0)
    spatial = torch.randn(B, 16, 9, 9)
    spatial[:, 14:16, 8, :] = 0.0
    spatial[:, 14:16, :, 8] = 0.0

    symm_0 = apply_symmetry_quoridor(spatial, 0)
    assert torch.allclose(symm_0, spatial)

    symm_1 = apply_symmetry_quoridor(spatial, 1)
    symm_2 = apply_symmetry_quoridor(symm_1, 1)
    # Applying reflection twice should return the original tensor
    assert torch.allclose(symm_2, spatial, atol=1e-6)

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


def test_quoridor_onnx_export():
    cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
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
        assert meta["katago.numInputChannels"] == "16"
        assert meta["katago.numInputGlobalChannels"] == "16"
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


if __name__ == "__main__":
    print("Testing Quoridor model configs...")
    test_quoridor_config_properties()
    print("  -> Passed!")

    print("Testing Quoridor forward shapes (b2c64 & tf3_b4c192)...")
    test_quoridor_forward_shapes()
    print("  -> Passed!")

    print("Testing Quoridor backward pass and gradient flow through all heads...")
    test_quoridor_backward_and_gradients()
    print("  -> Passed!")

    print("Testing Quoridor symmetries (spatial, policy, value targets)...")
    test_quoridor_symmetries()
    print("  -> Passed!")

    print("Testing Quoridor ONNX export and metadata validation...")
    test_quoridor_onnx_export()
    print("  -> Passed!")

    print("\nALL QUORIDOR MODEL TESTS PASSED SUCCESSFULLY!")
