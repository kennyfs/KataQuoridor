"""T15: Q4 model shapes, a loss on a synthetic batch and one optimizer step (CPU), plus the consistency of the Q4
constants between Python (modelconfigs.py) and C++ (cpp/q4/nn/q4nnconstants.h)."""

import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import modelconfigs
from katago.train.model_pytorch import Model

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q4_CONFIGS = ["b1c32_q4", "b2c64_q4", "tf2_b4c192_q4"]


def make_inputs(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    spatial = torch.randn(batch, 27, 11, 11, generator=g)
    spatial[:, 0, :, :] = 1.0  # channel 0 is the on-board mask
    return spatial, torch.randn(batch, 28, generator=g)


def cpp_constants():
    text = open(os.path.join(REPO_DIR, "cpp", "q4", "nn", "q4nnconstants.h")).read()
    return {name: int(value) for name, value in re.findall(r"static constexpr int (\w+) = (\d+);", text)}


@pytest.mark.parametrize("name", Q4_CONFIGS)
def test_q4_config_properties(name):
    cfg = modelconfigs.config_of_name[name]
    assert modelconfigs.is_quoridor4(cfg) and not modelconfigs.is_quoridor(cfg)
    assert modelconfigs.get_q4_io_version(cfg) == 1
    assert modelconfigs.get_num_bin_input_features(cfg) == 27
    assert modelconfigs.get_num_global_input_features(cfg) == 28


def test_python_and_cpp_constants_agree():
    c = cpp_constants()
    assert c["Q4_IO_VERSION_BASE"] == modelconfigs.Q4_IO_VERSION_BASE
    assert c["Q4_IO_VERSION_1"] == modelconfigs.Q4_IO_VERSION_BASE + 1
    assert c["Q4_IO_VERSION_2"] == modelconfigs.Q4_IO_VERSION_BASE + 2
    assert c["NUM_METADATA_INPUTS"] == 192 == modelconfigs.get_num_meta_encoder_input_features(1)
    assert c["NUM_STYLE_FEATURES"] == modelconfigs.Q4_NUM_STYLE_FEATURES == 76
    assert c["NUM_SPATIAL_CHANNELS"] == modelconfigs.Q4_NUM_BIN_INPUT_FEATURES[1]
    assert c["NUM_GLOBAL_FEATURES"] == modelconfigs.Q4_NUM_GLOBAL_INPUT_FEATURES[1]
    assert c["NUM_VALUE_LOGITS"] == modelconfigs.Q4_NUM_VALUE_LOGITS
    assert c["NUM_MISC"] == modelconfigs.Q4_NUM_MISC
    assert c["NUM_OWNERSHIP_CHANNELS"] == modelconfigs.Q4_NUM_OWNERSHIP_CHANNELS
    assert c["NUM_POLICY_VARIANTS"] * c["NUM_POLICY_PLANES"] == 6
    assert c["POS_LEN"] == 11


@pytest.mark.parametrize("name", Q4_CONFIGS)
def test_q4_forward_shapes(name):
    batch = 4
    spatial, glob = make_inputs(batch)
    model = Model(modelconfigs.config_of_name[name], pos_len=11)
    model.initialize()
    model.eval()
    with torch.no_grad():
        out = model(spatial, glob)
        post = model.postprocess_output(out)
    for outputs in (out, post):
        assert len(outputs) == 1
        policy, value, misc, trajectory, paths, walls, td_value, policy_aux = outputs[0]
        assert td_value.shape == (batch, 4, 5)         # training only: TD value logits, 4 horizons x 5 classes
        assert policy_aux.shape == (batch, 3, 3, 11, 11)  # training only: next-seat, soft, soft next-seat policy
        assert policy.shape == (batch, 2, 3, 11, 11)  # variant (search, style), plane (pawn, V wall, H wall)
        assert value.shape == (batch, 5)
        assert misc.shape == (batch, 6)
        assert trajectory.shape == (batch, 1, 11, 11)  # exported: my trajectory
        assert paths.shape == (batch, 4, 11, 11)       # training only: all seats' trajectories
        assert walls.shape == (batch, 8, 11, 11)       # training only: wall placements, seat x orientation


# FlexAttention has no CPU backward, so the transformer net takes its optimizer step on the GPU (skipped without one).
@pytest.mark.parametrize("name,device", [("b1c32_q4", "cpu"), ("b2c64_q4", "cpu"), ("tf2_b4c192_q4", "cuda")])
def test_q4_loss_and_optimizer_step(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    batch = 4
    spatial, glob = make_inputs(batch)
    spatial, glob = spatial.to(device), glob.to(device)
    model = Model(modelconfigs.config_of_name[name], pos_len=11)
    model.initialize()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    policy, value, misc, trajectory, paths, walls, td_value, policy_aux = model.postprocess_output(model(spatial, glob))[0]
    ce = torch.nn.functional.cross_entropy
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss = (
        ce(policy[:, 0].reshape(batch, -1), torch.zeros(batch, dtype=torch.long, device=device))
        + ce(policy[:, 1].reshape(batch, -1), torch.zeros(batch, dtype=torch.long, device=device))
        + ce(value, torch.zeros(batch, dtype=torch.long, device=device))
        + torch.nn.functional.huber_loss(misc, torch.randn(batch, 6, device=device))
        + bce(trajectory, torch.zeros_like(trajectory))
        + bce(paths, torch.zeros_like(paths))
        + bce(walls, torch.zeros_like(walls))
        + ce(td_value.reshape(batch * 4, 5), torch.zeros(batch * 4, dtype=torch.long, device=device))
        + ce(policy_aux[:, 0].reshape(batch, -1), torch.zeros(batch, dtype=torch.long, device=device))
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    heads = model.value_head
    for param in (model.conv_spatial.weight, model.policy_head.conv2p.weight, heads.linear_value.weight,
                  heads.linear_misc.weight, heads.conv_trajectory.weight, heads.conv_all_trajectories.weight,
                  heads.conv_wall_placements.weight, heads.linear_td_value.weight, model.policy_head.conv2p_aux.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    before = model.conv_spatial.weight.detach().clone()
    optimizer.step()
    after = model.conv_spatial.weight.detach()
    assert not torch.equal(before, after) and torch.isfinite(after).all()


def test_training_only_heads_are_not_exported_but_trained_params_exist():
    model = Model(modelconfigs.config_of_name["b1c32_q4"], pos_len=11)
    names = {n for n, _ in model.named_parameters()}
    assert "value_head.conv_all_trajectories.weight" in names and "value_head.conv_wall_placements.weight" in names
    assert "value_head.linear_td_value.weight" in names and "policy_head.conv2p_aux.weight" in names
