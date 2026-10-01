"""quoridor_upgrade_v2_to_v3.py: a v2 checkpoint (model, SWA model, SGD state) becomes a v3 checkpoint whose outputs
equal the v2 net's, whose optimizer state loads into a v3 model's optimizer and steps, and that is refused twice."""
import copy
import os
import sys

import pytest
import torch
from torch.optim.swa_utils import AveragedModel

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYTHON_DIR)

from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
import quoridor_upgrade_v2_to_v3 as upgrade  # noqa: E402


def make_v2_checkpoint(name):
    torch.manual_seed(7)
    config = copy.deepcopy(modelconfigs.config_of_name[name])
    model = Model(config, pos_len=9)
    model.initialize()
    # An SGD state as after some training (momentum buffers; no backward pass, which FlexAttention lacks on CPU).
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for p in model.parameters():
        opt.state[p]["momentum_buffer"] = torch.randn_like(p) * 1e-3
    swa = AveragedModel(model)
    return {"model": model.state_dict(), "swa_model": swa.state_dict(), "optimizer": opt.state_dict(),
            "config": config, "train_state": {"global_step_samples": 1234}}


def outputs(model, spatial, glob_):
    model.eval()
    with torch.no_grad():
        return [o for o in model.postprocess_output(model(spatial, glob_))[0] if torch.is_tensor(o)]


@pytest.mark.parametrize("name", ["b2c64_quoridor_v2", "tf2_b4c192_quoridor_v2"])
def test_upgrade(name):
    data = make_v2_checkpoint(name)
    v2 = Model(data["config"], pos_len=9)
    v2.load_state_dict(data["model"])
    upgraded = upgrade.upgrade_checkpoint(copy.deepcopy(data), log=lambda s: None)
    assert upgraded["config"]["quoridor_io_version"] == 3
    assert upgraded["train_state"][upgrade.MARKER] == 1234
    v3 = Model(upgraded["config"], pos_len=9)
    v3.load_state_dict(upgraded["model"])
    assert modelconfigs.get_num_bin_input_features(v3.config) == 21

    # Equal outputs, whatever the new inputs are (up to float summation order: CPU convolutions over 19 and 21 input
    # channels may sum in a different order; the run3 checkpoint check in docs/QuoridorIOv3.md was bit-exact).
    g = torch.Generator().manual_seed(3)
    spatial = torch.randn(8, 21, 9, 9, generator=g)
    spatial[:, 0] = 1.0
    glob_ = torch.randn(8, 19, generator=g)
    for a, b in zip(outputs(v2, spatial[:, :19], glob_[:, :17]), outputs(v3, spatial, glob_)):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-5), float((a - b).abs().max())

    # The SWA copy too.
    swa = AveragedModel(v3)
    swa.load_state_dict(upgraded["swa_model"])
    assert torch.equal(swa.module.conv_spatial.weight[:, 19:], torch.zeros_like(swa.module.conv_spatial.weight[:, 19:]))

    # The optimizer state loads into an optimizer over the v3 parameters and a step works; the momentum of the new
    # input columns starts at 0.
    opt = torch.optim.SGD(v3.parameters(), lr=0.01, momentum=0.9)
    opt.load_state_dict(upgraded["optimizer"])
    buf = opt.state[v3.conv_spatial.weight]["momentum_buffer"]
    assert buf.shape == v3.conv_spatial.weight.shape and (buf[:, 19:] == 0).all()
    assert torch.equal(buf[:, :19], data["optimizer"]["state"][0]["momentum_buffer"])
    for p in v3.parameters():
        p.grad = torch.full_like(p, 1e-3)
    opt.step()
    assert torch.isfinite(v3.conv_spatial.weight).all()

    # Already v3: refused.
    with pytest.raises(SystemExit):
        upgrade.upgrade_checkpoint(upgraded, log=lambda s: None)


def test_refuses_v1():
    config = copy.deepcopy(modelconfigs.config_of_name["b2c64_quoridor"])
    with pytest.raises(SystemExit):
        upgrade.upgrade_checkpoint({"model": {}, "config": config}, log=lambda s: None)
