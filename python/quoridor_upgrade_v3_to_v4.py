#!/usr/bin/env python3
"""Upgrade a Quoridor I/O v3 training checkpoint to I/O v4 (docs/QuoridorIOv4.md).

I/O v4 has the v3 inputs (the same 21 spatial / 19 global channels) and a third value logit, the draw (noResult) outcome,
in the value head and in each of the 4 TD value horizons. So only the value head changes:

- linear_value [2, c] -> [3, c] and linear_td_value [4 * 2, c] -> [4 * 3, c]: the new noResult rows get zero weights and bias
  NR_BIAS (default -8: P(noResult) = 3e-4 instead of exactly 0 so that it can be learned; every other probability
  changes by less than that, and the exported net's win / loss / score outputs by less than 1e-3);
- the same in the SWA model (state dict) and in the optimizer state (every state tensor of one of the old shapes is padded the
  same way: zeros), so train.py resumes with optimizer.load_state_dict as usual;
- config["quoridor_io_version"] = 4, config["num_value_outputs"] = 3, and a marker in train_state.

The upgraded net computes what the v3 net computed up to the noResult logit's tiny probability. To keep training it on
v3-style draw targets (a draw as win 0.5 / loss 0.5), run train.py with `-quoridor-draw-target half`.

Usage:
  python quoridor_upgrade_v3_to_v4.py -checkpoint IN.ckpt -output OUT.ckpt [-nr-bias -8]
  python quoridor_upgrade_v3_to_v4.py -checkpoint BASEDIR/train/NAME/checkpoint.ckpt -in-place
      (keeps the original as checkpoint.ckpt.pre_v4)

Refuses a checkpoint that is not I/O v3. To switch a running training loop, see docs/QuoridorIOv4.md.
"""
import argparse
import os
import shutil
import sys

import torch

import katago.train.load_model

NUM_TD = 4
VALUE_SUFFIXES = ("value_head.linear_value.weight", "value_head.linear_value.bias")
TD_SUFFIXES = ("value_head.linear_td_value.weight", "value_head.linear_td_value.bias")
MARKER = "quoridor_upgraded_v3_to_v4"
BACKUP_SUFFIX = ".pre_v4"
NR_BIAS_DEFAULT = -8.0


def add_nr_rows(t: torch.Tensor, groups: int, fill: float) -> torch.Tensor:
    """[groups * 2, ...] -> [groups * 3, ...]: inserts a row `fill` after each group's win and loss rows."""
    assert t.shape[0] == groups * 2, f"unexpected shape {tuple(t.shape)}"
    g = t.reshape(groups, 2, *t.shape[1:])
    extra = torch.full((groups, 1, *t.shape[1:]), fill, dtype=t.dtype, device=t.device)
    return torch.cat([g, extra], dim=1).reshape(groups * 3, *t.shape[1:])


def find_key(state: dict, suffix: str) -> str:
    keys = [k for k in state if k == suffix or k.endswith("." + suffix)]
    assert len(keys) == 1, f"expected exactly one *{suffix} in the state dict, found {keys}"
    return keys[0]


def transforms(nr_bias: float) -> dict:
    """suffix -> (old shape check, function) for every tensor that changes, applied to weights, SWA weights and
    optimizer state (optimizer state tensors of the old shape get fill 0 instead of nr_bias)."""
    return {
        VALUE_SUFFIXES[0]: lambda t, fill: add_nr_rows(t, 1, 0.0),
        VALUE_SUFFIXES[1]: lambda t, fill: add_nr_rows(t, 1, fill),
        TD_SUFFIXES[0]: lambda t, fill: add_nr_rows(t, NUM_TD, 0.0),
        TD_SUFFIXES[1]: lambda t, fill: add_nr_rows(t, NUM_TD, fill),
    }


def upgrade_state_dict(state: dict, what: str, nr_bias: float, log) -> dict:
    """Upgrades the tensors in place. Returns {suffix: old shape}."""
    old = {}
    for suffix, fn in transforms(nr_bias).items():
        k = find_key(state, suffix)
        old[suffix] = tuple(state[k].shape)
        state[k] = fn(state[k], nr_bias)
        log(f"{what}: {k} {old[suffix]} -> {tuple(state[k].shape)}")
    return old


def upgrade_optimizer(opt: dict, old_shapes: dict, nr_bias: float, log) -> int:
    """Extends every optimizer state tensor that has one of the old shapes (zero rows). Each old shape must belong to exactly one
    parameter's state, else the mapping is ambiguous. Returns the number of tensors padded."""
    fns = transforms(nr_bias)
    shape_to_suffix = {}
    for suffix, shape in old_shapes.items():
        assert shape not in shape_to_suffix, f"{suffix} and {shape_to_suffix[shape]} have the same shape {shape}"
        shape_to_suffix[shape] = suffix
    owners = {s: set() for s in shape_to_suffix}
    n = 0
    for idx, st in opt.get("state", {}).items():
        for name, v in st.items():
            if torch.is_tensor(v) and tuple(v.shape) in shape_to_suffix:
                shape = tuple(v.shape)
                st[name] = fns[shape_to_suffix[shape]](v, 0.0)
                owners[shape].add(idx)
                n += 1
    for shape, idxs in owners.items():
        assert len(idxs) <= 1, f"optimizer state of {len(idxs)} parameters has the shape {shape} of {shape_to_suffix[shape]}: ambiguous"
    log(f"optimizer: padded {n} state tensors (parameters {sorted(i for s in owners.values() for i in s)})")
    return n


def upgrade_checkpoint(data: dict, nr_bias: float = NR_BIAS_DEFAULT, log=print) -> dict:
    config = data.get("config")
    assert config is not None and config.get("game") == "quoridor", "not a Quoridor checkpoint (no config)"
    io_version = config.get("quoridor_io_version", 1)
    train_state = data.setdefault("train_state", {})
    if io_version != 3:
        raise SystemExit(f"checkpoint is Quoridor I/O v{io_version}, expected v3" +
                         (" (already upgraded)" if train_state.get(MARKER) is not None else ""))
    assert config.get("num_value_outputs", 2) == 2
    assert config.get("quoridor_io_version") == 3
    old_shapes = upgrade_state_dict(data["model"], "model", nr_bias, log)
    if data.get("swa_model") is not None:
        assert upgrade_state_dict(data["swa_model"], "swa_model", nr_bias, log) == old_shapes
    if data.get("optimizer") is not None:
        upgrade_optimizer(data["optimizer"], old_shapes, nr_bias, log)
    config["quoridor_io_version"] = 4
    config["num_value_outputs"] = 3
    train_state[MARKER] = int(train_state.get("global_step_samples", 0))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-checkpoint", required=True, help="I/O v3 training checkpoint (.ckpt)")
    ap.add_argument("-output", help="where to write the v4 checkpoint")
    ap.add_argument("-in-place", action="store_true", help="overwrite -checkpoint (keeping a .pre_v4 backup)")
    ap.add_argument("-nr-bias", type=float, default=NR_BIAS_DEFAULT, help="initial bias of the noResult logits")
    args = ap.parse_args(argv)
    if bool(args.output) == bool(args.in_place):
        raise SystemExit("give exactly one of -output and -in-place")
    data = katago.train.load_model.load_checkpoint(args.checkpoint)
    upgrade_checkpoint(data, args.nr_bias)
    out = args.output
    if args.in_place:
        shutil.copy2(args.checkpoint, args.checkpoint + BACKUP_SUFFIX)
        out = args.checkpoint
    tmp = out + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, out)
    print(f"Wrote I/O v4 checkpoint {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
