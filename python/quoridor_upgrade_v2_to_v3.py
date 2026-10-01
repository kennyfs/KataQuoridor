#!/usr/bin/env python3
"""Upgrade a Quoridor I/O v2 training checkpoint to I/O v3 (docs/QuoridorIOv3.md).

I/O v3 = v2 + 2 spatial input channels (19, 20: repeating / drawing pawn moves) and 2 global input channels
(17, 18: repetition rule on, repetition progress), appended. The only parameters that see the inputs are the first
layers, conv_spatial.weight [C, 19, k, k] and linear_global.weight [C, 17]; this adds the new input columns to them
with zeros, so the upgraded net computes exactly what the v2 net computed (the new inputs are multiplied by 0) and
then learns to use them. Updated consistently:

- the model and, if present, the SWA model (state dicts);
- the optimizer state (SGD momentum buffers, Adam moments, ...): every state tensor with the old shape of one of
  the two layers is zero-padded the same way, so train.py resumes with optimizer.load_state_dict as usual;
- config["quoridor_io_version"] = 3, and a marker in train_state (train.py and clean_checkpoint.py carry it forward).
  Lookahead slow weights are not stored in checkpoints (train.py re-initializes them from the parameters).

Usage:
  python quoridor_upgrade_v2_to_v3.py -checkpoint IN.ckpt -output OUT.ckpt
  python quoridor_upgrade_v2_to_v3.py -checkpoint BASEDIR/train/NAME/checkpoint.ckpt -in-place
      (keeps the original as checkpoint.ckpt.pre_v3)

Refuses a checkpoint that is not I/O v2. To switch a running training loop, see docs/QuoridorIOv3.md.
"""
import argparse
import os
import shutil
import sys

import torch

import katago.train.load_model

OLD_SPATIAL, NEW_SPATIAL = 19, 21
OLD_GLOBAL, NEW_GLOBAL = 17, 19
SPATIAL_SUFFIX = "conv_spatial.weight"
GLOBAL_SUFFIX = "linear_global.weight"
MARKER = "quoridor_upgraded_v2_to_v3"
BACKUP_SUFFIX = ".pre_v3"


def pad_input_dim(t: torch.Tensor, new_size: int) -> torch.Tensor:
    """Zero-pads dimension 1 (the input channels) of a weight-shaped tensor."""
    shape = list(t.shape)
    pad = new_size - shape[1]
    assert pad > 0, f"unexpected shape {tuple(t.shape)}"
    shape[1] = pad
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=1)


def find_key(state: dict, suffix: str) -> str:
    keys = [k for k in state if k == suffix or k.endswith("." + suffix)]
    assert len(keys) == 1, f"expected exactly one *{suffix} in the state dict, found {keys}"
    return keys[0]


def upgrade_state_dict(state: dict, what: str, log) -> tuple:
    """Pads the two input layers in place. Returns their old shapes."""
    ks = find_key(state, SPATIAL_SUFFIX)
    kg = find_key(state, GLOBAL_SUFFIX)
    assert state[ks].dim() == 4 and state[ks].shape[1] == OLD_SPATIAL, f"{what} {ks}: {tuple(state[ks].shape)}"
    assert state[kg].dim() == 2 and state[kg].shape[1] == OLD_GLOBAL, f"{what} {kg}: {tuple(state[kg].shape)}"
    old = (tuple(state[ks].shape), tuple(state[kg].shape))
    state[ks] = pad_input_dim(state[ks], NEW_SPATIAL)
    state[kg] = pad_input_dim(state[kg], NEW_GLOBAL)
    log(f"{what}: {ks} {old[0]} -> {tuple(state[ks].shape)}, {kg} {old[1]} -> {tuple(state[kg].shape)}")
    return old


def upgrade_optimizer(opt: dict, old_shapes: tuple, log) -> int:
    """Pads every optimizer state tensor that has one of the two old input-layer shapes. Each layer must match
    exactly one parameter's state (the shapes are unique in Quoridor models). Returns the number of tensors padded."""
    old_spatial, old_global = old_shapes
    matched = {old_spatial: [], old_global: []}
    n = 0
    for idx, st in opt.get("state", {}).items():
        hit = None
        for name, v in st.items():
            if torch.is_tensor(v) and tuple(v.shape) in matched:
                hit = tuple(v.shape)
                st[name] = pad_input_dim(v, NEW_SPATIAL if hit == old_spatial else NEW_GLOBAL)
                n += 1
        if hit is not None:
            matched[hit].append(idx)
    for shape, idxs in matched.items():
        assert len(idxs) <= 1, f"optimizer state of {len(idxs)} parameters has the input-layer shape {shape}: ambiguous"
    log(f"optimizer: padded {n} state tensors (parameters {matched[old_spatial]} and {matched[old_global]})")
    return n


def upgrade_checkpoint(data: dict, log=print) -> dict:
    config = data.get("config")
    assert config is not None and config.get("game") == "quoridor", "not a Quoridor checkpoint (no config)"
    io_version = config.get("quoridor_io_version", 1)
    train_state = data.setdefault("train_state", {})
    if io_version != 2:
        raise SystemExit(f"checkpoint is Quoridor I/O v{io_version}, expected v2" +
                         (" (already upgraded)" if train_state.get(MARKER) is not None else ""))
    old_shapes = upgrade_state_dict(data["model"], "model", log)
    if data.get("swa_model") is not None:
        assert upgrade_state_dict(data["swa_model"], "swa_model", log) == old_shapes
    if data.get("optimizer") is not None:
        upgrade_optimizer(data["optimizer"], old_shapes, log)
    config["quoridor_io_version"] = 3
    train_state[MARKER] = int(train_state.get("global_step_samples", 0))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-checkpoint", required=True, help="I/O v2 training checkpoint (.ckpt)")
    ap.add_argument("-output", help="where to write the v3 checkpoint")
    ap.add_argument("-in-place", action="store_true", help="overwrite -checkpoint (keeping a .pre_v3 backup)")
    args = ap.parse_args(argv)
    if bool(args.output) == bool(args.in_place):
        raise SystemExit("give exactly one of -output and -in-place")
    data = katago.train.load_model.load_checkpoint(args.checkpoint)
    upgrade_checkpoint(data)
    out = args.output
    if args.in_place:
        shutil.copy2(args.checkpoint, args.checkpoint + BACKUP_SUFFIX)
        out = args.checkpoint
    tmp = out + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, out)
    print(f"Wrote I/O v3 checkpoint {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
