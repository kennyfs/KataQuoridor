#!/usr/bin/env python3
"""Migrate a Quoridor checkpoint across the S8-S11 fix.

Before the fix, training data stored spatial inputs 8..11 (continuous BFS distances d/32)
truncated to binary, so nets were trained on (nearly) all-zero channels 8..11 while inference fed
the real distances. This zeroes the first-layer weights for those input channels
(conv_spatial.weight[:, 8:12]) in the model and, if present, the SWA model, plus the matching
optimizer state slices (momentum / Adam moments), so the migrated net ignores channels 8..11 just
as it effectively did during training, and then learns to use the real distances.
Lookahead slow weights are not stored in checkpoints (train.py re-initializes them from the
parameters on load), so they need no migration.

Idempotent: a marker is recorded in train_state (which train.py and clean_checkpoint.py carry
forward), and an already-migrated checkpoint is left untouched, even after further training.

Usage:
  # Training checkpoint:
  python quoridor_migrate_dist_planes.py -checkpoint BASEDIR/train/NAME/checkpoint.ckpt
  # Exported model directory (models/X with model.ckpt + model.bin.gz): migrate model.ckpt and
  # re-export model.bin.gz under the same model name:
  python quoridor_migrate_dist_planes.py -export-model-dir BASEDIR/models/X
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile

import torch

import katago.train.load_model

DIST_LO = 8
DIST_HI = 12
PARAM_SUFFIX = "conv_spatial.weight"
MARKER = "s8_s11_dist_planes_migrated"
BACKUP_SUFFIX = ".pre_distplanes"


def find_param_keys(model_state: dict) -> list:
    keys = [k for k in model_state if k == PARAM_SUFFIX or k.endswith("." + PARAM_SUFFIX)]
    assert len(keys) == 1, f"expected exactly one *{PARAM_SUFFIX} in state dict, found {keys}"
    return keys


def zero_slice(t: torch.Tensor) -> None:
    assert t.dim() == 4 and t.shape[1] >= DIST_HI, f"unexpected conv_spatial.weight shape {tuple(t.shape)}"
    t[:, DIST_LO:DIST_HI].zero_()


def migrate_state_dict(data: dict) -> list:
    """Migrates a loaded checkpoint dict in place. Returns a list of descriptions of what was zeroed,
    or [] if the checkpoint was already migrated."""
    train_state = data.setdefault("train_state", {})
    if train_state.get(MARKER, False):
        return []

    done = []
    (key,) = find_param_keys(data["model"])
    weight = data["model"][key]
    zero_slice(weight)
    done.append(f"model[{key}][:, {DIST_LO}:{DIST_HI}]")
    shape = tuple(weight.shape)

    if data.get("swa_model") is not None:
        (skey,) = find_param_keys(data["swa_model"])
        assert tuple(data["swa_model"][skey].shape) == shape
        zero_slice(data["swa_model"][skey])
        done.append(f"swa_model[{skey}][:, {DIST_LO}:{DIST_HI}]")

    # Optimizer state is keyed by parameter index, not name. conv_spatial.weight is the only
    # parameter with this shape (it is the only one with C_bin input channels), so find its
    # state by shape, and require that the match is unique.
    opt = data.get("optimizer")
    if opt is not None and "state" in opt:
        matches = []
        for pid, st in opt["state"].items():
            same = [n for n, v in st.items() if isinstance(v, torch.Tensor) and tuple(v.shape) == shape]
            if same:
                matches.append((pid, same))
        assert len(matches) <= 1, f"ambiguous optimizer state entries with shape {shape}: {matches}"
        for pid, names in matches:
            for n in names:
                zero_slice(opt["state"][pid][n])
                done.append(f"optimizer.state[{pid}][{n}][:, {DIST_LO}:{DIST_HI}]")

    train_state[MARKER] = True
    return done


def migrate_file(path: str) -> bool:
    data = katago.train.load_model.load_checkpoint(path)
    done = migrate_state_dict(data)
    if not done:
        print(f"{path}: already migrated, leaving unchanged")
        return False
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")
    tmp = path + ".tmp_distplanes"
    torch.save(data, tmp)
    os.replace(tmp, path)
    for d in done:
        print(f"  zeroed {d}")
    print(f"{path}: migrated")
    return True


def read_model_name(bin_gz: str) -> str:
    with gzip.open(bin_gz, "rb") as f:
        return f.readline().decode("utf-8").strip()


def migrate_export_dir(model_dir: str) -> None:
    ckpt = os.path.join(model_dir, "model.ckpt")
    bin_gz = os.path.join(model_dir, "model.bin.gz")
    assert os.path.isfile(ckpt), f"missing {ckpt}"
    assert os.path.isfile(bin_gz), f"missing {bin_gz}"
    migrate_file(ckpt)
    # The backup of model.bin.gz is written only when the re-export completes, so its absence
    # means model.bin.gz still holds the unmigrated net (also after an interrupted run).
    if os.path.exists(bin_gz + BACKUP_SUFFIX):
        print(f"{bin_gz}: already re-exported, leaving unchanged")
        return
    name = read_model_name(bin_gz)
    has_swa = katago.train.load_model.load_checkpoint(ckpt).get("swa_model") is not None
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(dir=model_dir) as tmpdir:
        cmd = [sys.executable, os.path.join(here, "export_model_pytorch.py"),
               "-checkpoint", ckpt, "-export-dir", tmpdir, "-model-name", name, "-filename-prefix", "model"]
        if has_swa:
            cmd.append("-use-swa")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=here)
        new_bin_gz = os.path.join(tmpdir, "model.bin.gz")
        if not os.path.exists(new_bin_gz):
            with open(os.path.join(tmpdir, "model.bin"), "rb") as fin, gzip.open(new_bin_gz, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        assert read_model_name(new_bin_gz) == name
        backup = bin_gz + BACKUP_SUFFIX
        shutil.copy2(bin_gz, backup + ".tmp")
        os.replace(new_bin_gz, bin_gz)
        os.replace(backup + ".tmp", backup)
        print(f"Backup: {backup}")
    print(f"Re-exported {bin_gz} (model name {name}, swa={has_swa})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-checkpoint", help="training checkpoint, e.g. BASEDIR/train/NAME/checkpoint.ckpt")
    group.add_argument("-export-model-dir", help="exported model dir containing model.ckpt and model.bin.gz")
    args = parser.parse_args()
    if args.checkpoint:
        migrate_file(args.checkpoint)
    else:
        migrate_export_dir(args.export_model_dir)


if __name__ == "__main__":
    main()
