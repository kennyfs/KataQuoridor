#!/usr/bin/env python3
"""Create a random-initialized Quoridor Four-at-a-Table (Q4) model of a preset, save it as a checkpoint and export it
(.bin.gz) from that checkpoint, so that the exported weights are exactly the ones of the PyTorch model.

Usage (from python/):
    python q4/make_random_model.py <preset> <export_dir> [--model-name NAME] [--seed SEED] [--scale-heads]

Presets: b1c32_q4 (tests), b2c64_q4, tf2_b4c192_q4. --scale-heads multiplies the output-head weights so that the
outputs are far from uniform (what the parity tests use); a plain random init gives nearly uniform outputs.
Writes <export_dir>/<name>.ckpt and <export_dir>/<name>/model.bin.gz, and prints the path of the latter.
"""

import argparse
import os
import subprocess
import sys

import torch

# python/ on the path, whatever the working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_random_model(preset: str, seed: int = 42, scale_heads: bool = False) -> Model:
    """A randomly initialized PyTorch model of a Q4 preset, in eval mode."""
    if preset not in modelconfigs.config_of_name:
        raise ValueError(f"Unknown preset: {preset}. Q4 presets: b1c32_q4, b2c64_q4, tf2_b4c192_q4")
    config = modelconfigs.config_of_name[preset]
    if not modelconfigs.is_quoridor4(config):
        raise ValueError(f"Preset {preset} is not a Q4 config (game != quoridor4)")
    torch.manual_seed(seed)
    model = Model(config, pos_len=11)
    model.initialize()
    if scale_heads:
        with torch.no_grad():
            model.policy_head.conv2p.weight.mul_(10.0)
            model.value_head.linear_value.weight.mul_(5.0)
            model.value_head.linear_misc.weight.mul_(3.0)
            model.value_head.conv_trajectory.weight.mul_(5.0)
    model.eval()
    return model


def export_model(model: Model, preset: str, export_dir: str, name: str) -> str:
    """Saves a checkpoint of model and exports it with export_model_pytorch.py; returns the .bin.gz path."""
    os.makedirs(export_dir, exist_ok=True)
    ckpt = os.path.join(export_dir, name + ".ckpt")
    torch.save({"model": model.state_dict(), "config": modelconfigs.config_of_name[preset]}, ckpt)
    proc = subprocess.run(
        [sys.executable, os.path.join(PYTHON_DIR, "export_model_pytorch.py"), "-checkpoint", ckpt,
         "-export-dir", os.path.join(export_dir, name), "-model-name", name, "-filename-prefix", "model"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"export failed:\n{(proc.stdout + proc.stderr)[-2000:]}")
    return os.path.join(export_dir, name, "model.bin.gz")


def export_random_model(preset: str, export_dir: str, model_name: str = None, seed: int = 42,
                        scale_heads: bool = False) -> str:
    name = model_name or preset
    return export_model(make_random_model(preset, seed, scale_heads), preset, export_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Export a random Q4 model of a preset")
    parser.add_argument("preset", type=str)
    parser.add_argument("export_dir", type=str)
    parser.add_argument("--model-name", type=str, default=None, help="defaults to the preset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-heads", action="store_true", help="scale the output heads up (non-uniform outputs)")
    args = parser.parse_args()
    print(export_random_model(args.preset, args.export_dir, args.model_name, args.seed, args.scale_heads))


if __name__ == "__main__":
    main()
