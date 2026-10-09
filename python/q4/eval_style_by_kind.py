"""Style-policy loss of a Q4 I/O v2 net per kind of the seat to move (C61), with the real style features vs the
features zeroed (S3 of docs/q4/rounds/R7.md).

For every row of the given npz files (held-out population games) the style policy (policy variant 1: the move that
seat actually played) cross-entropy, as in the training loss (Metrics.loss_policy_player_samplewise), is accumulated
per C61 over the rows with a style-policy weight (C29 x C25 > 0), once with the net's `metadataInputNC` input and once
with that input zeroed. Prints one JSON line. No assertion: it reports what the net does.

    python q4/eval_style_by_kind.py -npz <dir or files> -checkpoint model.ckpt [-use-swa]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import data_processing_pytorch as dp
from katago.train.load_model import load_checkpoint, load_model
from katago.train.metrics_pytorch import Metrics

KIND_NAMES = ["learner", "weak", "snapshot", "greedy", "randomPawn", "basher", "grudge"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-npz", nargs="+", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-use-swa", action="store_true")
    parser.add_argument("-batch-size", type=int, default=64)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = []
    for p in args.npz:
        files += sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)) if os.path.isdir(p) else [p]
    assert files, "no npz files"
    has_swa = "swa_model" in load_checkpoint(args.checkpoint)
    model, swa_model, _ = load_model(args.checkpoint, use_swa=args.use_swa and has_swa, device=args.device)
    if args.use_swa and swa_model is not None:
        model = swa_model
    assert model.get_has_metadata_encoder(), "this is for Q4 I/O v2 nets (-meta presets)"
    model.eval()
    metrics = Metrics(1, model)

    sums = {k: {"weight": 0.0, "real": 0.0, "zeroed": 0.0, "rows": 0} for k in range(7)}
    with torch.no_grad():
        for batch in dp.read_npz_training_data(files, args.batch_size, 1, 0, 11, args.device, False, True, model.config):
            gt = batch["globalTargetsNC"]
            n = gt.shape[0]
            weight = gt[:, 25] * gt[:, 29]
            kind = gt[:, 61].long()
            per = {}
            for name, meta in (("real", batch["metadataInputNC"]), ("zeroed", torch.zeros_like(batch["metadataInputNC"]))):
                out = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"], meta))
                logits, target = metrics.q4_policy_logits_and_target(out[0][0][:, 1], batch["policyTargetsNCMove"][:, 1], n)
                per[name] = metrics.loss_policy_player_samplewise(logits, target, gt[:, 29], gt[:, 25])
            for k in range(7):
                sel = (kind == k) & (weight > 0)
                if sel.any():
                    sums[k]["weight"] += float(weight[sel].sum())
                    sums[k]["rows"] += int(sel.sum())
                    for name in ("real", "zeroed"):
                        sums[k][name] += float(per[name][sel].sum())
    result = {}
    for k, s in sums.items():
        if s["rows"]:
            real, zeroed = s["real"] / s["weight"], s["zeroed"] / s["weight"]
            result[KIND_NAMES[k]] = {"rows": s["rows"], "style_loss_real": real, "style_loss_zeroed": zeroed,
                                     "zeroed_minus_real": zeroed - real}
    total_w = sum(s["weight"] for s in sums.values())
    result["all"] = {"rows": sum(s["rows"] for s in sums.values()),
                     "style_loss_real": sum(s["real"] for s in sums.values()) / total_w,
                     "style_loss_zeroed": sum(s["zeroed"] for s in sums.values()) / total_w}
    result["all"]["zeroed_minus_real"] = result["all"]["style_loss_zeroed"] - result["all"]["style_loss_real"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
