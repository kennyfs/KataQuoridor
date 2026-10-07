"""Per-term Q4 training losses of a net on Q4 npz files (no training).

Evaluates a checkpoint (its SWA weights with -use-swa) or a randomly initialized model of a preset on every whole
batch of the given npz files with the training loss (Metrics.metrics_dict_batchwise, symmetry 0) and prints each
term per unit of row weight (the *_sum metrics divided by wsum), as the training log does. Used for the R5 smoke run
to show the losses before and after a training step.

    python q4/eval_losses.py -npz <dir or files> (-checkpoint model.ckpt [-use-swa] | -random-preset b2c64_q4)
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.train import modelconfigs
from katago.train import data_processing_pytorch as dp
from katago.train.load_model import load_checkpoint, load_model
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-npz", nargs="+", required=True, help="npz files or directories (searched recursively)")
    parser.add_argument("-checkpoint", help="training checkpoint")
    parser.add_argument("-use-swa", action="store_true", help="evaluate the checkpoint's SWA weights")
    parser.add_argument("-random-preset", help="evaluate a randomly initialized model of this preset instead")
    parser.add_argument("-seed", type=int, default=0, help="seed of the random model")
    parser.add_argument("-batch-size", type=int, default=64)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = []
    for p in args.npz:
        files += sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)) if os.path.isdir(p) else [p]
    assert files, "no npz files"

    if args.checkpoint:
        has_swa = "swa_model" in load_checkpoint(args.checkpoint)
        model, swa_model, _ = load_model(args.checkpoint, use_swa=args.use_swa and has_swa, device=args.device)
        if args.use_swa and swa_model is not None:
            model = swa_model
    else:
        torch.manual_seed(args.seed)
        model = Model(modelconfigs.config_of_name[args.random_preset], pos_len=11)
        model.initialize()
        model.to(args.device)
    assert modelconfigs.is_quoridor4(model.config)
    model.eval()
    metrics = Metrics(1, model)

    sums = {}
    with torch.no_grad():
        for batch in dp.read_npz_training_data(files, args.batch_size, 1, 0, 11, args.device, False, False, model.config):
            out = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))
            r = metrics.metrics_dict_batchwise(model, out, None, batch, False, 1.0, False, False, 1.0, [1, 1, 1],
                                               1.0, 1.0, None, None, include_model_norms=False)
            for k, v in r.items():
                if k.endswith("_sum") or k == "wsum":
                    sums[k] = sums.get(k, 0.0) + float(v)
    wsum = sums.pop("wsum")
    result = {k[:-4]: v / wsum for k, v in sums.items()}
    result["rows_weight"] = wsum
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
