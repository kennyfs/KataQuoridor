#!/usr/bin/env python3
"""Held-out losses of an I/O v4 training checkpoint on sets of v4 npz files (docs/QuoridorIOv4.md, A/B).

Usage: quoridor_eval_sets_v4.py CKPT MAXROWS [-swa] NAME=GLOB_OR_DIR [NAME=...]

Prints one line per set: policy KL (p0loss - policy target entropy), top-1 accuracy, the value loss in the net's own
3-logit form (the draw as its own outcome; only comparable between nets trained that way) and with draws mapped back to 1/2
on both sides (`vhalf`, comparable between all v4 nets), the mean predicted / actual draw probability, lead and TD value
loss, in eval mode (running batch-norm statistics, like validation). -swa evaluates the SWA weights (what is exported).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from katago.train import load_model, data_processing_pytorch
from katago.train.metrics_pytorch import Metrics

args = [a for a in sys.argv[1:] if not a.startswith("-")]
use_swa = "-swa" in sys.argv
ckpt, maxrows = args[0], int(args[1])
dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model, swa, other = load_model.load_model(ckpt, use_swa=use_swa, device=dev, pos_len=9, verbose=False)
if use_swa:
    model = swa.module if hasattr(swa, "module") else swa
cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
model.eval()
metrics_obj = Metrics(1, model)
metrics_obj.quoridor_draw_target = "noresult"
keys = ["p0loss", "ptentr", "pacc1", "vloss", "vhalfloss", "pdraw", "tdraw", "leadloss", "tdvloss1", "p1loss", "smloss"]
for spec in args[2:]:
    name, pat = spec.split("=", 1)
    files = sorted(glob.glob(os.path.join(pat, "**", "*.npz"), recursive=True)) if os.path.isdir(pat) else sorted(glob.glob(pat))
    sums, n, wsum = {k: 0.0 for k in keys}, 0, 0.0
    with torch.no_grad():
        for batch in data_processing_pytorch.read_npz_training_data(files, 256, world_size=1, rank=0, pos_len=9, device=dev,
                randomize_symmetries=False, include_meta=False, model_config=cfg):
            out = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))
            m = metrics_obj.metrics_dict_batchwise(model, out, None, batch, is_training=False, soft_policy_weight_scale=8.0,
                    disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
                    td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0,
                    main_loss_scale=None, intermediate_loss_scale=None)
            for k in keys:
                sums[k] += float(m[k + "_sum"])
            n += 256
            if n >= maxrows:
                break
    r = {k: sums[k] / n for k in keys}
    r.update(name=name, rows=n, kl=r["p0loss"] - r["ptentr"], ckpt=os.path.basename(os.path.dirname(ckpt)), swa=use_swa)
    print(json.dumps(r), flush=True)
