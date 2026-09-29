#!/usr/bin/env python3
"""Static analysis of a KataQuoridor network, for the model viewer (index.html in this directory).

Examples:
    python3 python/model_viewer/analyze_model.py -model-dir ~/q0_run/models/run1-s4535552-d1215125
    python3 python/model_viewer/analyze_model.py -model-dir ... -selfplay-dir ~/q0_run/selfplay/run1-s4535552-d1215125 \
        -tdata-rows 8192 -mcts-positions 300 -visits 600
    python3 python/model_viewer/analyze_model.py -model-dir ... -mcts-positions 0     # skip the C++ search part

It writes data/<model name>.json next to this script (and updates data/index.json); open it with
    python3 python/model_viewer/serve.py

What it computes
  * architecture: the module tree with shapes / parameter counts, and every parameter (int8-quantized) with
    statistics, a histogram and its singular values;
  * activations: trunk / residual RMS per block and sub-block on real positions; attention head statistics;
  * feature importance on selfplay training rows (tdata, with their 600-visit search policy targets and game
    outcomes): permutation importance (single features and groups, overall and by game phase), gradient x input,
    and input-scaled first-layer weight norms;
  * the train/inference input mismatch of spatial channels 8-11 (see quoridor_state.CONTINUOUS_SPATIAL);
  * value calibration against game outcomes;
  * C++ search (GTP kata-search_analyze) on positions sampled from selfplay .sgfs: raw net vs search, with a
    position browser, per-position attributions and attention maps.
"""
import argparse
import base64
import datetime
import glob
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

# The flex-attention path cannot return attention weights and needs compilation; the plain path is identical.
os.environ.setdefault("KATAGO_FLEX_ATTENTION", "0")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from katago.train.load_model import load_model  # noqa: E402
from katago.train import model_pytorch  # noqa: E402
import quoridor_state as qs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
N = qs.N

POLICY_VARIANTS = ["policy", "opp reply", "soft policy", "soft opp reply", "long-term optimistic", "short-term optimistic"]
PHASES = [  # by number of walls already placed (both players)
    ("Opening", 0, 3),
    ("Early middle", 4, 9),
    ("Late middle", 10, 15),
    ("Endgame", 16, 20),
]

GROUPS = [
    ("Pawns", [1, 2], []),
    ("Blocked edges", [3, 4, 5, 6], []),
    ("Distance fields", [8, 9, 10, 11], []),
    ("Shortest-path masks", [12, 13], []),
    ("Wall anchors", [14, 15], []),
    ("My fences", [], [1, 3, 4, 5, 6]),
    ("Opp fences", [], [2, 7, 8, 9, 10, 11]),
    ("Shortest distances", [], [13, 14]),
    ("All fence info", [], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("All path / distance info", [8, 9, 10, 11, 12, 13], [13, 14]),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def r4(x):
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, np.ndarray):
        return [r4(v) for v in x.tolist()]
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.5g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ------------------------------------------------------------------------------------------------
# Model loading and forward helpers
# ------------------------------------------------------------------------------------------------
def load_net(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    has_swa = "swa_model" in sd
    train_state = sd.get("train_state", {})
    del sd
    model, swa, other = load_model(ckpt, use_swa=has_swa, device=device)
    net = swa if swa is not None else model
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, has_swa, train_state


def forward(net, sp, gl, device, bs=512, extra=None):
    """Runs the main heads on numpy inputs. Returns dict of numpy arrays."""
    outs = {k: [] for k in ("logits", "value", "margin", "traj", "wall", "td", "misc")}
    for i in range(0, sp.shape[0], bs):
        s = torch.from_numpy(np.ascontiguousarray(sp[i:i + bs])).to(device)
        g = torch.from_numpy(np.ascontiguousarray(gl[i:i + bs])).to(device)
        with torch.no_grad():
            o = net(s, g, extra_outputs=extra)[0]
        outs["logits"].append(o[0].reshape(s.shape[0], 18, N * N).float().cpu().numpy())
        outs["value"].append(torch.softmax(o[1].float(), 1).cpu().numpy())
        outs["td"].append(torch.softmax(o[2].float(), 2).cpu().numpy())
        outs["margin"].append((o[4][:, 0].float() * net.scoremean_multiplier).cpu().numpy())
        outs["misc"].append(o[5].float().cpu().numpy())
        outs["traj"].append(torch.sigmoid(o[6].float()).cpu().numpy())
        outs["wall"].append(torch.sigmoid(o[7].float()).cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def main_log_policy(logits, variant=0):
    """Training-style log-softmax of one policy variant over the valid 243-slot action space."""
    lg = logits[:, variant * 3:(variant + 1) * 3].reshape(logits.shape[0], -1).astype(np.float64)
    lg = np.where(qs.VALID_POLICY_MASK[None], lg, -1e4)
    lg = lg - lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))


# ------------------------------------------------------------------------------------------------
# Architecture description
# ------------------------------------------------------------------------------------------------
class ShapeRecorder:
    def __init__(self, net):
        self.shapes = {}
        self.handles = []
        for name, m in net.named_modules():
            if name:
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            while isinstance(t, (tuple, list)):
                t = t[0]
            if isinstance(t, torch.Tensor) and name not in self.shapes:
                self.shapes[name] = list(t.shape[1:])
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def fmt_shape(shape):
    if shape is None:
        return None
    if len(shape) == 2 and shape[0] == N * N:  # sequence layout (S, C)
        return [shape[1], N, N]
    return shape


def build_arch(net, shapes):
    cfg = net.config
    pnames = dict(net.named_parameters())

    def params_under(prefix):
        return [n for n in pnames if n == prefix or n.startswith(prefix + ".")]

    def node(id, kind, title, sub="", prefix=None, params=None, shape=None, desc="", **kw):
        ps = params if params is not None else (params_under(prefix) if prefix else [])
        d = {"id": id, "kind": kind, "title": title, "sub": sub, "params": ps,
             "nparams": int(sum(pnames[p].numel() for p in ps)), "desc": desc}
        sh = shape if shape is not None else fmt_shape(shapes.get(prefix)) if prefix else None
        if sh is not None:
            d["shape"] = sh
        d.update(kw)
        return d

    C = cfg["trunk_num_channels"]
    act = cfg.get("activation", "relu")
    nodes = []
    nodes.append({"id": "inputs", "kind": "row", "children": [
        node("in_spatial", "input", "Spatial input", "17 planes", shape=[qs.NUM_SPATIAL, N, N],
             desc="17 binary / distance planes in the side-to-move's canonical orientation (my goal row on top).",
             features="spatial"),
        node("in_global", "input", "Global input", "15 scalars", shape=[qs.NUM_GLOBAL],
             desc="15 scalar features: side to move, fence counts and encodings, jump parity, shortest distances.",
             features="global"),
    ]})
    nodes.append({"id": "stem", "kind": "row", "children": [
        node("conv_spatial", "conv", "conv_spatial", f"3x3 conv 17 -> {C}", prefix="conv_spatial",
             desc="First layer on the spatial planes (no bias). Its weights show how each input plane is read."),
        node("linear_global", "linear", "linear_global", f"linear 15 -> {C}", prefix="linear_global",
             shape=[C], desc="Projects the global features to a per-channel bias broadcast over all 81 cells."),
    ]})
    nodes.append(node("embed_add", "add", "Embedding", "spatial + broadcast global", shape=[C, N, N],
                      desc="The trunk starts as conv_spatial(x) + linear_global(g)."))

    trunk = []
    for bi, (bname, bkind) in enumerate(cfg["block_kind"]):
        block = net.blocks[bi]
        prefix = f"blocks.{bi}"
        children = []
        for cname, child in block.named_children():
            cprefix = f"{prefix}.{cname}"
            if isinstance(child, torch.nn.ModuleList):
                for j, sub in enumerate(child):
                    sprefix = f"{cprefix}.{j}"
                    children.append(describe_sub(sub, sprefix, node, cfg, act))
            else:
                children.append(describe_sub(child, cprefix, node, cfg, act))
        blk = node(prefix, "block", f"Block {bi + 1}", bkind, prefix=prefix, shape=[C, N, N],
                   desc=block_desc(block, cfg), residual=True, children=children, block_index=bi)
        trunk.append(blk)
    nodes.append({"id": "trunk", "kind": "trunk", "children": trunk})
    nodes.append(node("norm_trunkfinal", "norm", "Trunk final norm + " + act, "fixup affine", prefix="norm_trunkfinal",
                      shape=[C, N, N], desc="Final per-channel affine (fixup) and activation; both heads read this."))

    ph = net.policy_head
    p1, g1 = ph.conv1p.out_channels, ph.conv1g.out_channels
    policy = [
        {"id": "ph_row1", "kind": "row", "children": [
            node("policy_head.conv1p", "conv", "conv1p", f"1x1 conv {C} -> {p1}", prefix="policy_head.conv1p"),
            node("policy_head.conv1g", "conv", "conv1g", f"1x1 conv {C} -> {g1}", prefix="policy_head.conv1g"),
        ]},
        node("policy_head.gpool", "pool", "biasg + " + act + " + global pool", f"{g1} -> {3 * g1}",
             params=params_under("policy_head.biasg"), shape=[3 * g1],
             desc="KataGPool: mean, mean x (sqrt(area)-14)/10, and max over the board."),
        node("policy_head.linear_g", "linear", "linear_g", f"{3 * g1} -> {p1}, added to conv1p", prefix="policy_head.linear_g",
             shape=[p1], desc="Global summary broadcast-added to every cell of the conv1p output."),
        node("policy_head.bias2", "norm", "bias2 + " + act, "", prefix="policy_head.bias2", shape=[p1, N, N]),
        node("policy_head.conv2p", "conv", "conv2p", f"1x1 conv {p1} -> {ph.conv2p.out_channels}", prefix="policy_head.conv2p",
             shape=[ph.conv2p.out_channels, N, N],
             desc="18 = 6 policy variants x 3 planes (pawn target 9x9, vertical-wall anchor 8x8, horizontal-wall anchor 8x8). "
                  "The C++ engine uses variant 0 (policy) and variant 5 (short-term optimistic)."),
        node("out_policy", "output", "Policy outputs", "6 x (pawn, V-wall, H-wall)", shape=[18, N, N],
             variants=POLICY_VARIANTS),
    ]
    vh = net.value_head
    v1, v2 = vh.conv1.out_channels, vh.linear2.out_features
    value = [
        node("value_head.conv1", "conv", "conv1 + bias1 + " + act, f"1x1 conv {C} -> {v1}",
             params=params_under("value_head.conv1") + params_under("value_head.bias1"), shape=[v1, N, N]),
        {"id": "vh_split", "kind": "row", "children": [
            {"id": "vh_scalar", "kind": "col", "children": [
                node("value_head.gpool", "pool", "Value global pool", f"{v1} -> {3 * v1}", shape=[3 * v1],
                     desc="KataValueHeadGPool: mean and two area-scaled means."),
                node("value_head.linear2", "linear", "linear2 + " + act, f"{3 * v1} -> {v2}", prefix="value_head.linear2"),
                {"id": "vh_outs", "kind": "row", "children": [
                    node("value_head.linear_value", "output", "Win / loss", "2 logits", prefix="value_head.linear_value"),
                    node("value_head.linear_td_value", "output", "TD value", "4 horizons x 2", prefix="value_head.linear_td_value"),
                    node("value_head.linear_game_margin", "output", "Game margin", "moves", prefix="value_head.linear_game_margin"),
                    node("value_head.linear_misc", "output", "Uncertainty", "stdev, st errors", prefix="value_head.linear_misc"),
                    node("value_head.linear_variance_time", "output", "Variance time", "1", prefix="value_head.linear_variance_time"),
                ]},
            ]},
            {"id": "vh_spatial", "kind": "col", "children": [
                node("value_head.conv_trajectory", "output", "Trajectory", f"1x1 conv {v1} -> 2", prefix="value_head.conv_trajectory",
                     desc="Per-cell probability that my / the opponent's pawn will visit the cell later in the game."),
                node("value_head.conv_wall_graph", "output", "Final walls", f"1x1 conv {v1} -> 2", prefix="value_head.conv_wall_graph",
                     desc="Per-anchor probability that a vertical / horizontal wall stands there at the end of the game."),
            ]},
        ]},
    ]
    nodes.append({"id": "heads", "kind": "heads", "children": [
        {"id": "policy_head", "kind": "head", "title": "Policy head", "nparams": sum(pnames[p].numel() for p in params_under("policy_head")),
         "children": policy},
        {"id": "value_head", "kind": "head", "title": "Value head", "nparams": sum(pnames[p].numel() for p in params_under("value_head")),
         "children": value},
    ]})
    return nodes


def block_desc(block, cfg):
    if isinstance(block, model_pytorch.NestedBottleneckTransformerBlock):
        return (f"Nested bottleneck: 1x1 conv {block.normactconvp.c_in} -> {block.normactconvp.c_out}, "
                f"{block.internal_length} x (self-attention + FFN) each with its own residual, "
                f"1x1 conv back to {block.normactconvq.c_out}; the result is added to the trunk.")
    return type(block).__name__


def describe_sub(m, prefix, node, cfg, act):
    if isinstance(m, model_pytorch.NormActConv):
        k = m.conv.kernel_size[0] if m.conv is not None else 1
        gamma = " (with gamma)" if m.norm.use_gamma else ""
        return node(prefix, "conv", f"Norm{gamma} + {act} + {k}x{k} conv", f"{m.c_in} -> {m.c_out}", prefix=prefix,
                    desc="Fixup affine (beta, and gamma where present), activation, then a 1x1 convolution (a per-cell linear map).")
    if isinstance(m, model_pytorch.TransformerAttentionBlock):
        rope = "learnable 2D RoPE" if getattr(m, "learnable_rope", False) else ("2D RoPE" if m.use_rope else "no RoPE")
        return node(prefix, "attn", "Self-attention", f"{m.num_heads} heads x {m.q_head_dim}, {rope}", prefix=prefix,
                    residual=True, heads=m.num_heads, attn_name=m.name,
                    desc=f"RMSNorm, Q/K/V projections {m.c_main} -> {m.num_heads}x{m.q_head_dim}, rotary position encoding with "
                         f"learned (omega_x, omega_y) frequencies per head, softmax attention over all 81 cells, "
                         f"output projection back to {m.c_main}. Residual.")
    if isinstance(m, model_pytorch.TransformerFFNBlock):
        kind = "SwiGLU FFN" if m.use_swiglu else "FFN"
        return node(prefix, "ffn", kind, f"{m.c_main} -> {m.ffn_dim} -> {m.c_main}", prefix=prefix, residual=True,
                    desc=f"RMSNorm, then {'silu(W1 x) * (Wg x)' if m.use_swiglu else 'act(W1 x)'}, then W2 back to {m.c_main}. "
                         "Applied to every cell independently. Residual.")
    return node(prefix, "generic", type(m).__name__, "", prefix=prefix)


def param_entry(name, p):
    w = p.detach().float().cpu().numpy()
    flat = w.reshape(-1)
    absmax = float(np.abs(flat).max()) if flat.size else 0.0
    e = {"shape": list(w.shape), "n": int(flat.size), "mean": r4(flat.mean()), "std": r4(flat.std()),
         "rms": r4(np.sqrt((flat ** 2).mean())), "absmax": r4(absmax)}
    if absmax > 0:
        hist, edges = np.histogram(flat, bins=48, range=(-absmax, absmax))
        e["hist"] = hist.tolist()
        e["q"] = b64(np.clip(np.round(flat / absmax * 127), -127, 127).astype(np.int8))
        e["qscale"] = absmax / 127.0
    if w.ndim >= 2 and min(w.shape[0], int(np.prod(w.shape[1:]))) > 1:
        m = w.reshape(w.shape[0], -1)
        sv = np.linalg.svd(m, compute_uv=False)
        pn = sv / sv.sum()
        e["sv"] = r4(sv[:128])
        e["erank"] = r4(float(np.exp(-(pn * np.log(pn + 1e-12)).sum())))
    return e


# ------------------------------------------------------------------------------------------------
# Training rows
# ------------------------------------------------------------------------------------------------
def load_tdata(tdata_dir, num_rows, seed):
    files = sorted(glob.glob(os.path.join(tdata_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {tdata_dir}")
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    parts, total = [], 0
    per_file = max(256, num_rows // max(1, min(len(files), 16)))
    for f in files:
        z = np.load(f)
        gt = z["globalTargetsNC"]
        ok = np.nonzero(gt[:, 25] > 0)[0]
        if len(ok) == 0:
            continue
        take = rng.choice(ok, size=min(per_file, len(ok)), replace=False)
        take.sort()
        b = np.unpackbits(z["binaryInputNCHWPacked"][take], axis=2)[:, :, :N * N]
        parts.append({
            "bin": b.reshape(len(take), qs.NUM_SPATIAL, N, N).astype(np.float32),
            "glob": z["globalInputNC"][take].astype(np.float32),
            "pol": z["policyTargetsNCMove"][take, 0].astype(np.float32),
            "gt": gt[take].astype(np.float32),
        })
        total += len(take)
        if total >= num_rows:
            break
    d = {k: np.concatenate([p[k] for p in parts])[:num_rows] for k in parts[0]}
    log(f"loaded {d['bin'].shape[0]} training rows from {tdata_dir}")
    return d


def phase_of_walls(walls_placed):
    out = np.zeros(len(walls_placed), dtype=np.int32)
    for i, (_, lo, hi) in enumerate(PHASES):
        out[(walls_placed >= lo) & (walls_placed <= hi)] = i
    return out


def row_losses(out, pol_target, gt):
    """Per-row training-style losses and weights."""
    logp = main_log_policy(out["logits"])
    tp = pol_target * qs.VALID_POLICY_MASK[None]
    tsum = tp.sum(1, keepdims=True)
    tp = tp / np.maximum(tsum, 1e-8)
    ce_pol = -(tp * logp).sum(1)
    w_pol = gt[:, 25] * gt[:, 26] * (tsum[:, 0] > 0)
    tv = gt[:, 0:2] / np.maximum(gt[:, 0:2].sum(1, keepdims=True), 1e-8)
    ce_val = -(tv * np.log(np.maximum(out["value"], 1e-12))).sum(1)
    w_val = gt[:, 25] * (1.0 - gt[:, 35])
    err = out["margin"] - gt[:, 21]
    hub = np.where(np.abs(err) <= 1.0, 0.5 * err ** 2, np.abs(err) - 0.5)
    w_mar = gt[:, 25] * gt[:, 29]
    return {"pol": ce_pol, "val": ce_val, "mar": hub, "w_pol": w_pol, "w_val": w_val, "w_mar": w_mar, "logp": logp}


def wmean(x, w, sel=None):
    if sel is not None:
        x, w = x[sel], w[sel]
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else None


def compare_outputs(base, other, lb, lo):
    """Per-row output differences between two runs of the net."""
    pb, po = np.exp(lb["logp"]), np.exp(lo["logp"])
    kl = (pb * (lb["logp"] - lo["logp"])).sum(1)
    return {
        "kl": np.maximum(kl, 0.0),
        "dwin": np.abs(base["value"][:, 0] - other["value"][:, 0]),
        "dmargin": np.abs(base["margin"] - other["margin"]),
        "top1": (pb.argmax(1) != po.argmax(1)).astype(np.float32),
    }


def summarize_metrics(diff, lb, lo, phase):
    def agg(sel):
        d = {
            "kl": float(diff["kl"][sel].mean()) if sel.any() else None,
            "dwin": float(diff["dwin"][sel].mean()) if sel.any() else None,
            "dmargin": float(diff["dmargin"][sel].mean()) if sel.any() else None,
            "top1": float(diff["top1"][sel].mean()) if sel.any() else None,
        }
        for k in ("pol", "val", "mar"):
            a, b = wmean(lo[k], lo["w_" + k], sel), wmean(lb[k], lb["w_" + k], sel)
            d["d" + k] = (a - b) if a is not None and b is not None else None
        return {k: r4(v) for k, v in d.items()}
    allsel = np.ones(len(phase), dtype=bool)
    return {"all": agg(allsel), "phase": [agg(phase == i) for i in range(len(PHASES))]}


def feature_importance(net, device, sp, gl, pol_t, gt, phase, seed):
    rng = np.random.default_rng(seed + 1)
    base = forward(net, sp, gl, device)
    lb = row_losses(base, pol_t, gt)
    n = sp.shape[0]
    const_sp = [ch for ch in range(qs.NUM_SPATIAL) if np.all(sp[:, ch] == sp[0:1, ch])]
    const_gl = [i for i in range(qs.NUM_GLOBAL) if np.all(gl[:, i] == gl[0, i])]

    def run(sp_chs, gl_idx, mode):
        s2, g2 = sp.copy(), gl.copy()
        if mode == "zero":
            s2[:, sp_chs] = 0.0
            g2[:, gl_idx] = 0.0
        else:
            perm = rng.permutation(n)
            s2[:, sp_chs] = sp[perm][:, sp_chs]
            g2[:, gl_idx] = gl[perm][:, gl_idx]
        o = forward(net, s2, g2, device)
        lo = row_losses(o, pol_t, gt)
        return summarize_metrics(compare_outputs(base, o, lb, lo), lb, lo, phase)

    res = {"spatial": [], "global": [], "groups": [], "constant_spatial": const_sp, "constant_global": const_gl}
    for ch in range(1, qs.NUM_SPATIAL):
        mode = "zero" if ch in const_sp else "permute"
        res["spatial"].append({"idx": ch, "mode": mode, **run([ch], [], mode)})
    for i in range(qs.NUM_GLOBAL):
        mode = "zero" if i in const_gl else "permute"
        res["global"].append({"idx": i, "mode": mode, **run([], [i], mode)})
    for name, schs, gidx in GROUPS:
        res["groups"].append({"name": name, "spatial": schs, "global": gidx, "mode": "permute", **run(schs, gidx, "permute")})
    log("permutation importance done")
    return base, lb, res


def grad_x_input(net, device, sp, gl, bs=256):
    """Mean |grad x (x - mean)| per channel for the value logit difference and the chosen-move log-prob."""
    mean_sp = sp.mean(0, keepdims=True)
    mean_gl = gl.mean(0, keepdims=True)
    acc = {k: (np.zeros(qs.NUM_SPATIAL), np.zeros(qs.NUM_GLOBAL)) for k in ("value", "policy", "margin")}
    signed = {k: (np.zeros(qs.NUM_SPATIAL), np.zeros(qs.NUM_GLOBAL)) for k in ("value", "policy", "margin")}
    valid = torch.tensor(qs.VALID_POLICY_MASK, device=device)
    for i in range(0, sp.shape[0], bs):
        for key in ("value", "policy", "margin"):
            s = torch.tensor(sp[i:i + bs], device=device, requires_grad=True)
            g = torch.tensor(gl[i:i + bs], device=device, requires_grad=True)
            o = net(s, g)[0]
            if key == "value":
                obj = (o[1][:, 0] - o[1][:, 1]).sum()
            elif key == "margin":
                obj = o[4][:, 0].sum()
            else:
                lg = o[0][:, 0:3].reshape(s.shape[0], -1).masked_fill(~valid, -1e4)
                lp = torch.log_softmax(lg, 1)
                obj = lp.gather(1, lp.argmax(1, keepdim=True)).sum()
            gs, gg = torch.autograd.grad(obj, (s, g))
            a_s = (gs.cpu().numpy() * (sp[i:i + bs] - mean_sp))
            a_g = (gg.cpu().numpy() * (gl[i:i + bs] - mean_gl))
            acc[key][0][:] += np.abs(a_s).sum((2, 3)).sum(0)
            acc[key][1][:] += np.abs(a_g).sum(0)
            signed[key][0][:] += a_s.sum((2, 3)).sum(0)
            signed[key][1][:] += a_g.sum(0)
    n = sp.shape[0]
    return {k: {"spatial": r4(acc[k][0] / n), "global": r4(acc[k][1] / n),
                "spatial_signed": r4(signed[k][0] / n), "global_signed": r4(signed[k][1] / n)} for k in acc}


def stem_weights(net, sp, gl):
    w = net.conv_spatial.weight.detach().float().cpu().numpy()  # (C, 17, 3, 3)
    wg = net.linear_global.weight.detach().float().cpu().numpy()  # (C, 15)
    rms_sp = np.sqrt((sp ** 2).mean((0, 2, 3)))
    rms_gl = np.sqrt((gl ** 2).mean(0))
    foot = np.sqrt((w ** 2).sum(0))  # (17, 3, 3)
    return {
        "spatial_norm": r4(np.sqrt((w ** 2).sum((0, 2, 3)))),
        "spatial_scaled": r4(np.sqrt((w ** 2).sum((0, 2, 3))) * rms_sp),
        "spatial_rms": r4(rms_sp),
        "spatial_footprint": r4(foot.reshape(qs.NUM_SPATIAL, 9)),
        "global_norm": r4(np.sqrt((wg ** 2).sum(0))),
        "global_scaled": r4(np.sqrt((wg ** 2).sum(0)) * rms_gl),
        "global_rms": r4(rms_gl),
        "global_mean": r4(gl.mean(0)),
        "spatial_mean": r4(sp.mean((0, 2, 3))),
    }


def calibration(out, gt, bins=10):
    w = gt[:, 25] * (1.0 - gt[:, 35])
    pw = out["value"][:, 0]
    tv = gt[:, 0] / np.maximum(gt[:, 0:2].sum(1), 1e-8)
    edges = np.linspace(0, 1, bins + 1)
    res = []
    for i in range(bins):
        sel = (pw >= edges[i]) & (pw < edges[i + 1] if i < bins - 1 else pw <= 1.0) & (w > 0)
        if sel.sum() == 0:
            res.append({"lo": r4(edges[i]), "hi": r4(edges[i + 1]), "n": 0})
            continue
        res.append({"lo": r4(edges[i]), "hi": r4(edges[i + 1]), "n": int(sel.sum()),
                    "pred": r4(pw[sel].mean()), "actual": r4(tv[sel].mean())})
    wm = gt[:, 29] * gt[:, 25] > 0
    mbins = []
    mp, mt = out["margin"][wm], gt[wm, 21]
    for lo in range(-14, 14, 2):
        sel = (mp >= lo) & (mp < lo + 2)
        if sel.sum() >= 5:
            mbins.append({"lo": lo, "hi": lo + 2, "n": int(sel.sum()), "pred": r4(mp[sel].mean()),
                          "actual": r4(mt[sel].mean()), "p10": r4(np.percentile(mt[sel], 10)), "p90": r4(np.percentile(mt[sel], 90))})
    brier = float(((pw - tv) ** 2 * w).sum() / max(w.sum(), 1e-8))
    return {"value": res, "margin": mbins, "brier": r4(brier),
            "margin_mae": r4(float(np.abs(mp - mt).mean())) if len(mp) else None}


# ------------------------------------------------------------------------------------------------
# Activations & attention statistics
# ------------------------------------------------------------------------------------------------
def rms_onboard(t):
    t = t.float()
    if t.dim() == 3:  # N S C
        return torch.sqrt((t ** 2).mean(dim=(1, 2)))
    return torch.sqrt((t ** 2).mean(dim=(1, 2, 3)))


def activation_stats(net, device, sp, gl):
    rec = {}
    handles = []

    def pre(name):
        def f(m, args, kwargs=None):
            rec.setdefault(name + ":in", []).append(rms_onboard(args[0]).cpu())
        return f

    def post(name):
        def f(m, args, out):
            rec.setdefault(name + ":out", []).append(rms_onboard(out).cpu())
        return f

    names = []
    for bi, block in enumerate(net.blocks):
        names.append(f"blocks.{bi}")
        handles.append(block.register_forward_pre_hook(pre(f"blocks.{bi}")))
        handles.append(block.register_forward_hook(post(f"blocks.{bi}")))
        if hasattr(block, "blockstack"):
            for j, sub in enumerate(block.blockstack):
                nm = f"blocks.{bi}.blockstack.{j}"
                names.append(nm)
                handles.append(sub.register_forward_pre_hook(pre(nm)))
                handles.append(sub.register_forward_hook(post(nm)))
    handles.append(net.norm_trunkfinal.register_forward_pre_hook(pre("norm_trunkfinal")))
    forward(net, sp, gl, device, bs=256)
    for h in handles:
        h.remove()
    out = {}
    for nm in names + ["norm_trunkfinal"]:
        d = {}
        if nm + ":in" in rec:
            d["in"] = r4(float(torch.cat(rec[nm + ":in"]).mean()))
        if nm + ":out" in rec:
            d["out"] = r4(float(torch.cat(rec[nm + ":out"]).mean()))
        out[nm] = d
    return out


def attention_modules(net):
    return [(n, m) for n, m in net.named_modules() if isinstance(m, model_pytorch.TransformerAttentionBlock)]


def capture_attention(net, device, sp, gl):
    mods = attention_modules(net)
    if not mods:
        return mods, None
    eo = model_pytorch.ExtraOutputs([m.name + ".attn_weights" for _, m in mods])
    s = torch.from_numpy(sp).to(device)
    g = torch.from_numpy(gl).to(device)
    with torch.no_grad():
        net(s, g, extra_outputs=eo)
    maps = [eo.returned[m.name + ".attn_weights"].float().cpu().numpy() for _, m in mods]  # each (B, H, S, S)
    return mods, maps


def attention_head_stats(net, device, sp, gl, bs=128):
    mods = attention_modules(net)
    if not mods:
        return None
    idx = np.arange(N * N)
    rr, cc = idx // N, idx % N
    dist = np.abs(rr[:, None] - rr[None]) + np.abs(cc[:, None] - cc[None])  # Manhattan cell distance
    sums = None
    count = 0
    for i in range(0, sp.shape[0], bs):
        mods, maps = capture_attention(net, device, sp[i:i + bs], gl[i:i + bs])
        me = sp[i:i + bs, 1].reshape(-1, N * N)
        op = sp[i:i + bs, 2].reshape(-1, N * N)
        pathme = sp[i:i + bs, 12].reshape(-1, N * N)
        pathop = sp[i:i + bs, 13].reshape(-1, N * N)
        rows = []
        for a in maps:  # (B, H, S, S)
            ent = -(a * np.log(a + 1e-12)).sum(-1).mean(-1)  # (B, H)
            md = (a * dist[None, None]).sum(-1).mean(-1)
            selfw = np.einsum("bhii->bh", a) / (N * N)
            tome = np.einsum("bhqk,bk->bh", a, me) / (N * N)
            toop = np.einsum("bhqk,bk->bh", a, op) / (N * N)
            topm = np.einsum("bhqk,bk->bh", a, pathme) / (N * N)
            topo = np.einsum("bhqk,bk->bh", a, pathop) / (N * N)
            rows.append(np.stack([md, ent, selfw, tome, toop, topm, topo], -1).sum(0))  # (H, 7)
        rows = np.stack(rows)  # (L, H, 7)
        sums = rows if sums is None else sums + rows
        count += sp[i:i + bs].shape[0]
    avg = sums / count
    keys = ["mean_dist", "entropy", "self", "to_my_pawn", "to_opp_pawn", "to_my_path", "to_opp_path"]
    return {"layers": [m.name for _, m in mods],
            "stats": [[{k: r4(avg[l, h, j]) for j, k in enumerate(keys)} for h in range(avg.shape[1])] for l in range(avg.shape[0])],
            "uniform_entropy": r4(math.log(N * N))}


# ------------------------------------------------------------------------------------------------
# GTP search on sampled selfplay positions
# ------------------------------------------------------------------------------------------------
class GTP:
    def __init__(self, katago, config, model, overrides):
        args = [katago, "gtp", "-config", config, "-model", model, "-override-config", overrides]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def cmd(self, c):
        self.p.stdin.write(c + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError(f"katago exited during: {c}")
            line = line.rstrip("\n")
            if line.strip() == "":
                if out:
                    break
                continue
            out.append(line)
        if out and out[0].startswith("?"):
            raise RuntimeError(f"GTP error for '{c}': {out}")
        if out:
            out[0] = out[0][1:].strip()
        return [l for l in out if l != ""]

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def parse_raw_nn(lines):
    kv, pol, mode = {}, [], None
    for l in lines:
        t = l.split()
        if not t:
            continue
        if t[0] in ("policy", "whiteOwnership", "policyPass", "symmetry"):
            mode = t[0]
            continue
        if mode == "policy":
            pol.append([float(v) if v != "NAN" else np.nan for v in t])
        elif len(t) == 2 and mode != "whiteOwnership":
            try:
                kv[t[0]] = float(t[1])
            except ValueError:
                pass
    return kv, np.array(pol)


def parse_analyze(lines):
    moves, root, play = [], {}, None
    for l in lines:
        if l.startswith("play "):
            play = l.split()[1]
            continue
        segs = l.replace("rootInfo", "info rootInfo").split("info ")
        for seg in segs:
            t = seg.split()
            if not t:
                continue
            if t[0] == "rootInfo":
                t = t[1:]
                d = root
            elif t[0] == "move":
                d = {"move": t[1]}
                t = t[2:]
                moves.append(d)
            else:
                continue
            k = 0
            while k < len(t) - 1:
                key = t[k]
                if key == "pv":
                    d["pv"] = t[k + 1:]
                    break
                try:
                    d[key] = float(t[k + 1])
                except ValueError:
                    pass
                k += 2
    return moves, root, play


def sample_positions(selfplay_dir, num, seed):
    files = sorted(glob.glob(os.path.join(selfplay_dir, "sgfs", "*.sgfs")))
    if not files:
        files = sorted(glob.glob(os.path.join(selfplay_dir, "**", "*.sgfs"), recursive=True))
    rng = random.Random(seed)
    lines = []
    for f in files:
        with open(f) as fh:
            for l in fh:
                if l.startswith("(;"):
                    lines.append((os.path.basename(f), l))
    rng.shuffle(lines)
    picks = []
    for fname, l in lines:
        g = qs.parse_sgfs_line(l)
        if len(g["moves"]) < 6 or g["result"][:1] not in ("B", "W"):
            continue
        ply = rng.randrange(0, len(g["moves"]))
        picks.append((fname, g, ply))
        if len(picks) >= num:
            break
    log(f"sampled {len(picks)} positions from {len(lines)} games in {selfplay_dir}")
    return picks


def state_at(g, ply):
    st = qs.QState()
    st.pawn = dict(g["start"])
    for (pla, mv, _) in g["moves"][:ply]:
        st.play(pla, mv)
    st.to_move = g["moves"][ply][0] if ply < len(g["moves"]) else st.to_move
    return st


def state_json(st):
    return {"B": list(st.pawn["B"]), "W": list(st.pawn["W"]),
            "v": [[int(c), int(r)] for c, r in zip(*np.nonzero(st.vwalls))],
            "h": [[int(c), int(r)] for c, r in zip(*np.nonzero(st.hwalls))],
            "fB": st.fences["B"], "fW": st.fences["W"], "toMove": st.to_move}


def board_policy_arrays(probs_by_move):
    """dict move->prob -> compact board-space arrays: pawn [r][c] 81, v [r][c] 64, h [r][c] 64 (None where absent)."""
    pawn, v, h = [None] * 81, [None] * 64, [None] * 64
    for mvs, p in probs_by_move.items():
        kind, c, r = qs.parse_move_str(mvs)
        if kind == "pawn":
            pawn[r * 9 + c] = r4(p)
        elif kind == "vwall":
            v[r * 8 + c] = r4(p)
        else:
            h[r * 8 + c] = r4(p)
    return {"pawn": pawn, "v": v, "h": h}


def canon_to_board_map(arr81, pla, is_wall=False):
    """Canonical [rCanon][c] (9x9) map -> board [r][c] list."""
    a = np.asarray(arr81).reshape(N, N)
    if pla == "W":
        a = np.concatenate([a[:8][::-1], a[8:9]]) if is_wall else a[::-1]
    return a


def position_attribution(net, device, s, g):
    """Per-channel and per-cell grad x input for value, margin and the top policy move of one position."""
    res = {}
    valid = torch.tensor(qs.VALID_POLICY_MASK, device=device)
    for key in ("value", "policy"):
        st = torch.tensor(s[None], device=device, requires_grad=True)
        gt_ = torch.tensor(g[None], device=device, requires_grad=True)
        o = net(st, gt_)[0]
        if key == "value":
            obj = (o[1][:, 0] - o[1][:, 1]).sum()
        else:
            lg = o[0][:, 0:3].reshape(1, -1).masked_fill(~valid, -1e4)
            lp = torch.log_softmax(lg, 1)
            obj = lp.max()
        gs, gg = torch.autograd.grad(obj, (st, gt_))
        a_s = gs[0].cpu().numpy() * s
        a_g = gg[0].cpu().numpy() * g
        res[key] = {"spatial": r4(a_s.sum((1, 2))), "global": r4(a_g), "cells": r4(a_s.sum(0).reshape(-1))}
    return res


def run_mcts(net, device, args, model_bin, picks):
    tmpdir = tempfile.mkdtemp(prefix="kq_modelviewer_gtp_")
    overrides = ",".join([f"logDir={tmpdir}", "logAllGTPCommunication=false", "logSearchInfo=false", "logToStderr=false",
                          f"maxVisits={args.visits}", f"numSearchThreads={args.search_threads}", "ponderingEnabled=false"])
    gtp = GTP(args.katago, args.gtp_config, model_bin, overrides)
    log(f"katago GTP started ({args.visits} visits per position)")
    positions = []
    t0 = time.time()
    try:
        for pi, (fname, g, ply) in enumerate(picks):
            st = state_at(g, ply)
            pla = st.to_move
            gtp.cmd("clear_board")
            gtp.cmd("clear_cache")
            for (p, mv) in st.history:
                gtp.cmd(f"play {p} {mv}")
            kv, grid = parse_raw_nn(gtp.cmd("kata-raw-nn 0"))
            moves, root, play = parse_analyze(gtp.cmd(f"kata-search_analyze {pla} rootInfo true"))
            legal, cpp_prob = [], {}
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    if not np.isnan(grid[y, x]):
                        mv = qs.sgf_point_to_move(chr(97 + x) + chr(97 + y))
                        ms = qs.move_str(*mv)
                        legal.append((ms, qs.move_to_policy_index(mv, pla)))
                        cpp_prob[ms] = float(grid[y, x])
            s, gv = qs.fill_row(st)
            sb = qs.binarize_like_training(s)
            o = forward(net, np.stack([s, sb]), np.stack([gv, gv]), device)
            idxs = np.array([i for _, i in legal])
            names = [m for m, _ in legal]
            pols = {}
            for k, name in ((0, "py"), (1, "pybin")):
                lg = o["logits"][k, 0:3].reshape(-1)[idxs]
                p = np.exp(lg - lg.max())
                pols[name] = p / p.sum()
            opt = o["logits"][0, 15:18].reshape(-1)[idxs]
            popt = np.exp(opt - opt.max()); popt /= popt.sum()
            cp = np.array([cpp_prob[m] for m in names])
            tot = sum(m.get("visits", 0) for m in moves)
            visits = {m["move"]: m.get("visits", 0) / max(tot, 1) for m in moves}
            mcts_vec = np.array([visits.get(m, 0.0) for m in names])
            best = play or (moves[0]["move"] if moves else None)

            def kl(p, q):
                return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())

            win_sign = 1 if pla == "W" else -1
            wwin_cpp = kv.get("whiteWin")
            nn_win = wwin_cpp if pla == "W" else 1 - wwin_cpp  # side to move
            mcts_win = root.get("winrate")
            ev = qs.parse_eval_comment(g["moves"][ply][2]) if ply < len(g["moves"]) else None
            rec = {
                "file": fname, "ply": ply, "nMoves": len(g["moves"]), "result": g["result"],
                "gtype": g["kv"].get("gtype", ""), "state": state_json(st),
                "played": qs.move_str(*g["moves"][ply][1]) if ply < len(g["moves"]) else None,
                "history": [m for _, m in st.history],
                "nn": {"win": r4(nn_win), "margin": r4(kv.get("whiteLead", 0) * win_sign),
                       "policy": board_policy_arrays(dict(zip(names, cp))),
                       "top": names[int(cp.argmax())], "wallMass": r4(float(sum(p for m, p in zip(names, cp) if m[-1] in "hv")))},
                "pybin": {"win": r4(o["value"][1, 0]), "margin": r4(o["margin"][1]),
                          "top": names[int(pols["pybin"].argmax())],
                          "policy": board_policy_arrays(dict(zip(names, pols["pybin"])))},
                "optimistic": {"top": names[int(popt.argmax())]},
                "mcts": {"win": r4(mcts_win), "margin": r4(root.get("scoreMean", 0.0) * 1.0) if root else None,
                         "visits": int(root.get("visits", tot)) if root else tot, "best": best,
                         "policy": board_policy_arrays(visits),
                         "moves": [{"move": m["move"], "visits": int(m.get("visits", 0)), "win": r4(m.get("winrate")),
                                    "margin": r4(m.get("scoreMean")), "prior": r4(m.get("prior")),
                                    "pv": m.get("pv", [])[:8]} for m in moves[:10]],
                         "wallShare": r4(float(sum(v for m, v in visits.items() if m[-1] in "hv")))},
                "check": {"py_vs_cpp_policy": r4(float(np.abs(pols["py"] - cp).max())),
                          "py_vs_cpp_win": r4(abs(float(o["value"][0, 0]) - nn_win)),
                          "pybin_vs_cpp_policy": r4(float(np.abs(pols["pybin"] - cp).max()))},
                "kl_mcts_nn": r4(kl(mcts_vec, cp)), "kl_mcts_pybin": r4(kl(mcts_vec, pols["pybin"])),
                "ce_mcts_nn": r4(float(-(mcts_vec * np.log(cp + 1e-12)).sum())),
                "ce_mcts_pybin": r4(float(-(mcts_vec * np.log(pols["pybin"] + 1e-12)).sum())),
                "prior_of_best": r4(cpp_prob.get(best, 0.0)),
                "traj": {"me": r4(canon_to_board_map(o["traj"][0, 0], pla).reshape(-1)),
                         "opp": r4(canon_to_board_map(o["traj"][0, 1], pla).reshape(-1))},
                "finalWalls": {"v": r4(canon_to_board_map(o["wall"][0, 0], pla, True)[:8, :8].reshape(-1)),
                               "h": r4(canon_to_board_map(o["wall"][0, 1], pla, True)[:8, :8].reshape(-1))},
                "attr": position_attribution(net, device, s, gv),
                "selfplayEval": ev and {"wWin": ev["wWin"], "wScore": ev["wScore"], "v": ev.get("v")},
                "walls": st.num_walls(),
            }
            for key in ("value", "policy"):
                rec["attr"][key]["cells"] = r4(canon_to_board_map(rec["attr"][key]["cells"], pla).reshape(-1))
            positions.append(rec)
            if (pi + 1) % 20 == 0 or pi + 1 == len(picks):
                log(f"  searched {pi + 1}/{len(picks)} positions ({time.time() - t0:.0f}s)")
    finally:
        gtp.close()
    return positions


def mcts_summary(positions):
    def agg(sel):
        ps = [p for p in positions if sel(p)]
        if not ps:
            return {"n": 0}
        return {
            "n": len(ps),
            "top1_nn": r4(np.mean([p["nn"]["top"] == p["mcts"]["best"] for p in ps])),
            "top1_pybin": r4(np.mean([p["pybin"]["top"] == p["mcts"]["best"] for p in ps])),
            "kl_nn": r4(np.mean([p["kl_mcts_nn"] for p in ps])),
            "kl_pybin": r4(np.mean([p["kl_mcts_pybin"] for p in ps])),
            "ce_nn": r4(np.mean([p["ce_mcts_nn"] for p in ps])),
            "ce_pybin": r4(np.mean([p["ce_mcts_pybin"] for p in ps])),
            "value_mae": r4(np.mean([abs(p["nn"]["win"] - p["mcts"]["win"]) for p in ps if p["mcts"]["win"] is not None])),
            "value_mae_pybin": r4(np.mean([abs(p["pybin"]["win"] - p["mcts"]["win"]) for p in ps if p["mcts"]["win"] is not None])),
            "margin_mae": r4(np.mean([abs(p["nn"]["margin"] - p["mcts"]["margin"]) for p in ps if p["mcts"]["margin"] is not None])),
            "wall_mass_nn": r4(np.mean([p["nn"]["wallMass"] for p in ps])),
            "wall_share_mcts": r4(np.mean([p["mcts"]["wallShare"] for p in ps])),
            "best_is_wall": r4(np.mean([p["mcts"]["best"] is not None and p["mcts"]["best"][-1] in "hv" for p in ps])),
            "nn_top_is_wall": r4(np.mean([p["nn"]["top"][-1] in "hv" for p in ps])),
            "prior_of_best": r4(np.mean([p["prior_of_best"] for p in ps])),
        }
    phase_idx = lambda p: int(phase_of_walls(np.array([p["walls"]]))[0])
    return {
        "all": agg(lambda p: True),
        "phase": [agg(lambda p, i=i: phase_idx(p) == i) for i in range(len(PHASES))],
        "max_check_policy": r4(max((p["check"]["py_vs_cpp_policy"] for p in positions), default=None)),
        "max_check_win": r4(max((p["check"]["py_vs_cpp_win"] for p in positions), default=None)),
        "mean_check_pybin_policy": r4(np.mean([p["check"]["pybin_vs_cpp_policy"] for p in positions])) if positions else None,
    }


def attention_examples(net, device, positions, k):
    if not positions:
        return None
    ex = positions[:k]
    sps, gls = [], []
    for p in ex:
        st = state_from_json(p["state"])
        s, g = qs.fill_row(st)
        sps.append(s)
        gls.append(g)
    mods, maps = capture_attention(net, device, np.stack(sps), np.stack(gls))
    if not mods:
        return None
    out = []
    for e in range(len(ex)):
        # sqrt-encoded uint8 so small weights keep resolution; decode p = (u / 255)^2. Canonical cell order.
        arr = np.stack([m[e] for m in maps])  # (L, H, S, S)
        out.append({"pos": e, "q": b64(np.round(np.sqrt(np.clip(arr, 0, 1)) * 255).astype(np.uint8)), "shape": list(arr.shape),
                    "spatial": r4(sps[e].reshape(qs.NUM_SPATIAL, -1)), "global": r4(gls[e])})
    return {"layers": [m.name for _, m in mods], "examples": out}


def state_from_json(j):
    st = qs.QState()
    st.pawn = {"B": tuple(j["B"]), "W": tuple(j["W"])}
    for c, r in j["v"]:
        st.vwalls[c, r] = True
    for c, r in j["h"]:
        st.hwalls[c, r] = True
    st.fences = {"B": j["fB"], "W": j["fW"]}
    st.to_move = j["toMove"]
    return st


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-model-dir", help="dir with model.ckpt (and model.bin.gz for the C++ search)")
    ap.add_argument("-checkpoint", help="pytorch checkpoint (overrides -model-dir/model.ckpt)")
    ap.add_argument("-model-bin", help="exported .bin.gz for the C++ search (default: -model-dir/model.bin.gz)")
    ap.add_argument("-name", help="name shown in the viewer (default: model dir name)")
    ap.add_argument("-selfplay-dir", help="selfplay dir with sgfs/ and tdata/ (default: ../../selfplay/<name> next to the model dir)")
    ap.add_argument("-tdata-dir", help="training .npz dir (default: <selfplay-dir>/tdata)")
    ap.add_argument("-tdata-rows", type=int, default=8192)
    ap.add_argument("-mcts-positions", type=int, default=200, help="0 to skip the C++ search")
    ap.add_argument("-visits", type=int, default=400)
    ap.add_argument("-search-threads", type=int, default=4)
    ap.add_argument("-attention-examples", type=int, default=12)
    ap.add_argument("-katago", default=os.path.join(REPO, "cpp", "katago"))
    ap.add_argument("-gtp-config", default=os.path.join(REPO, "cpp", "configs", "gtp_quoridor.cfg"))
    ap.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-seed", type=int, default=12345)
    ap.add_argument("-out", help="output json (default: data/<name>.json next to this script)")
    args = ap.parse_args()

    if not args.checkpoint and not args.model_dir:
        ap.error("need -model-dir or -checkpoint")
    ckpt = args.checkpoint or os.path.join(args.model_dir, "model.ckpt")
    model_dir = args.model_dir or os.path.dirname(ckpt)
    name = args.name or os.path.basename(os.path.normpath(model_dir))
    model_bin = args.model_bin or os.path.join(model_dir, "model.bin.gz")
    selfplay_dir = args.selfplay_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(model_dir))), "selfplay", name)
    tdata_dir = args.tdata_dir or os.path.join(selfplay_dir, "tdata")
    out_path = args.out or os.path.join(HERE, "data", name + ".json")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    log(f"loading {ckpt} on {device}")
    net, used_swa, train_state = load_net(ckpt, device)
    cfg = net.config
    result = {"meta": {
        "name": name, "checkpoint": ckpt, "swa": used_swa, "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "numParams": int(sum(p.numel() for p in net.parameters())), "device": str(device),
        "trainState": {k: train_state.get(k) for k in ("global_step_samples", "total_num_data_rows") if k in train_state},
        "selfplayDir": selfplay_dir, "tdataDir": tdata_dir, "visits": args.visits,
    }}
    result["features"] = {
        "spatial": [{"idx": i, "name": qs.SPATIAL_NAMES[i], "desc": qs.SPATIAL_DESCS[i], "continuous": i in qs.CONTINUOUS_SPATIAL}
                    for i in range(qs.NUM_SPATIAL)],
        "global": [{"idx": i, "name": qs.GLOBAL_NAMES[i], "desc": qs.GLOBAL_DESCS[i]} for i in range(qs.NUM_GLOBAL)],
        "groups": [{"name": n, "spatial": s, "global": g} for n, s, g in GROUPS],
        "phases": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in PHASES],
        "policyVariants": POLICY_VARIANTS,
    }

    # --- training rows, inference-time inputs rebuilt from the stored bits ------------------------------------
    td = load_tdata(tdata_dir, args.tdata_rows, args.seed)
    sp_train = td["bin"]
    sp_inf = qs.continuous_from_training(sp_train)
    rebuilt_ok = float(np.mean(qs.binarize_like_training(sp_inf) == sp_train))
    gl = td["glob"]
    walls_placed = 20 - np.round(gl[:, 1] * 10) - np.round(gl[:, 2] * 10)
    phase = phase_of_walls(walls_placed)

    # --- architecture ------------------------------------------------------------------------------------------
    rec = ShapeRecorder(net)
    forward(net, sp_inf[:4], gl[:4], device)
    rec.close()
    result["arch"] = build_arch(net, rec.shapes)
    result["params"] = {n: param_entry(n, p) for n, p in net.named_parameters()}
    log("architecture + parameters done")
    result["activations"] = activation_stats(net, device, sp_inf[:1024], gl[:1024])
    result["attentionHeads"] = attention_head_stats(net, device, sp_inf[:512], gl[:512])
    log("activation / attention statistics done")

    # --- feature importance ------------------------------------------------------------------------------------
    base, lb, perm = feature_importance(net, device, sp_inf, gl, td["pol"], td["gt"], phase, args.seed)
    gxi = grad_x_input(net, device, sp_inf[:2048], gl[:2048])
    log("gradient x input done")
    result["importance"] = {
        "rows": int(sp_inf.shape[0]),
        "phaseCounts": [int((phase == i).sum()) for i in range(len(PHASES))],
        "baseLoss": {k: r4(wmean(lb[k], lb["w_" + k])) for k in ("pol", "val", "mar")},
        "permutation": perm,
        "gradInput": gxi,
        "stem": stem_weights(net, sp_inf, gl),
    }
    result["calibration"] = calibration(base, td["gt"])

    # --- train / inference mismatch ----------------------------------------------------------------------------
    otr = forward(net, sp_train, gl, device)
    ltr = row_losses(otr, td["pol"], td["gt"])
    diff = compare_outputs(base, otr, lb, ltr)
    ch = qs.CONTINUOUS_SPATIAL
    result["mismatch"] = {
        "channels": ch,
        "rebuiltBitsMatch": r4(rebuilt_ok),
        "trainNonzeroRows": r4(float((sp_train[:, ch].reshape(len(sp_train), -1).max(1) > 0).mean())),
        "trainMean": r4(sp_train[:, ch].mean((0, 2, 3))),
        "inferMean": r4(sp_inf[:, ch].mean((0, 2, 3))),
        "lossInfer": {k: r4(wmean(lb[k], lb["w_" + k])) for k in ("pol", "val", "mar")},
        "lossTrain": {k: r4(wmean(ltr[k], ltr["w_" + k])) for k in ("pol", "val", "mar")},
        "lossInferPhase": [{k: r4(wmean(lb[k], lb["w_" + k], phase == i)) for k in ("pol", "val", "mar")} for i in range(len(PHASES))],
        "lossTrainPhase": [{k: r4(wmean(ltr[k], ltr["w_" + k], phase == i)) for k in ("pol", "val", "mar")} for i in range(len(PHASES))],
        "diff": {k: r4(float(v.mean())) for k, v in diff.items()},
        "diffPhase": [{k: r4(float(v[phase == i].mean())) if (phase == i).any() else None for k, v in diff.items()} for i in range(len(PHASES))],
        "dwinHist": np.histogram(diff["dwin"], bins=[0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0])[0].tolist(),
        "dwinEdges": [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0],
    }
    log("mismatch analysis done")

    # --- C++ search on selfplay positions ----------------------------------------------------------------------
    positions = []
    if args.mcts_positions > 0:
        if not os.path.exists(args.katago) or not os.path.exists(model_bin):
            log(f"skipping search: missing {args.katago if not os.path.exists(args.katago) else model_bin}")
        else:
            picks = sample_positions(selfplay_dir, args.mcts_positions, args.seed)
            positions = run_mcts(net, device, args, model_bin, picks)
    if not positions and args.attention_examples > 0:
        # still give the attention explorer something to show
        picks = sample_positions(selfplay_dir, args.attention_examples, args.seed) if os.path.isdir(selfplay_dir) else []
        positions_for_attn = [{"state": state_json(state_at(g, ply)), "ply": ply, "file": f, "history": []} for f, g, ply in picks]
    else:
        positions_for_attn = positions
    result["mcts"] = {"positions": positions, "summary": mcts_summary(positions) if positions else None}
    result["attention"] = attention_examples(net, device, positions_for_attn, args.attention_examples)
    if result["attention"]:
        result["attention"]["states"] = [p["state"] for p in positions_for_attn[:args.attention_examples]]
        result["attention"]["labels"] = [f"{p['file']} ply {p['ply']}" for p in positions_for_attn[:args.attention_examples]]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    log(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    idx_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "index.json")
    try:
        with open(idx_path) as f:
            idx = json.load(f)
    except Exception:
        idx = []
    idx = [e for e in idx if e.get("file") != os.path.basename(out_path)]
    idx.append({"file": os.path.basename(out_path), "name": name, "generated": result["meta"]["generated"],
                "samples": result["meta"]["trainState"].get("global_step_samples")})
    idx.sort(key=lambda e: e.get("samples") or 0)
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=1)


if __name__ == "__main__":
    main()
