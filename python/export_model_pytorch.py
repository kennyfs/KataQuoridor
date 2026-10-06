#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import struct
import json
import datetime
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
from torch.optim.swa_utils import AveragedModel
from katago.train.model_pytorch import RMSNormMask

from katago.train import modelconfigs
from katago.train.model_pytorch import Model, ResBlock, NestedBottleneckResBlock, TransformerAttentionBlock, TransformerFFNBlock, NestedBottleneckTransformerBlock
from katago.train.model_pytorch import compute_attn_logit_dataless_bounds
from katago.train.load_model import load_model

try:
    import onnx
except ImportError:
    onnx = None

# A transformer FFN hidden channel is pruned when no weight touching it exceeds this in
# absolute value. Training with Muon and weight decay can leave channels at exactly zero on
# every side, and such channels never recover, so this only needs to catch exact zeros.
DEAD_FFN_CHANNEL_MAX_ABS = 1e-15
# Pruned FFN widths are multiples of this, or of the largest power of two dividing the original
# width when that is smaller, so that the inference backends' matrix kernels keep tiling them
# evenly. The OpenCL tensor core path needs a multiple of 32 and the CUDA fused FFN a multiple
# of 8, and larger multiples keep the GEMM tiles full.
FFN_CHANNEL_MULTIPLE = 64


def ffn_width_granularity(ffn_dim):
    """The step between allowed pruned widths for an FFN of width ffn_dim."""
    g = 1
    while g < FFN_CHANNEL_MULTIPLE and ffn_dim % (2 * g) == 0:
        g *= 2
    return g


def ffn_blocks_in_export_order(model):
    """(name, block) for every transformer FFN block, in the order the trunk is written."""
    out = []
    for i, block in enumerate(model.blocks):
        if isinstance(block, TransformerFFNBlock):
            out.append((f"blocks.{i}", block))
        elif hasattr(block, "blockstack"):
            for j, sub in enumerate(block.blockstack):
                if isinstance(sub, TransformerFFNBlock):
                    out.append((f"blocks.{i}.blockstack.{j}", sub))
    return out


def choose_ffn_widths(live_counts, ffn_dim, granularity, scratch_factor):
    """Picks the width each FFN block of original width ffn_dim is pruned to. Each block gets the
    smallest width from a chosen set of allowed widths that still holds its live channels. The
    set is chosen to minimize the total width over the blocks, which is the compute spent on
    dead channels, under a memory bound. The inference backends pool scratch buffers by exact
    size and never free them, so every distinct width in use keeps FFN scratch memory
    proportional to that width, and the sum of the distinct widths is held to at most
    scratch_factor * ffn_dim. ffn_dim itself is always in the set, since the block the OpenCL
    tuner keys on is kept at full width. Widths are multiples of granularity. Returns the list
    of widths, one per entry of live_counts."""
    def round_up(c):
        return max(granularity, math.ceil(c / granularity) * granularity)
    # Only widths that some block would exactly fill, plus ffn_dim, need considering: any other
    # allowed width could be lowered to the next such width without changing any block's
    # assignment or raising the sum.
    cands = sorted(set(round_up(c) for c in live_counts) | {ffn_dim})
    num_cands = len(cands)
    units = [w // granularity for w in cands]
    budget = math.floor(scratch_factor * ffn_dim / granularity + 1e-9)
    if budget < units[-1]:
        raise Exception(f"-prune-ffn-scratch-factor {scratch_factor} does not leave room for the full width {ffn_dim}")
    # count_le[j] is the number of blocks whose live channels fit in cands[j].
    count_le = [sum(1 for c in live_counts if c <= w) for w in cands]
    # best[j][b] is the minimal total width for the blocks fitting in cands[j], using a set of
    # allowed widths whose largest is cands[j] and whose sum is at most b units of granularity.
    # prev holds the next smaller set element, for backtracking.
    inf = float("inf")
    best = [[inf] * (budget + 1) for _ in range(num_cands)]
    prev = [[None] * (budget + 1) for _ in range(num_cands)]
    for j in range(num_cands):
        for b in range(units[j], budget + 1):
            best[j][b] = count_le[j] * cands[j]
            for i in range(j):
                if best[i][b - units[j]] < inf:
                    cost = best[i][b - units[j]] + (count_le[j] - count_le[i]) * cands[j]
                    if cost < best[j][b]:
                        best[j][b] = cost
                        prev[j][b] = i
    allowed = []
    j = num_cands - 1
    b = budget
    while j is not None:
        allowed.append(cands[j])
        j, b = prev[j][b], b - units[j]
    allowed.sort()
    return [next(w for w in allowed if w >= c) for c in live_counts]


def prune_dead_ffn_channels(model, scratch_factor):
    """Removes hidden channels of the transformer FFN blocks whose ffn_linear1 row, gate row and
    ffn_linear2 column are all zero (below DEAD_FFN_CHANNEL_MAX_ABS in absolute value), which
    leaves the model's function unchanged. Each block keeps its live channels in their original
    order plus the lowest-index dead channels needed to reach the width chosen for it by
    choose_ffn_widths. The last FFN block is not pruned at all, because the OpenCL tuner sizes
    its FFN tuning from that block and its tuning file is shared with unpruned exports of the
    same architecture. Returns (channels before, channels after, live channels) summed over the
    blocks."""
    blocks = ffn_blocks_in_export_order(model)
    if len(blocks) == 0:
        logging.info("No transformer FFN blocks, nothing to prune")
        return 0, 0, 0
    dead_masks = []
    for name, block in blocks:
        assert not getattr(block, "use_depthwise_conv", False), f"{name}: pruning an FFN with a depthwise conv is not supported"
        w1 = block.ffn_linear1.weight.detach()
        w2 = block.ffn_linear2.weight.detach()
        dead = (w1.abs().amax(dim=1) < DEAD_FFN_CHANNEL_MAX_ABS) & (w2.abs().amax(dim=0) < DEAD_FFN_CHANNEL_MAX_ABS)
        if block.use_swiglu:
            dead &= block.ffn_linear_gate.weight.detach().abs().amax(dim=1) < DEAD_FFN_CHANNEL_MAX_ABS
        dead_masks.append(dead)

    # Widths are chosen per original width, normally a single group.
    widths = [None] * len(blocks)
    widths[-1] = blocks[-1][1].ffn_dim
    for ffn_dim in sorted(set(block.ffn_dim for _, block in blocks)):
        idx = [i for i, (_, block) in enumerate(blocks) if block.ffn_dim == ffn_dim and widths[i] is None]
        if len(idx) == 0:
            continue
        live_counts = [ffn_dim - int(dead_masks[i].sum().item()) for i in idx]
        chosen = choose_ffn_widths(live_counts, ffn_dim, ffn_width_granularity(ffn_dim), scratch_factor)
        for i, w in zip(idx, chosen):
            widths[i] = w

    total_before = total_after = total_live = 0
    for (name, block), dead, num_keep in zip(blocks, dead_masks, widths):
        ffn_dim = block.ffn_dim
        num_live = ffn_dim - int(dead.sum().item())
        total_before += ffn_dim
        total_after += num_keep
        total_live += num_live
        logging.info(f"{name}: FFN channels {ffn_dim} -> {num_keep}, {num_live} live")
        if num_keep == ffn_dim:
            continue
        live_idx = torch.nonzero(~dead).flatten()
        dead_idx = torch.nonzero(dead).flatten()
        keep = torch.sort(torch.cat([live_idx, dead_idx[:num_keep - num_live]])).values
        w1 = block.ffn_linear1.weight.detach()
        w2 = block.ffn_linear2.weight.detach()
        with torch.no_grad():
            block.ffn_linear1.weight = torch.nn.Parameter(w1[keep].clone())
            block.ffn_linear1.out_features = num_keep
            if block.use_swiglu:
                block.ffn_linear_gate.weight = torch.nn.Parameter(block.ffn_linear_gate.weight.detach()[keep].clone())
                block.ffn_linear_gate.out_features = num_keep
            block.ffn_linear2.weight = torch.nn.Parameter(w2[:, keep].clone())
            block.ffn_linear2.in_features = num_keep
        block.ffn_dim = num_keep
        if getattr(block, "fused_swiglu_kernel", False):
            from katago.train.fused_swiglu import is_supported_shape
            block.fused_swiglu_kernel = is_supported_shape(block.c_main, num_keep)
    distinct = sorted(set(widths))
    logging.info(
        f"Pruned dead FFN channels: {total_before} -> {total_after} hidden channels over {len(blocks)} FFN blocks "
        f"({total_live} live), distinct widths {distinct} summing to {sum(distinct)}"
    )
    return total_before, total_after, total_live


#Command and args-------------------------------------------------------------------


class QuoridorOnnxExportWrapper(torch.nn.Module):
    """
    Wrapper for KataQuoridor neural net inference export.
    Exposes InputSpatial (B, C, 9, 9) and InputGlobal (B, G, 1, 1) as inputs (C / G = 17 / 15 for I/O v1,
    19 / 17 for I/O v2),
    and OutputPolicy (B, 3, 9, 9) and OutputValue (B, 2, 1, 1) as outputs.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor):
        if input_global.dim() == 4:
            input_global_2d = input_global.view(input_global.shape[0], -1)
        else:
            input_global_2d = input_global
        outputs_byheads = self.model(input_spatial, input_global_2d)
        policy_out = outputs_byheads[0][0]  # (B, 18, 9, 9)
        value_out = outputs_byheads[0][1]   # (B, 2)

        # Extract Target 0 (first 3 channels: pawn, v-wall, h-wall)
        raw_policy = policy_out[:, 0:3, :, :].contiguous()  # (B, 3, 9, 9)
        raw_value = value_out.view(value_out.shape[0], 2, 1, 1).contiguous()  # (B, 2, 1, 1)
        return raw_policy, raw_value


def _register_rms_norm_symbolic(opset: int = 18):
    try:
        from torch.onnx import register_custom_op_symbolic
        def rms_norm_symbolic(g, input, normalized_shape, weight, eps=None):
            square = g.op('Mul', input, input)
            axes = g.op('Constant', value_t=torch.tensor([-1], dtype=torch.int64))
            mean = g.op('ReduceMean', square, axes, keepdims_i=1)
            if eps is None or (isinstance(eps, torch._C.Value) and eps.node().kind() == 'prim::Constant' and eps.type().kind() == 'NoneType'):
                eps_val = g.op('Constant', value_t=torch.tensor(1e-5, dtype=torch.float32))
            elif isinstance(eps, torch._C.Value):
                eps_val = eps
            else:
                eps_val = g.op('Constant', value_t=torch.tensor(float(eps), dtype=torch.float32))
            denom = g.op('Add', mean, eps_val)
            rsqrt = g.op('Sqrt', denom)
            normed = g.op('Div', input, rsqrt)
            if weight is not None and not (isinstance(weight, torch._C.Value) and weight.node().kind() == 'prim::Constant' and weight.type().kind() == 'NoneType'):
                return g.op('Mul', normed, weight)
            return normed
        register_custom_op_symbolic('::rms_norm', rms_norm_symbolic, opset)
    except Exception:
        pass


def export_quoridor_onnx(
    model: torch.nn.Module,
    export_path: str,
    model_name: str = "kataquoridor",
    opset_version: int = 18,
    verbose: bool = False,
) -> str:
    """
    Exports a KataQuoridor PyTorch model to ONNX format with KataGo metadata props.
    """
    _register_rms_norm_symbolic(opset_version)
    model.eval()
    prev_use_flex = getattr(model, "use_flex_attention", False)
    model.use_flex_attention = False
    wrapper = QuoridorOnnxExportWrapper(model)
    wrapper.eval()

    device = next(model.parameters()).device
    num_spatial = modelconfigs.get_num_bin_input_features(model.config)
    num_global = modelconfigs.get_num_global_input_features(model.config)
    dummy_spatial = torch.zeros(1, num_spatial, 9, 9, dtype=torch.float32, device=device)
    dummy_global = torch.zeros(1, num_global, 1, 1, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_spatial, dummy_global),
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["InputSpatial", "InputGlobal"],
        output_names=["OutputPolicy", "OutputValue"],
        dynamic_axes={
            "InputSpatial": {0: "batch"},
            "InputGlobal": {0: "batch"},
            "OutputPolicy": {0: "batch"},
            "OutputValue": {0: "batch"},
        },
        verbose=verbose,
        dynamo=False,
    )

    if onnx is not None:
        model_proto = onnx.load(export_path)

        meta: Dict[str, str] = {
            "katago.metadataVersion": "1",
            "katago.name": model_name,
            "katago.modelVersion": "1",
            "katago.numInputChannels": str(num_spatial),
            "katago.numInputGlobalChannels": str(num_global),
            "katago.numInputMetaChannels": "0",
            "katago.numPolicyChannels": "3",
            "katago.numValueChannels": "2",
            "katago.numScoreValueChannels": "0",
            "katago.numOwnershipChannels": "0",
            "katago.build.nnXLen": "9",
            "katago.build.nnYLen": "9",
            "katago.build.requireExactNNLen": "true",
        }

        del model_proto.metadata_props[:]
        for k, v in meta.items():
            entry = model_proto.metadata_props.add()
            entry.key = k
            entry.value = v

        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, export_path)

    return export_path


def parse_args():
    description = """
Export neural net weights to file for KataGo engine.
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-checkpoint', help='Checkpoint to test', required=False)
    parser.add_argument('-export-random-initialized-model', help='Instead of loading a checkpoint, export a freshly random-initialized model of the given model config name (e.g. b15c512h8nbttflrs-fson-silu-rsnh)', required=False)
    parser.add_argument('-export-dir', help='model file dir to save to', required=True)
    parser.add_argument('-model-name', help='name to record in model file', required=True)
    parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
    parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
    parser.add_argument('-export-14-as-15', help='Export model version 14 as 15', action="store_true", required=False)
    parser.add_argument('-export-15-or-16-as-17', help='Export model version 15 or 16 as 17', action="store_true", required=False)
    parser.add_argument('-attn-logit-bound-limit', help='Refuse to export if the data-free attention logit bound of any layer exceeds this (inference backends mask off-board keys with -3e4 in fp16, so genuine logits must stay well below that)', type=float, default=2.5e4, required=False)
    parser.add_argument('-ignore-attn-logit-bound', help='Export anyway when the attention logit bound limit is exceeded', action="store_true", required=False)
    parser.add_argument('-prune-dead-ffn-channels', help='Leave out transformer FFN hidden channels whose weights are all zero, to improve performance', action="store_true", required=False)
    parser.add_argument('-prune-ffn-scratch-factor', help='With -prune-dead-ffn-channels, the sum of the distinct FFN widths in use may be at most this multiple of the original width. Inference backends keep scratch memory for each distinct width, so this bounds the growth of that memory (default 2.0)', type=float, default=2.0, required=False)
    return vars(parser.parse_args())


def main(args):
    checkpoint_file = args["checkpoint"]
    export_random_initialized_model = args["export_random_initialized_model"]
    export_dir = args["export_dir"]
    model_name = args["model_name"]
    filename_prefix = args["filename_prefix"]
    use_swa = args["use_swa"]
    export_14_as_15 = args["export_14_as_15"]
    export_15_or_16_as_17 = args["export_15_or_16_as_17"]

    if (checkpoint_file is None) == (export_random_initialized_model is None):
        raise Exception("Exactly one of -checkpoint or -export-random-initialized-model must be specified")
    if export_random_initialized_model is not None and use_swa:
        raise Exception("-use-swa cannot be used with -export-random-initialized-model")

    os.makedirs(export_dir,exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(export_dir + "/log.txt"),
        ],
    )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # LOAD MODEL ---------------------------------------------------------------------
    if export_random_initialized_model is not None:
        if export_random_initialized_model not in modelconfigs.config_of_name:
            raise Exception(f"Unknown model config name: {export_random_initialized_model}")
        model_config = modelconfigs.config_of_name[export_random_initialized_model]
        logging.info(f"Exporting freshly random-initialized model with config: {export_random_initialized_model}")
        pos_len = 11 if modelconfigs.is_quoridor4(model_config) else (9 if modelconfigs.is_quoridor(model_config) else 19)
        model = Model(model_config, pos_len=pos_len)
        model.initialize()
        model.to("cpu")
        swa_model = None
        other_state_dict = {}
    else:
        model, swa_model, other_state_dict = load_model(checkpoint_file, use_swa, device="cpu", verbose=True)
    model_config = model.config

    # ATTN LOGIT BOUND GUARD -------------------------------------------------------
    # Data-free certification that attention logits can never approach the additive mask
    # constants the inference backends use for off-board keys (-3e4 in fp16 backends).
    # See compute_attn_logit_dataless_bounds for the derivation and its tightness.
    model_to_export = swa_model if swa_model is not None else model

    if args["prune_dead_ffn_channels"]:
        prune_dead_ffn_channels(model_to_export, args["prune_ffn_scratch_factor"])

    attn_logit_bounds = compute_attn_logit_dataless_bounds(model_to_export)
    if len(attn_logit_bounds) > 0:
        top = sorted(attn_logit_bounds.items(), key=lambda kv: -kv[1])[:5]
        logging.info(
            f"Data-free attention logit bound, max over {len(attn_logit_bounds)} layers: "
            + ", ".join(f"{n}={v:.0f}" for n, v in top)
        )
        worst_name, worst = top[0]
        if worst > args["attn_logit_bound_limit"]:
            msg = (
                f"Attention logit bound of layer {worst_name} is {worst:.0f}, exceeding the export limit "
                f"{args['attn_logit_bound_limit']:.0f}. Logits of this magnitude threaten the correctness of the "
                f"additive off-board masking constants in fp16 inference backends (-3e4). The bound is data-free "
                f"and usually several times above actual logit magnitudes - measure the actual logits and/or "
                f"consider train.py -attn-logit-penalty-cap. Use -ignore-attn-logit-bound to export anyway."
            )
            if args["ignore_attn_logit_bound"]:
                logging.warning("WARNING (-ignore-attn-logit-bound): " + msg)
            else:
                raise Exception(msg)

    # QUORIDOR ONNX EXPORT ---------------------------------------------------------
    # Optional: .bin.gz (written below) is the canonical, and only required, model format. ONNX
    # export needs the `onnx` package; skip it rather than failing the whole export if unavailable.
    if modelconfigs.is_quoridor(model_config):
        if onnx is not None:
            logging.info("Exporting Quoridor model to ONNX format")
            onnx_path = os.path.join(export_dir, filename_prefix + ".onnx")
            export_quoridor_onnx(model_to_export, onnx_path, model_name=model_name)
            logging.info(f"Exported Quoridor ONNX model to {onnx_path}")
        else:
            logging.info("onnx package not available, skipping optional ONNX export")

    # WRITING MODEL ----------------------------------------------------------------
    extension = ".bin"
    mode = "wb"
    f = open(export_dir + "/" + filename_prefix + extension, mode)
    def writeln(s):
        f.write((str(s)+"\n").encode(encoding="ascii",errors="backslashreplace"))
    def writestr(s):
        f.write(s.encode(encoding="ascii",errors="backslashreplace"))

    if modelconfigs.is_quoridor(model_config) or modelconfigs.is_quoridor4(model_config):
        # Quoridor models are always KataGo architecture version 17 (see docs/
        # KataQuoridor_Review_and_Roadmap.md §4.2): the Quoridor I/O version is a separate number,
        # written below as model option D.
        version = 17
    else:
        # Ignore what's in the config if less than 11 since a lot of testing models
        # are on old version but actually have various new architectures.
        version = max(model_config["version"],11)
    true_version = version
    # Hack to be able to export version 14 as version 15
    if version == 14 and export_14_as_15:
        version = 15
    elif export_14_as_15:
        raise Exception("export_14_as_15 specified but version was " + str(version))

    # Configs that use rmsnorm for trunk tip that claim version 15 or 16 automatically get upgraded
    # to version 17. Version 17 is a model format change that supports a strict superset of version 15 and 16,
    # representing q value predictions by outputting the expected policy outputs (2 or 4), having a few more
    # slots for unused params to allow for flexibility without future version bumps
    # Version 15 and/or 16 was used to train some experimental transformer with trunk tip rmsnorm
    # that we want to still be able to export and run, so we autoupgrade to version 17 for exporting them.
    if (version == 15 or version == 16) and export_15_or_16_as_17:
        version = 17
    elif export_15_or_16_as_17:
        raise Exception("export_15_or_16_as_17 specified but version was " + str(version))
    elif version == 15 or version == 16:
        if model.trunk_final_rmsnorm:
            logging.warn("Autoupgrading v15 or v16 model to v17 due to trunk rmsnorm")
            version = 17
        elif model_config.get("always_compute_pass_alive_under_suicide_rules"):
            logging.warn("Autoupgrading v15 or v16 model to v17 due to always_compute_pass_alive_under_suicide_rules")
            version = 17
        elif model_config.get("exclude_territory_adjacent_to_atari"):
            logging.warn("Autoupgrading v15 or v16 model to v17 due to exclude_territory_adjacent_to_atari")
            version = 17

    if model_config.get("always_compute_pass_alive_under_suicide_rules") and version < 17:
        raise Exception(
            "always_compute_pass_alive_under_suicide_rules is set but model would be exported as version "
            + str(version) + " < 17, which can technically represent it but we're not outputting in practice"
        )
    if model_config.get("exclude_territory_adjacent_to_atari") and version < 17:
        raise Exception(
            "exclude_territory_adjacent_to_atari is set but model would be exported as version "
            + str(version) + " < 17, which can technically represent it but we're not outputting in practice"
        )

    writeln(model_name)
    writeln(version)
    writeln(modelconfigs.get_num_bin_input_features(model_config))
    writeln(modelconfigs.get_num_global_input_features(model_config))

    if version <= 12:
        assert model.td_score_multiplier == 20.0
        assert model.scoremean_multiplier == 20.0
        assert model.scorestdev_multiplier == 20.0
        assert model.lead_multiplier == 20.0
        assert model.variance_time_multiplier == 40.0
        assert model.shortterm_value_error_multiplier == 0.25
        assert model.shortterm_score_error_multiplier == 30.0
    else:
        writeln(model.td_score_multiplier)
        writeln(model.scoremean_multiplier)
        writeln(model.scorestdev_multiplier)
        writeln(model.lead_multiplier)
        writeln(model.variance_time_multiplier)
        writeln(model.shortterm_value_error_multiplier)
        writeln(model.shortterm_score_error_multiplier)

    if version >= 15:
        if model.metadata_encoder is not None:
            writeln(model.metadata_encoder.meta_encoder_version)
        else:
            writeln(0)

        # preferPassAliveUnderSuicideRules: 1 if the model expects pass-alive area input features
        # to be computed as if multi-stone suicide were always legal, regardless of the actual
        # suicide rule. Only version 17+ engines parse nonzero values here (checked above).
        if model_config.get("always_compute_pass_alive_under_suicide_rules"):
            writeln(1)
        else:
            writeln(0)
        # preferExcludeTerritoryAdjacentToAtari: 1 if the model expects territory scoring with
        # TaxRule NONE to exclude empty points adjacent to chains in atari (rules version 3),
        # both for adjudication and for its territory input features.
        if model_config.get("exclude_territory_adjacent_to_atari"):
            writeln(1)
        else:
            writeln(0)
        # Model option D: Quoridor I/O version (0 = not a Quoridor network, i.e. a Go network; else the config's
        # quoridor_io_version, 1, 2 or 3; Q4 four-player networks: 100 + q4_io_version, docs/q4/Q4IO.md).
        # Options E-H are unused spare slots for future model options.
        if modelconfigs.is_quoridor4(model_config):
            writeln(modelconfigs.Q4_IO_VERSION_BASE + modelconfigs.get_q4_io_version(model_config))
        else:
            writeln(modelconfigs.get_quoridor_io_version(model_config) if modelconfigs.is_quoridor(model_config) else 0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)


    def write_weights(weights):
        # Little endian
        reshaped = np.reshape(weights.detach().numpy(),[-1])
        num_weights = len(reshaped)
        writestr("@BIN@")
        f.write(struct.pack(f'<{num_weights}f',*reshaped))
        writestr("\n")

    def write_conv_weight(name,convweight):
        (out_channels, in_channels, diamy, diamx) = convweight.shape
        dilation = 1
        writeln(name)
        writeln(diamy) #y
        writeln(diamx) #x
        writeln(in_channels)
        writeln(out_channels)
        writeln(dilation) #y
        writeln(dilation) #x
        # Torch order is oc,ic,y,x
        # Desired output order is y,x,ic,oc
        write_weights(torch.permute(convweight,(2,3,1,0)))

    def write_conv(name,conv):
        assert conv.bias is None
        write_conv_weight(name, conv.weight)

    def write_bn(name,normmask):
        writeln(name)

        writeln(normmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = normmask.scale is not None or normmask.gamma is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        if hasattr(normmask,"running_mean") and normmask.running_mean is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_mean.shape == (normmask.c_in,)
            write_weights(normmask.running_mean)
        else:
            assert not normmask.is_using_batchnorm
            write_weights(torch.zeros(normmask.c_in, dtype=torch.float))

        if hasattr(normmask,"running_std") and normmask.running_std is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_std.shape == (normmask.c_in,)
            write_weights(torch.maximum(torch.tensor(1e-20), normmask.running_std * normmask.running_std - epsilon))
        else:
            assert not normmask.is_using_batchnorm
            write_weights((1.0-epsilon) * torch.ones(normmask.c_in, dtype=torch.float))

        if normmask.scale is not None:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                if normmask.gamma_weight_decay_center_1:
                    write_weights(normmask.scale * (normmask.gamma+1.0))
                else:
                    write_weights(normmask.scale * normmask.gamma)
            else:
                assert has_gamma_or_scale
                write_weights(normmask.scale * torch.ones(normmask.c_in, dtype=torch.float, device="cpu"))
        else:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                if normmask.gamma_weight_decay_center_1:
                    write_weights(normmask.gamma+1.0)
                else:
                    write_weights(normmask.gamma)
            else:
                assert not has_gamma_or_scale
                pass

        assert normmask.beta.shape == (1, normmask.c_in, 1, 1)
        write_weights(normmask.beta)

    def write_biasmask(name,biasmask):
        writeln(name)

        writeln(biasmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = biasmask.scale is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        write_weights(torch.zeros(biasmask.c_in, dtype=torch.float))
        write_weights((1.0-epsilon) * torch.ones(biasmask.c_in, dtype=torch.float))

        if biasmask.scale is not None:
            write_weights(biasmask.scale * torch.ones(biasmask.c_in, dtype=torch.float, device="cpu"))

        assert biasmask.beta.shape == (1, biasmask.c_in, 1, 1)
        write_weights(biasmask.beta)

    def write_activation(name, activation):
        writeln(name)
        if isinstance(activation,torch.nn.ReLU):
            writeln("ACTIVATION_RELU")
        elif isinstance(activation,torch.nn.Mish):
            writeln("ACTIVATION_MISH")
        elif isinstance(activation,torch.nn.SiLU):
            writeln("ACTIVATION_SILU")
        elif isinstance(activation,torch.nn.Identity):
            writeln("ACTIVATION_IDENTITY")
        else:
            assert False, f"Activation not supported for export: {activation}"


    def write_matmul(name,linearweight):
        writeln(name)
        (out_channels,in_channels) = linearweight.shape
        writeln(in_channels)
        writeln(out_channels)
        # Torch order is oc,ic
        # Desired output order is ic,oc
        write_weights(torch.permute(linearweight,(1,0)))

    def write_matbias(name,linearbias):
        writeln(name)
        (out_channels,) = linearbias.shape
        writeln(out_channels)
        write_weights(linearbias)

    def write_rmsnorm(name,rmsnormmask):
        writeln(name)
        writeln(rmsnormmask.c_in)
        writeln(rmsnormmask.eps)
        writeln(1 if rmsnormmask.spatial else 0)
        cgroup_size = rmsnormmask.cgroup_size if rmsnormmask.cgroup_size is not None else 0
        writeln(cgroup_size)
        if not rmsnormmask.spatial:
            # Non-spatial: weight comes from torch.nn.RMSNorm
            assert rmsnormmask.norm is not None
            assert rmsnormmask.norm.weight.shape == (rmsnormmask.c_in,)
            write_weights(rmsnormmask.norm.weight)
        else:
            # Spatial: weight comes from self.gamma
            assert rmsnormmask.gamma.shape == (rmsnormmask.c_in,)
            write_weights(rmsnormmask.gamma)
        assert rmsnormmask.beta.shape == (rmsnormmask.c_in,)
        write_weights(rmsnormmask.beta)

    def write_normactconv(name,normactconv):
        if normactconv.c_gpool is None:
            assert normactconv.convpool is None
            if normactconv.conv1x1 is None:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                write_conv(name+".conv", normactconv.conv)
            else:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                # Torch conv order is oc,ic,h,w
                # We want to add the 1x1 conv to the center of the h,w
                h,w = (normactconv.conv.weight.shape[2],normactconv.conv.weight.shape[3])
                assert h % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                assert w % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                combined_conv = normactconv.conv.weight.detach().clone()
                combined_conv[:,:,h//2:h//2+1,w//2:w//2+1] += normactconv.conv1x1.weight
                assert normactconv.conv.bias is None
                assert normactconv.conv1x1.bias is None
                write_conv_weight(name+".conv", combined_conv)
        else:
            assert normactconv.convpool is not None
            assert normactconv.conv1x1 is None
            write_bn(name+".norm", normactconv.norm)
            write_activation(name+".act", normactconv.act)
            write_conv(name+".convpool.conv1r", normactconv.convpool.conv1r)
            write_conv(name+".convpool.conv1g", normactconv.convpool.conv1g)
            write_bn(name+".convpool.normg", normactconv.convpool.normg)
            write_activation(name+".convpool.actg", normactconv.convpool.actg)
            write_matmul(name+".convpool.linear_g", normactconv.convpool.linear_g.weight)
            assert normactconv.convpool.linear_g.bias is None

    def write_transformer_norm(name,rmsnorm):
        """Write an inline RMSNorm (torch.nn.RMSNorm) used inside transformer blocks.
        These are simpler than the trunk-final RMSNormMask: just a weight vector, no bias, no spatial modes."""
        writeln(name)
        num_channels = rmsnorm.weight.shape[0]
        writeln(num_channels)
        writeln(rmsnorm.eps)
        write_weights(rmsnorm.weight)

    def write_transformer_attention_block(name,block):
        assert not getattr(block, 'use_qk_norm', False), \
            f"{name}: QK normalization is not yet supported for export"
        assert not getattr(block, 'use_gab', False), \
            f"{name}: GAB attention bias is not yet supported for export"
        assert not getattr(block, 'use_tab', False), \
            f"{name}: TAB attention bias is not yet supported for export"
        assert not getattr(block, 'inline_registers', False), \
            f"{name}: Inline registers are not yet supported for export"
        assert getattr(block, 'num_rw_registers', 0) == 0, \
            f"{name}: RW registers are not yet supported for export"

        writeln("transformer_attention_block")
        writeln(name)
        writeln(block.num_heads)
        writeln(block.num_kv_heads)
        writeln(block.q_head_dim)
        writeln(block.v_head_dim)
        writeln(1 if block.use_rope else 0)
        writeln(1 if block.learnable_rope else 0)

        write_transformer_norm(name+".norm1", block.norm1)
        write_matmul(name+".q_proj", block.q_proj.weight)
        write_matmul(name+".k_proj", block.k_proj.weight)
        write_matmul(name+".v_proj", block.v_proj.weight)
        write_matmul(name+".out_proj", block.out_proj.weight)

        if block.use_rope:
            if block.learnable_rope:
                # rope_freqs: (num_kv_heads, q_head_dim//2, 2)
                writeln(name+".rope_freqs")
                freqs = block.rope_freqs.detach()
                writeln(freqs.shape[0])  # num_kv_heads
                writeln(freqs.shape[1])  # num_pairs
                writeln(freqs.shape[2])  # 2 (omega_x, omega_y)
                write_weights(freqs)
            else:
                # Non-learnable: write theta so C++ can recompute
                writeln(name+".rope_theta")
                writeln(block.rope_theta)

    def write_transformer_ffn_block(name,block):
        assert not getattr(block, 'use_depthwise_conv', False), \
            f"{name}: FFN depthwise conv is not yet supported for export"
        assert not getattr(block, 'inline_registers', False), \
            f"{name}: Inline registers are not yet supported for export"
        assert getattr(block, 'num_rw_registers', 0) == 0, \
            f"{name}: RW registers are not yet supported for export"

        writeln("transformer_ffn_block")
        writeln(name)
        writeln(block.c_main)
        writeln(block.ffn_dim)
        writeln(1 if block.use_swiglu else 0)

        write_transformer_norm(name+".norm", block.norm)
        write_matmul(name+".ffn_linear1", block.ffn_linear1.weight)
        if block.use_swiglu:
            write_matmul(name+".ffn_linear_gate", block.ffn_linear_gate.weight)
        write_matmul(name+".ffn_linear2", block.ffn_linear2.weight)

    def write_block(name,block):
        if isinstance(block,ResBlock) and block.normactconv1.c_gpool is None:
            assert block.normactconv2.c_gpool is None
            writeln("ordinary_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,ResBlock) and block.normactconv1.c_gpool is not None:
            assert block.normactconv2.c_gpool is None
            writeln("gpool_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,NestedBottleneckResBlock):
            writeln("nested_bottleneck_block")
            writeln(name)
            writeln(block.internal_length)
            assert block.internal_length == len(block.blockstack)
            write_normactconv(name+".normactconvp", block.normactconvp)
            for i,subblock in enumerate(block.blockstack):
                write_block(name+".blockstack."+str(i),subblock)
            write_normactconv(name+".normactconvq", block.normactconvq)
        elif isinstance(block,TransformerAttentionBlock):
            write_transformer_attention_block(name, block)
        elif isinstance(block,TransformerFFNBlock):
            write_transformer_ffn_block(name, block)
        elif isinstance(block,NestedBottleneckTransformerBlock):
            writeln("nested_bottleneck_block")
            writeln(name)
            writeln(2 * block.internal_length)
            assert len(block.blockstack) == 2 * block.internal_length
            write_normactconv(name+".normactconvp", block.normactconvp)
            for i,subblock in enumerate(block.blockstack):
                write_block(name+".blockstack."+str(i), subblock)
            write_normactconv(name+".normactconvq", block.normactconvq)
        else:
            assert False, f"This kind of block is not supported for export right now: {type(block)}"

    def write_metadata_encoder(name,encoder):
        writeln(name)
        writeln(encoder.c_input)
        # Torch order is oc,ic. Flatten feature mask into the first mul
        write_matmul(name+".mul1", encoder.linear1.weight * encoder.feature_mask.reshape((1,-1)))
        write_matbias(name+".bias1", encoder.linear1.bias)
        write_activation(name+".act1", encoder.act1)
        write_matmul(name+".mul2", encoder.linear2.weight)
        write_matbias(name+".bias2", encoder.linear2.bias)
        write_activation(name+".act2", encoder.act2)
        write_matmul(name+".mul3", encoder.out_scale * encoder.linear_output_to_trunk.weight)
        assert encoder.linear_output_to_trunk.bias is None

    def write_trunk(name,model):
        writeln("trunk")
        writeln(len(model.blocks))
        writeln(model.c_trunk)
        writeln(model.c_mid)
        writeln(model.c_mid-model.c_gpool)
        writeln(model.c_gpool)
        writeln(model.c_gpool)
        if version >= 15:
            # Trunk final norm kind:
            # 0 = standard (batch norm or bias mask)
            # 1 = RMSNorm (non-spatial)
            # 2 = RMSNorm spatial (no groups)
            # 3 = RMSNorm spatial grouped
            if model.trunk_final_rmsnorm:
                assert isinstance(model.norm_trunkfinal, RMSNormMask)
                assert model.norm_trunkfinal.cgroup_size is None, \
                    "Grouped spatial RMSNorm (cgroup_size != None) is not supported for export"
                trunk_norm_kind = 1  # TRUNK_NORM_KIND_RMSNORM; spatial flag is in the RMSNormLayerDesc

                # Older katago c++ versions didn't actually do proper checking here and yet didn't support
                # this so it's not actually sound to output a non-zero trunk kind.
                # Version 17 loading code on c++ does do proper checking.
                assert version >= 17
            else:
                trunk_norm_kind = 0
            writeln(trunk_norm_kind)
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)

        write_conv("model.conv_spatial", model.conv_spatial)
        write_matmul("model.linear_global", model.linear_global.weight)
        assert model.linear_global.bias is None
        if model.metadata_encoder is not None:
            assert version >= 15
            write_metadata_encoder("model.sgf_metadata_encoder",model.metadata_encoder)

        for i,block in enumerate(model.blocks):
            write_block("model.blocks."+str(i), block)
        if model.trunk_final_rmsnorm:
            write_rmsnorm("model.norm_trunkfinal", model.norm_trunkfinal)
        elif model.trunk_normless:
            write_biasmask("model.norm_trunkfinal", model.norm_trunkfinal)
        else:
            write_bn("model.norm_trunkfinal", model.norm_trunkfinal)
        write_activation("model.act_trunkfinal", model.act_trunkfinal)

    def write_policy_head(name,policyhead):
        writeln(name)
        if version >= 17:
            if modelconfigs.is_quoridor4(model_config):
                assert policyhead.conv2p.weight.shape[0] == 6
                writeln(2) # search policy and style policy, each with 3 planes (see policy option A below)
            elif modelconfigs.is_quoridor(model_config):
                assert policyhead.conv2p.weight.shape[0] == 18
                writeln(2) # regular and (short-term) optimistic policy, each with 3 planes (see policy option A below)
            else:
                assert policyhead.conv2p.weight.shape[0] == 6 or policyhead.conv2p.weight.shape[0] == 8
                if policyhead.conv2p.weight.shape[0] == 6:
                    writeln(2) # we're going to write 2 policy output channels - regular and optimistic (see below)
                else:
                    writeln(4) # we're going to write 4 policy output channels - regular, optimistic, q winloss, q score (see below)
            # Policy option A: numPolicyPlanes, packed variant-major (channel = variant*numPolicyPlanes
            # + plane). 3 for Quoridor (pawn, vertical wall, horizontal wall); 0 (meaning 1) for Go.
            writeln(3 if modelconfigs.is_quoridor(model_config) or modelconfigs.is_quoridor4(model_config) else 0)
            # Options B and C are unused spare slots for future policy-head options.
            writeln(0)
            writeln(0)
        write_conv(name+".conv1p", policyhead.conv1p)
        write_conv(name+".conv1g", policyhead.conv1g)
        write_biasmask(name+".biasg", policyhead.biasg)
        write_activation(name+".actg", policyhead.actg)
        write_matmul(name+".linear_g", policyhead.linear_g.weight)
        assert policyhead.linear_g.bias is None
        write_biasmask(name+".bias2", policyhead.bias2)
        write_activation(name+".act2", policyhead.act2)

        # Write the this-move prediction and the (short-term) optimistic policy prediction, each
        # with 3 planes (pawn, vertical wall, horizontal wall): PolicyHead's 18 outputs are
        # [policy, opp reply, soft, soft opp reply, long-term-optimistic, short-term-optimistic] x
        # 3 planes each; channels [0,1,2] are target 0 (policy) and [15,16,17] are target 5
        # (short-term optimistic), matching how upstream Go picks channels 0 and 5.
        if modelconfigs.is_quoridor4(model_config):
            # Q4: Q4PolicyHead's 6 outputs are already [search policy, style policy] x 3 planes (variant-major).
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", policyhead.conv2p.weight)
            c_g1 = policyhead.conv1g.weight.shape[0]
            c_p1 = int(policyhead.linear_g.weight.shape[0])
            # Zero-weight pass layers, as for Duel: there is no pass move.
            write_matmul(name+".linear_pass", torch.zeros((c_p1, 3 * c_g1), dtype=torch.float32))
            write_matbias(name+".linear_pass_bias", torch.zeros((c_p1,), dtype=torch.float32))
            write_activation(name+".act_pass", torch.nn.Identity())
            write_matmul(name+".linear_pass2", torch.zeros((2, c_p1), dtype=torch.float32))
        elif modelconfigs.is_quoridor(model_config):
            assert policyhead.conv2p.weight.shape[0] == 18
            write_conv_weight(
                name+".conv2p",
                torch.cat((policyhead.conv2p.weight[0:3], policyhead.conv2p.weight[15:18]), dim=0)
            )
            c_g1 = policyhead.conv1g.weight.shape[0]
            c_p1 = int(policyhead.linear_g.weight.shape[0])
            # Zero-weight pass layers: KataQuoridor never has a meaningful pass move, and the
            # Eigen backend ignores this output entirely for numPolicyPlanes > 1. Keeping the
            # structure (rather than the single-matmul pre-v15 form) costs nothing and needs no
            # parser changes.
            write_matmul(name+".linear_pass", torch.zeros((c_p1, 3 * c_g1), dtype=torch.float32))
            write_matbias(name+".linear_pass_bias", torch.zeros((c_p1,), dtype=torch.float32))
            write_activation(name+".act_pass", torch.nn.Identity())
            write_matmul(name+".linear_pass2", torch.zeros((2, c_p1), dtype=torch.float32))
        elif version <= 11:
            assert policyhead.conv2p.weight.shape[0] == 4
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0],), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 4
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0],), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version <= 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version == 15 and true_version == 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            linear_pass_stack = [policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]]
            c_p1 = int(policyhead.linear_g.weight.shape[0])
            for _ in range(c_p1-2):
                linear_pass_stack.append(torch.zeros_like(linear_pass_stack[0]))
            write_matmul(name+".linear_pass", torch.stack(linear_pass_stack, dim=0))
            assert policyhead.linear_pass.bias is None
            write_matbias(name+".linear_pass_bias", torch.tensor([0.0]*c_p1,dtype=torch.float32,device="cpu"))
            write_activation(name+".act_pass", torch.nn.Identity())
            write_matmul(name+".linear_pass2", torch.tensor([[1.0,0.0]+[0.0]*(c_p1-2),[0.0,1.0]+[0.0]*(c_p1-2)],dtype=torch.float32,device="cpu"))
        elif version <= 15 or policyhead.conv2p.weight.shape[0] == 6:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            write_matmul(name+".linear_pass", policyhead.linear_pass.weight)
            write_matbias(name+".linear_pass_bias", policyhead.linear_pass.bias)
            write_activation(name+".act_pass", policyhead.act_pass)
            assert policyhead.linear_pass2.weight.shape[0] == 6
            write_matmul(name+".linear_pass2", torch.stack((policyhead.linear_pass2.weight[0], policyhead.linear_pass2.weight[5]), dim=0))
            assert policyhead.linear_pass2.bias is None
        else:
            assert policyhead.conv2p.weight.shape[0] == 8
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5], policyhead.conv2p.weight[6], policyhead.conv2p.weight[7]), dim=0))
            write_matmul(name+".linear_pass", policyhead.linear_pass.weight)
            write_matbias(name+".linear_pass_bias", policyhead.linear_pass.bias)
            write_activation(name+".act_pass", policyhead.act_pass)
            assert policyhead.linear_pass2.weight.shape[0] == 8
            write_matmul(name+".linear_pass2", torch.stack((policyhead.linear_pass2.weight[0], policyhead.linear_pass2.weight[5], policyhead.linear_pass2.weight[6], policyhead.linear_pass2.weight[7]), dim=0))
            assert policyhead.linear_pass2.bias is None

        assert policyhead.conv2p.bias is None


    def write_value_head(name, valuehead):
        writeln(name)
        if version >= 17:
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
        write_conv(name+".conv1", valuehead.conv1)
        write_biasmask(name+".bias1", valuehead.bias1)
        write_activation(name+".act1", valuehead.act1)
        write_matmul(name+".linear2", valuehead.linear2.weight)
        write_matbias(name+".bias2", valuehead.linear2.bias)
        write_activation(name+".act2", valuehead.act2)

        if modelconfigs.is_quoridor4(model_config):
            # Q4: 5 value logits (me, next, across, previous, draw), 6 misc values, and my pawn's trajectory in the
            # ownership slot. Q4ValueHead's training-only heads (all-seat trajectories, wall placements) are not exported.
            assert valuehead.linear_value.weight.shape[0] == modelconfigs.Q4_NUM_VALUE_LOGITS
            assert valuehead.linear_misc.weight.shape[0] == modelconfigs.Q4_NUM_MISC
            assert valuehead.conv_trajectory.weight.shape[0] == modelconfigs.Q4_NUM_OWNERSHIP_CHANNELS
            write_matmul(name+".linear_valuehead", valuehead.linear_value.weight)
            write_matbias(name+".bias_valuehead", valuehead.linear_value.bias)
            write_matmul(name+".linear_miscvaluehead", valuehead.linear_misc.weight)
            write_matbias(name+".bias_miscvaluehead", valuehead.linear_misc.bias)
            write_conv_weight(name+".conv_ownership", valuehead.conv_trajectory.weight)
            return

        if modelconfigs.is_quoridor(model_config):
            # v17 value channels: [win, loss, noResult]. noResult gets zero weights and bias -30
            # so the engine never predicts it: Quoridor has no no-result games (a maxPlies draw is a result, trained
            # as win 0.5 / loss 0.5, docs/QuoridorIOv2.md).
            assert valuehead.linear_value.weight.shape[0] == 2
            c_v2 = valuehead.linear2.weight.shape[0]
            c_v1 = valuehead.conv1.weight.shape[0]
            zero_row_w = torch.zeros((1, c_v2), dtype=torch.float32)
            value_w = torch.cat((valuehead.linear_value.weight, zero_row_w), dim=0)
            value_b = torch.cat((valuehead.linear_value.bias, torch.tensor([-30.0], dtype=torch.float32)), dim=0)
            write_matmul(name+".linear_valuehead", value_w)
            write_matbias(name+".bias_valuehead", value_b)

            # v17 scoreValue channels: [scoreMean, scoreStdev(pre-softplus), lead, varTimeLeft,
            # shorttermWinlossError, shorttermScoreError] <- [utility score u, its stdev, lead s, variance time,
            # shortterm winloss error, shortterm score error] (I/O v2), with the lead from the utility-score head
            # for I/O v1 nets (whose one head predicts the margin). The post-processing multipliers written in the
            # header (Model.__init__, Quoridor branch) make whiteScoreMean and whiteLead come out in moves,
            # matching Model.postprocess_output. The remaining-plies head is training-only and not exported.
            linear_utility_score, linear_variance_time, misc = (
                valuehead.linear_utility_score,
                valuehead.linear_variance_time,
                valuehead.linear_misc,
            )
            linear_lead = valuehead.linear_lead if valuehead.io_version >= 2 else linear_utility_score
            misc_w = torch.cat((
                linear_utility_score.weight,
                misc.weight[0:1],
                linear_lead.weight,
                linear_variance_time.weight,
                misc.weight[1:3],
            ) ,dim=0)
            misc_b = torch.cat((
                linear_utility_score.bias,
                misc.bias[0:1],
                linear_lead.bias,
                linear_variance_time.bias,
                misc.bias[1:3],
            ), dim=0)

            write_matmul(name+".linear_miscvaluehead", misc_w)
            write_matbias(name+".bias_miscvaluehead", misc_b)

            # Ownership: zero weights. nneval.cpp refuses includeOwnership requests for Quoridor
            # for now (docs/KataQuoridor_Review_and_Roadmap.md §4.2), so this is never read.
            write_conv_weight(name+".conv_ownership", torch.zeros((1, c_v1, 1, 1), dtype=torch.float32))
            return

        write_matmul(name+".linear_valuehead", valuehead.linear_valuehead.weight)
        write_matbias(name+".bias_valuehead", valuehead.linear_valuehead.bias)

        # For now, only output the scoremean and scorestdev and lead and vtime channels
        w = valuehead.linear_miscvaluehead.weight[0:4]
        b = valuehead.linear_miscvaluehead.bias[0:4]
        # Grab the shortterm channels
        w2 = valuehead.linear_moremiscvaluehead.weight[0:2]
        b2 = valuehead.linear_moremiscvaluehead.bias[0:2]
        w = torch.cat((w,w2),dim=0)
        b = torch.cat((b,b2),dim=0)
        write_matmul(name+".linear_miscvaluehead", w)
        write_matbias(name+".bias_miscvaluehead", b)

        write_conv(name+".conv_ownership",valuehead.conv_ownership)

    def write_model(model):
        write_trunk("model",model)
        write_policy_head("model.policy_head",model.policy_head)
        write_value_head("model.value_head",model.value_head)

    if swa_model is not None:
        logging.info("Writing SWA model")
        write_model(swa_model)
    else:
        logging.info("Writing model")
        write_model(model)
    f.close()

    bin_path = os.path.join(export_dir, filename_prefix + extension)
    bin_gz_path = os.path.join(export_dir, filename_prefix + ".bin.gz")
    logging.info(f"Gzipping {bin_path} to {bin_gz_path}")
    import gzip
    import shutil
    with open(bin_path, 'rb') as f_in, gzip.open(bin_gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    with open(os.path.join(export_dir,"metadata.json"),"w") as f:
        train_state = other_state_dict.get("train_state", {})
        data = {}
        if "global_step_samples" in train_state:
            data["global_step_samples"] = train_state["global_step_samples"]
        if "total_num_data_rows" in train_state:
            data["total_num_data_rows"] = train_state["total_num_data_rows"]
        if "running_metrics" in other_state_dict:
            assert sorted(list(other_state_dict["running_metrics"].keys())) == ["sums", "weights"]
            data["extra_stats"] = {
                "sums": { key: value for (key,value) in other_state_dict["running_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                "weights": { key: value for (key,value) in other_state_dict["running_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
            }
            if "last_val_metrics" in other_state_dict and "sums" in other_state_dict["last_val_metrics"] and "weights" in other_state_dict["last_val_metrics"]:
                data["extra_stats"]["last_val_metrics"] = {
                    "sums": { key: value for (key,value) in other_state_dict["last_val_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                    "weights": { key: value for (key,value) in other_state_dict["last_val_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
                }
        json.dump(data,f)


    logging.info("Exported at: ")
    logging.info(str(datetime.datetime.utcnow()) + " UTC")

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main(parse_args())
