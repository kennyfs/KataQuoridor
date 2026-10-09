import logging
import os

import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

from ..train import modelconfigs

# Needs to be kept in sync with GLOBAL_TARGET_NUM_CHANNELS in trainingwrite.cpp C++ code among other places.
# Data format version 2 files (recorded in channel 63 of each row) had only 64 channels; they are zero-padded
# up to this width when loading, which correctly encodes "not reanalyzed" for the version 3 channels.
GLOBAL_TARGETS_NC_CHANNELS = 80

def pad_global_targets_nc(globalTargetsNC: np.ndarray) -> np.ndarray:
    """Zero-pad older-format globalTargetsNC rows up to the current channel count."""
    num_channels = globalTargetsNC.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return globalTargetsNC
    assert num_channels < GLOBAL_TARGETS_NC_CHANNELS, f"globalTargetsNC has {num_channels} channels, more than the expected {GLOBAL_TARGETS_NC_CHANNELS}"
    padded = np.zeros((globalTargetsNC.shape[0], GLOBAL_TARGETS_NC_CHANNELS), dtype=globalTargetsNC.dtype)
    padded[:, :num_channels] = globalTargetsNC
    return padded

# Quoridor spatial input channels 8..11 are continuous BFS distances, d < 0 ? 1 : min(1, d/32)
# (QuoridorNN::fillRow). They cannot be stored in the packed bit planes (which hold 0 there);
# the npz carries them raw in spatialDistNCHW (uint8 [N,4,H,W], canonical orientation, 255 =
# unreachable), and the loader rebuilds the channels from it.
# Spatial input channels on the 8x8 wall-anchor grid (mirrored c -> 7 - c): placed V / H walls, the anchor domain
# mask, and (I/O v2) the legal V / H wall placements. The I/O v3 repetition planes 19, 20 are pawn-cell planes and
# mirror like the pawns (c -> 8 - c).
QUORIDOR_WALL_ANCHOR_CHANNELS = (14, 15, 16, 17, 18)
QUORIDOR_DIST_FIRST_CHANNEL = 8
QUORIDOR_DIST_NUM_CHANNELS = 4
QUORIDOR_DIST_UNREACHABLE = 255
QUORIDOR_DIST_SCALE = 32.0

def decode_quoridor_dist_planes(spatialDistNCHW: np.ndarray) -> np.ndarray:
    d = spatialDistNCHW.astype(np.float32)
    return np.where(spatialDistNCHW == QUORIDOR_DIST_UNREACHABLE, np.float32(1.0), np.minimum(np.float32(1.0), d / np.float32(QUORIDOR_DIST_SCALE))).astype(np.float32)

def apply_quoridor_dist_planes(binaryInputNCHW: np.ndarray, spatialDistNCHW: np.ndarray, npz_file="") -> None:
    """Overwrites channels 8..11 of unpacked float inputs [N,C,H,W] in place with decoded distances.
    Must run before any symmetry is applied (spatialDistNCHW is stored unmirrored)."""
    lo = QUORIDOR_DIST_FIRST_CHANNEL
    hi = lo + QUORIDOR_DIST_NUM_CHANNELS
    assert spatialDistNCHW.dtype == np.uint8, f"{npz_file}: spatialDistNCHW dtype {spatialDistNCHW.dtype}"
    assert spatialDistNCHW.shape == (binaryInputNCHW.shape[0], QUORIDOR_DIST_NUM_CHANNELS) + binaryInputNCHW.shape[2:], (
        f"{npz_file}: spatialDistNCHW shape {spatialDistNCHW.shape} vs inputs {binaryInputNCHW.shape}")
    assert not binaryInputNCHW[:, lo:hi].any(), f"{npz_file}: packed bit planes 8..11 are nonzero; data not upgraded correctly"
    binaryInputNCHW[:, lo:hi] = decode_quoridor_dist_planes(spatialDistNCHW)

def decode_binary_input(binaryInputNCHWPacked: np.ndarray, spatialDistNCHW: np.ndarray, pos_len: int, npz_file="") -> np.ndarray:
    """Packed bit planes + raw distances from an npz -> float32 spatial inputs [N,C,H,W], unmirrored."""
    binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
    assert len(binaryInputNCHW.shape) == 3
    assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
    binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
    binaryInputNCHW = np.reshape(binaryInputNCHW, (
        binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
    )).astype(np.float32)
    apply_quoridor_dist_planes(binaryInputNCHW, spatialDistNCHW, npz_file)
    return binaryInputNCHW

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    include_meta: bool,
    model_config: modelconfigs.ModelConfig,
    prefetch_depth: int = 1,
):
    if modelconfigs.is_quoridor4(model_config):
        yield from read_npz_training_data_q4(
            npz_files, batch_size, world_size, rank, pos_len, device, randomize_symmetries, model_config, prefetch_depth,
            include_meta=include_meta)
        return

    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    (h_base,h_builder) = build_history_matrices(model_config, device)

    # Version 16 always predicts q values; version 17+ does so only when configured.
    include_qvalues = model_config["version"] == 16 or (
        model_config["version"] >= 17 and bool(model_config.get("predict_q_values"))
    )

    def load_npz_file(npz_file):
        # Select only THIS rank's rows up front, while the arrays are still in
        # their compact on-disk dtypes (packed bits / int8 / int16), so the
        # expensive unpackbits + float32 expansion runs on 1/world_size of the
        # data rather than the whole shard in every rank.
        with np.load(npz_file) as npz:
            num_samples = npz["globalInputNC"].shape[0]
            num_whole_steps = num_samples // (batch_size * world_size)
            used = num_whole_steps * world_size * batch_size

            def select_rank_rows(arr):
                # Keep only the rows this rank will consume:
                # reshape the used prefix to (steps, world_size, batch, ...) and
                # take this rank's slice. For world_size>1 the trailing reshape
                # forces a compact 1/world_size-size copy and lets the full
                # decompressed array be freed immediately.
                # Drop any trailing suffix that doesn't match the overall world batch size.
                arr = arr[:used]
                rest = arr.shape[1:]
                arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
                arr = arr[:, rank]
                return arr.reshape(num_whole_steps * batch_size, *rest)

            binaryInputNCHWPacked = select_rank_rows(npz["binaryInputNCHWPacked"])
            if "spatialDistNCHW" not in npz:
                raise KeyError(
                    f"{npz_file} lacks spatialDistNCHW (S8-S11 raw distances). Data written before the S8-S11 fix "
                    "must be upgraded with python/quoridor_add_dist_planes.py; see docs/DistPlanesUpgrade.md."
                )
            spatialDistNCHW = select_rank_rows(npz["spatialDistNCHW"])
            globalInputNC = select_rank_rows(npz["globalInputNC"])
            policyTargetsNCMove = select_rank_rows(npz["policyTargetsNCMove"]).astype(np.float32)
            globalTargetsNC = pad_global_targets_nc(select_rank_rows(npz["globalTargetsNC"]))
            scoreDistrN = select_rank_rows(npz["scoreDistrN"]).astype(np.float32)
            valueTargetsNCHW = select_rank_rows(npz["valueTargetsNCHW"]).astype(np.float32)
            if include_meta:
                metadataInputNC = select_rank_rows(npz["metadataInputNC"]).astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = select_rank_rows(npz["qValueTargetsNCMove"]).astype(np.float32)
            else:
                qValueTargetsNCMove = None
        del npz

        binaryInputNCHW = decode_binary_input(binaryInputNCHWPacked, spatialDistNCHW, pos_len, npz_file)

        # Data of another Quoridor I/O version (v1: 17 / 15, v2: 19 / 17 input channels) cannot train an I/O v3 model.
        # v2 and v3 rows have the same targets but v2 rows lack the repetition inputs (their history is not stored).
        assert binaryInputNCHW.shape[1] == num_bin_features and globalInputNC.shape[1] == num_global_features, (
            f"{npz_file}: {binaryInputNCHW.shape[1]} spatial / {globalInputNC.shape[1]} global input channels, the model"
            f" expects {num_bin_features} / {num_global_features} (training data of an older Quoridor I/O version?)")
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    # Prefetch up to prefetch_depth files *ahead* of the one currently being
    # consumed, so the GPU does not stall at a file boundary waiting on disk +
    # decompress + unpackbits for the next shard.
    # Each in-flight file holds its full expanded (float32) arrays in RAM,
    # so memory scales linearly with prefetch_depth
    # (times world_size, since every rank loads each file).
    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        # Keep a queue of in-flight loads: the head is the file being consumed,
        # and up to prefetch_depth more are loading/loaded behind it.
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1

        while pending:
            future = pending.popleft()
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            # The arrays already hold only this rank's rows (selected in load_npz_file),
            # so the first dim is num_whole_steps * batch_size.
            num_whole_steps = binaryInputNCHW.shape[0] // batch_size

            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            # Top the pipeline back up so prefetch_depth files stay in flight.
            if next_index < len(npz_files):
                logging.info(f"Preloading {npz_files[next_index]} while processing this file")
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1

            for n in range(num_whole_steps):
                start = n * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                    model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                )

                if randomize_symmetries:
                    if modelconfigs.is_quoridor(model_config):
                        symm = int(rand.integers(0, 2))
                        batch_binaryInputNCHW = apply_symmetry_quoridor(batch_binaryInputNCHW, symm)
                        batch_policyTargetsNCMove = apply_symmetry_policy_quoridor(batch_policyTargetsNCMove, symm)
                        batch_valueTargetsNCHW = apply_symmetry_value_targets_quoridor(batch_valueTargetsNCHW, symm)
                    else:
                        symm = int(rand.integers(0, 8))
                        batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                        batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                        batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                        if include_qvalues:
                            batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_quoridor(tensor, symm):
    """
    Apply Quoridor symmetry:
    symm == 0: Identity
    symm == 1: Horizontal Reflection (Left-Right flip)
    """
    if symm == 0:
        return tensor
    assert symm == 1

    out = torch.flip(tensor, dims=[-1]).clone()

    # Swap Channel 5 (Blocked East) <-> Channel 6 (Blocked West)
    ch5 = out[:, 5, :, :].clone()
    ch6 = out[:, 6, :, :].clone()
    out[:, 5, :, :] = ch6
    out[:, 6, :, :] = ch5

    # Fix wall anchor and domain mask channels 14, 15, 16, and in I/O v2 the legal-wall channels 17, 18:
    # Wall anchors/masks are in :8, :8. When 9x9 is flipped, :8 shifts to 1:9.
    # We must flip :8 within :8 so c -> 7 - c.
    for ch in QUORIDOR_WALL_ANCHOR_CHANNELS:
        if ch >= tensor.shape[1]:
            continue
        orig_wall = tensor[:, ch, :8, :8]
        out[:, ch, :, :] = 0.0
        out[:, ch, :8, :8] = torch.flip(orig_wall, dims=[-1])

    return out


def apply_symmetry_policy_quoridor(tensor, symm):
    """
    tensor shape: (B, 18, 9, 9) or (B, 6, 3, 9, 9) or (B, 18, 81)
    """
    if symm == 0:
        return tensor
    assert symm == 1
    orig_shape = tensor.shape
    assert 3 <= len(orig_shape) <= 5
    batch_size = orig_shape[0]

    # Reshape to (B, N, 3, 9, 9) where N is number of policy heads/targets
    t = tensor.view(batch_size, -1, 3, 9, 9).clone()

    out = torch.zeros_like(t)
    # Plane 0 (Pawn): flip horizontally across all 9 columns
    out[:, :, 0, :, :] = torch.flip(t[:, :, 0, :, :], dims=[-1])

    # Plane 1 (V-walls) & Plane 2 (H-walls): flip active 8x8 anchor region (c -> 7 - c)
    for p in (1, 2):
        wall_region = t[:, :, p, :8, :8]
        out[:, :, p, :8, :8] = torch.flip(wall_region, dims=[-1])

    return out.view(orig_shape)


def apply_symmetry_value_targets_quoridor(tensor, symm):
    """
    Spatial value targets: Trajectory (ch 0..1), Wall Graph (ch 2..3)
    tensor shape: (B, C, 9, 9)
    """
    if symm == 0:
        return tensor
    assert symm == 1

    out = torch.zeros_like(tensor)
    # Ch 0 and 1: Trajectory (pawn visited 9x9)
    if tensor.shape[1] >= 2:
        out[:, :2, :, :] = torch.flip(tensor[:, :2, :, :], dims=[-1])
    # Ch 2 and 3: Wall graph (8x8 active wall anchors)
    if tensor.shape[1] >= 4:
        for ch in (2, 3):
            wall_region = tensor[:, ch, :8, :8]
            out[:, ch, :8, :8] = torch.flip(wall_region, dims=[-1])

    return out


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    if modelconfigs.is_quoridor(model_config):
        return (None, None)
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    if modelconfigs.is_quoridor(model_config):
        return batch_binaryInputNCHW, batch_globalInputNC
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC


# ---------------------------------------------------------------------------------------------------------------------
# Quoridor Four-at-a-Table (Q4) training rows (docs/q4/Q4IO.md §3, §3.1, §5, §8).
#
# Same structure as the Duel path above (load_npz_file + prefetch + per-batch random symmetry); the Q4 differences:
# 27 spatial channels on 11 x 11 with the raw-distance channels 10..14 rebuilt from spatialDistNCHW (dist01 of §3),
# globalTargetsNC has the 64 Q4 columns (no padding to the Duel width), and the symmetry is the full D4 group of
# cpp/q4/nn/q4rawsymmetry.cpp (cells, wall anchors with the V <-> H swap, the four blocked-direction channels).
# Relative seats are not permuted by a symmetry (seat identities do not change, Q4IO §3), so globals and the per-seat
# channel order of the targets stay as they are.

Q4_POS_LEN = 11
Q4_POS_AREA = Q4_POS_LEN * Q4_POS_LEN
Q4_NUM_ANCHORS = 10
Q4_DIST_FIRST_CHANNEL = 10
Q4_DIST_NUM_CHANNELS = 5
Q4_DIST_UNREACHABLE = 255
Q4_DIST_SCALE = 64.0
Q4_NUM_GLOBAL_TARGETS = 64
# Spatial input channels (Q4IO §3): cell planes, the blocked N / E / S / W planes, and the wall-anchor plane pairs
# (vertical, horizontal) that swap when the symmetry exchanges the axes. Channel 21 (the anchor domain) is invariant.
Q4_INPUT_CELL_CHANNELS = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
Q4_INPUT_DIR_FIRST_CHANNEL = 6
Q4_INPUT_WALL_PAIRS = ((19, 20), (22, 23))
Q4_INPUT_INVARIANT_CHANNELS = (21,)
# valueTargetsNCHW (Q4IO §8): C0-3 paths (cell planes), C4-11 future walls of relative seat k at (4 + 2k, 5 + 2k).
Q4_VALUE_TARGET_CELL_CHANNELS = (0, 1, 2, 3)
Q4_VALUE_TARGET_WALL_PAIRS = ((4, 5), (6, 7), (8, 9), (10, 11))


def _q4_transform_coords(x, y, sym):
    # q4rawsymmetry.cpp transformCoords
    return [(x, y), (-y, x), (-x, -y), (y, -x), (-x, y), (x, -y), (y, x), (-y, -x)][sym]


def q4_sym_swaps_axes(sym: int) -> bool:
    return sym in (1, 3, 6, 7)


def _build_q4_symmetry_tables():
    """Port of q4rawsymmetry.cpp buildTables, as gather indices on flattened 11 x 11 planes.

    cell_src[sym][dst] = the source cell that lands on cell dst; anchor_src[sym][dst] = the source position
    (ay * 11 + ax) of the anchor that lands on position dst, or Q4_POS_AREA (a zero pad) for the padding row / column;
    dir_map[sym][d] = the direction d moves to."""
    cell_src = np.zeros((8, Q4_POS_AREA), dtype=np.int64)
    anchor_src = np.full((8, Q4_POS_AREA), Q4_POS_AREA, dtype=np.int64)
    dir_map = np.zeros((8, 4), dtype=np.int64)
    dir_dx = (0, 1, 0, -1)
    dir_dy = (1, 0, -1, 0)
    for sym in range(8):
        for c in range(Q4_POS_AREA):
            ox, oy = _q4_transform_coords(c % Q4_POS_LEN - 5, c // Q4_POS_LEN - 5, sym)
            cell_src[sym, (oy + 5) * Q4_POS_LEN + (ox + 5)] = c
        for a in range(Q4_NUM_ANCHORS * Q4_NUM_ANCHORS):
            ax, ay = a % Q4_NUM_ANCHORS, a // Q4_NUM_ANCHORS
            oax, oay = _q4_transform_coords(2 * ax - 9, 2 * ay - 9, sym)
            nax, nay = (oax + 9) // 2, (oay + 9) // 2
            anchor_src[sym, nay * Q4_POS_LEN + nax] = ay * Q4_POS_LEN + ax
        for d in range(4):
            odx, ody = _q4_transform_coords(dir_dx[d], dir_dy[d], sym)
            dir_map[sym, d] = [nd for nd in range(4) if dir_dx[nd] == odx and dir_dy[nd] == ody][0]
    return cell_src, anchor_src, dir_map

Q4_CELL_SRC, Q4_ANCHOR_SRC, Q4_DIR_MAP = _build_q4_symmetry_tables()


def decode_q4_dist_planes(spatialDistNCHW: np.ndarray) -> np.ndarray:
    """Q4IO §3 dist01: 1.0 if unreachable (255), else min(d, 64) / 64."""
    d = spatialDistNCHW.astype(np.float32)
    return np.where(
        spatialDistNCHW == Q4_DIST_UNREACHABLE, np.float32(1.0), np.minimum(d, np.float32(Q4_DIST_SCALE)) / np.float32(Q4_DIST_SCALE)
    ).astype(np.float32)


def decode_q4_binary_input(binaryInputNCHWPacked: np.ndarray, spatialDistNCHW: np.ndarray, npz_file="") -> np.ndarray:
    """Q4 packed bit planes [N,27,16] + raw distances [N,5,11,11] -> float32 spatial inputs [N,27,11,11], symmetry 0."""
    n = binaryInputNCHWPacked.shape[0]
    assert binaryInputNCHWPacked.shape[2] == (Q4_POS_AREA + 7) // 8, f"{npz_file}: packed shape {binaryInputNCHWPacked.shape}"
    binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)[:, :, :Q4_POS_AREA]
    binaryInputNCHW = binaryInputNCHW.reshape(n, binaryInputNCHWPacked.shape[1], Q4_POS_LEN, Q4_POS_LEN).astype(np.float32)
    lo = Q4_DIST_FIRST_CHANNEL
    hi = lo + Q4_DIST_NUM_CHANNELS
    assert spatialDistNCHW.dtype == np.uint8, f"{npz_file}: spatialDistNCHW dtype {spatialDistNCHW.dtype}"
    assert spatialDistNCHW.shape == (n, Q4_DIST_NUM_CHANNELS, Q4_POS_LEN, Q4_POS_LEN), (
        f"{npz_file}: spatialDistNCHW shape {spatialDistNCHW.shape}")
    assert not binaryInputNCHW[:, lo:hi].any(), f"{npz_file}: packed bit planes 10..14 are nonzero"
    binaryInputNCHW[:, lo:hi] = decode_q4_dist_planes(spatialDistNCHW)
    return binaryInputNCHW


def _q4_gather_planes(planes: torch.Tensor, src: np.ndarray) -> torch.Tensor:
    """planes [..., 11, 11] -> out[..., dst] = planes[..., src[dst]] (src == 121 reads zero)."""
    flat = planes.reshape(*planes.shape[:-2], Q4_POS_AREA)
    flat = torch.nn.functional.pad(flat, (0, 1))
    idx = torch.as_tensor(src, device=planes.device)
    return flat.index_select(-1, idx).reshape(planes.shape)


def _q4_apply_wall_pair(tensor: torch.Tensor, out: torch.Tensor, v_ch: int, h_ch: int, sym: int):
    src = Q4_ANCHOR_SRC[sym]
    new_v = _q4_gather_planes(tensor[:, v_ch], src)
    new_h = _q4_gather_planes(tensor[:, h_ch], src)
    if q4_sym_swaps_axes(sym):
        new_v, new_h = new_h, new_v
    out[:, v_ch] = new_v
    out[:, h_ch] = new_h


def apply_symmetry_q4_inputs(tensor: torch.Tensor, sym: int) -> torch.Tensor:
    """Q4 spatial inputs [N,27,11,11] in symmetry 0 -> the inputs of the same position under sym (Q4IO §3)."""
    if sym == 0:
        return tensor
    out = torch.empty_like(tensor)
    cells = list(Q4_INPUT_CELL_CHANNELS)
    out[:, cells] = _q4_gather_planes(tensor[:, cells], Q4_CELL_SRC[sym])
    d0 = Q4_INPUT_DIR_FIRST_CHANNEL
    moved = _q4_gather_planes(tensor[:, d0:d0 + 4], Q4_CELL_SRC[sym])
    for d in range(4):
        out[:, d0 + int(Q4_DIR_MAP[sym, d])] = moved[:, d]
    for (v_ch, h_ch) in Q4_INPUT_WALL_PAIRS:
        _q4_apply_wall_pair(tensor, out, v_ch, h_ch, sym)
    for ch in Q4_INPUT_INVARIANT_CHANNELS:
        out[:, ch] = tensor[:, ch]
    return out


def apply_symmetry_q4_policy(tensor: torch.Tensor, sym: int) -> torch.Tensor:
    """Q4 policy targets [N, C, 363] (3 planes: pawn cells, V anchors, H anchors) -> the targets under sym."""
    if sym == 0:
        return tensor
    n, c = tensor.shape[0], tensor.shape[1]
    t = tensor.reshape(n * c, 3, Q4_POS_LEN, Q4_POS_LEN)
    out = torch.empty_like(t)
    out[:, 0] = _q4_gather_planes(t[:, 0], Q4_CELL_SRC[sym])
    _q4_apply_wall_pair(t, out, 1, 2, sym)
    return out.reshape(tensor.shape)


def apply_symmetry_q4_value_targets(tensor: torch.Tensor, sym: int) -> torch.Tensor:
    """Q4 valueTargetsNCHW [N,12,11,11]: paths (cell map) and future walls (anchor map, V <-> H swap) under sym."""
    if sym == 0:
        return tensor
    out = torch.empty_like(tensor)
    cells = list(Q4_VALUE_TARGET_CELL_CHANNELS)
    out[:, cells] = _q4_gather_planes(tensor[:, cells], Q4_CELL_SRC[sym])
    for (v_ch, h_ch) in Q4_VALUE_TARGET_WALL_PAIRS:
        _q4_apply_wall_pair(tensor, out, v_ch, h_ch, sym)
    return out


def apply_symmetry_q4_batch(batch: dict, sym: int) -> dict:
    """Applies sym to every spatial tensor of a Q4 batch (inputs and targets together); globals are invariant."""
    out = dict(batch)
    out["binaryInputNCHW"] = apply_symmetry_q4_inputs(batch["binaryInputNCHW"], sym)
    out["policyTargetsNCMove"] = apply_symmetry_q4_policy(batch["policyTargetsNCMove"], sym)
    out["valueTargetsNCHW"] = apply_symmetry_q4_value_targets(batch["valueTargetsNCHW"], sym)
    return out


def load_q4_npz_rows(npz_file, batch_size: int = 1, world_size: int = 1, rank: int = 0, model_config=None,
                     include_meta: bool = False):
    """Loads this rank's whole batches of a Q4 npz as numpy arrays (decoded inputs, float targets), symmetry 0.
    include_meta (Q4 I/O v2 nets): also loads `metadataInputNC` [N, 192], a hard error if the file lacks it; I/O v1
    nets (include_meta False) ignore the key."""
    with np.load(npz_file) as npz:
        num_samples = npz["globalInputNC"].shape[0]
        num_whole_steps = num_samples // (batch_size * world_size)
        used = num_whole_steps * world_size * batch_size

        def select_rank_rows(arr):
            arr = arr[:used]
            rest = arr.shape[1:]
            arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
            arr = arr[:, rank]
            return arr.reshape(num_whole_steps * batch_size, *rest)

        for key in ("binaryInputNCHWPacked", "spatialDistNCHW", "globalInputNC", "policyTargetsNCMove",
                    "globalTargetsNC", "valueTargetsNCHW"):
            if key not in npz:
                raise KeyError(f"{npz_file} lacks {key} (not a Q4 training file, docs/q4/Q4IO.md §8)")
        binaryInputNCHWPacked = select_rank_rows(npz["binaryInputNCHWPacked"])
        spatialDistNCHW = select_rank_rows(npz["spatialDistNCHW"])
        globalInputNC = select_rank_rows(npz["globalInputNC"]).astype(np.float32)
        policyTargetsNCMove = select_rank_rows(npz["policyTargetsNCMove"]).astype(np.float32)
        globalTargetsNC = select_rank_rows(npz["globalTargetsNC"]).astype(np.float32)
        scoreDistrN = select_rank_rows(npz["scoreDistrN"]).astype(np.float32)
        valueTargetsNCHW = select_rank_rows(npz["valueTargetsNCHW"]).astype(np.float32)
        metadataInputNC = None
        if include_meta:
            if "metadataInputNC" not in npz:
                raise KeyError(
                    f"{npz_file} lacks metadataInputNC: a Q4 I/O v2 net (metadata encoder) cannot train on data without "
                    f"the style features (docs/q4/Q4IO.md §10); shuffle with -include-meta")
            metadataInputNC = select_rank_rows(npz["metadataInputNC"]).astype(np.float32)
            assert metadataInputNC.shape[1:] == (192,), f"{npz_file}: metadataInputNC shape {metadataInputNC.shape}"

    binaryInputNCHW = decode_q4_binary_input(binaryInputNCHWPacked, spatialDistNCHW, npz_file)
    if model_config is not None:
        num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
        num_global_features = modelconfigs.get_num_global_input_features(model_config)
        assert binaryInputNCHW.shape[1] == num_bin_features and globalInputNC.shape[1] == num_global_features, (
            f"{npz_file}: {binaryInputNCHW.shape[1]} spatial / {globalInputNC.shape[1]} global input channels, the model"
            f" expects {num_bin_features} / {num_global_features}")
    # shuffle.py zero-pads globalTargetsNC to the Duel width (pad_global_targets_nc); the Q4 columns are the first 64.
    assert globalTargetsNC.shape[1] >= Q4_NUM_GLOBAL_TARGETS, f"{npz_file}: globalTargetsNC shape {globalTargetsNC.shape}"
    assert not globalTargetsNC[:, Q4_NUM_GLOBAL_TARGETS:].any(), f"{npz_file}: nonzero globalTargetsNC beyond column 63"
    globalTargetsNC = np.ascontiguousarray(globalTargetsNC[:, :Q4_NUM_GLOBAL_TARGETS])
    assert policyTargetsNCMove.shape[1:] == (3, 3 * Q4_POS_AREA), f"{npz_file}: policyTargetsNCMove shape {policyTargetsNCMove.shape}"
    assert valueTargetsNCHW.shape[1:] == (12, Q4_POS_LEN, Q4_POS_LEN), f"{npz_file}: valueTargetsNCHW shape {valueTargetsNCHW.shape}"
    rows = dict(
        binaryInputNCHW=binaryInputNCHW,
        globalInputNC=globalInputNC,
        policyTargetsNCMove=policyTargetsNCMove,
        globalTargetsNC=globalTargetsNC,
        scoreDistrN=scoreDistrN,
        valueTargetsNCHW=valueTargetsNCHW,
    )
    if metadataInputNC is not None:
        rows["metadataInputNC"] = metadataInputNC   # style features: invariant under the board symmetries
    return rows


def read_npz_training_data_q4(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    model_config: modelconfigs.ModelConfig,
    prefetch_depth: int = 1,
    include_meta: bool = False,
):
    assert pos_len == Q4_POS_LEN, f"Q4 trains on the {Q4_POS_LEN} x {Q4_POS_LEN} board, got pos_len {pos_len}"
    rand = np.random.default_rng(seed=list(os.urandom(12)))

    def load_npz_file(npz_file):
        return (npz_file, load_q4_npz_rows(npz_file, batch_size, world_size, rank, model_config, include_meta))

    if not npz_files:
        return

    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1

        while pending:
            future = pending.popleft()
            (npz_file, rows) = future.result()
            num_whole_steps = rows["globalInputNC"].shape[0] // batch_size
            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            if next_index < len(npz_files):
                logging.info(f"Preloading {npz_files[next_index]} while processing this file")
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1

            for n in range(num_whole_steps):
                start = n * batch_size
                end = start + batch_size
                batch = {key: torch.from_numpy(arr[start:end]).to(device) for key, arr in rows.items()}
                if randomize_symmetries:
                    symm = int(rand.integers(0, 8))
                    batch = apply_symmetry_q4_batch(batch, symm)
                for key in ("binaryInputNCHW", "policyTargetsNCMove", "valueTargetsNCHW"):
                    batch[key] = batch[key].contiguous()
                yield batch
