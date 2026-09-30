"""Opening sampling.

Engines are (near-)deterministic, so game diversity comes from the referee: each opening is a
short random move sequence sampled from the arbiter's legal moves. Every opening is later played
twice with colours swapped.
"""
import json
import os
import random


def is_wall(move):
    return move[-1] in "hv"


def move_weights(moves, wall_frac):
    """Pawn moves get weight 1 each; walls share a total mass so that they make up wall_frac of it."""
    pawns = [m for m in moves if not is_wall(m)]
    walls = [m for m in moves if is_wall(m)]
    if not walls or wall_frac <= 0.0:
        return pawns, [1.0] * len(pawns)
    if not pawns:
        return walls, [1.0] * len(walls)
    wall_mass = len(pawns) * wall_frac / (1.0 - wall_frac)
    return pawns + walls, [1.0] * len(pawns) + [wall_mass / len(walls)] * len(walls)


def sample_one(position, plies, wall_frac, rng):
    """position: object with clear(), legal_moves() -> list[str] (for the side to move),
    play(color, move) -> bool, winner() -> str|None. Returns a list of moves (may be shorter than
    plies if the game ends, which cannot happen for small plies)."""
    position.clear()
    moves = []
    for ply in range(plies):
        if position.winner() is not None:
            break
        legal = sorted(position.legal_moves())  # sort: independent of the engine's listing order
        if not legal:
            break
        cands, weights = move_weights(legal, wall_frac)
        mv = rng.choices(cands, weights=weights, k=1)[0]
        color = "b" if ply % 2 == 0 else "w"
        if not position.play(color, mv):
            raise RuntimeError("arbiter rejected its own legal move %s" % mv)
        moves.append(mv)
    return moves


def sample_openings(position, count, plies, wall_frac, seed, max_attempts=50):
    """Sample `count` distinct openings. Opening k depends only on (seed, k, previous openings),
    so the list is reproducible and extending count keeps the earlier openings unchanged."""
    out = []
    seen = set()
    for k in range(count):
        for attempt in range(max_attempts):
            rng = random.Random("%s:%d:%d" % (seed, k, attempt))
            mv = sample_one(position, plies, wall_frac, rng)
            if tuple(mv) not in seen:
                break
        seen.add(tuple(mv))
        out.append(mv)
    return out


def load_or_create(path, position_factory, count, plies, wall_frac, seed):
    """Cache openings in `path` (JSON). Reuses the file when the parameters match and it has enough
    openings; refuses to silently change openings of an existing run."""
    params = {"plies": plies, "wall_frac": wall_frac, "seed": seed}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data["params"] != params:
            raise SystemExit("openings file %s was made with %s, but this run asks for %s; "
                             "use a new --out directory or the old parameters" % (path, data["params"], params))
        if len(data["openings"]) >= count:
            return data["openings"]
    position = position_factory()
    try:
        openings = sample_openings(position, count, plies, wall_frac, seed)
    finally:
        close = getattr(position, "close", None)
        if close:
            close()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"params": params, "openings": openings}, f, indent=1)
    os.replace(tmp, path)
    return openings
