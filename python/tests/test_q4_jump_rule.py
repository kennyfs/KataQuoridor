"""The diagonal-jump rule (docs/q4/Q4Rules.md 4.3-4.5): the table of the rule in the reference, the rule version in
SGF records (jump=diag), and a differential check of C++ against the reference on crowded random positions."""

import pytest

from q4.jump_rule_stats import Cpp, compare, random_positions
from q4.record import Q4Record
from q4.reference import Pos, pawn_moves, play
from tests.test_q4_differential import find_katago_bin


def mk(pawns, walls_h=(), walls_v=()):
    pos = Pos()
    for s, c in enumerate(pawns):
        if c is None:
            pos.alive[s] = False
            pos.pawn[s] = None
        else:
            pos.pawn[s] = c
    pos.hwalls.update(walls_h)
    pos.vwalls.update(walls_v)
    pos.to_move = 0
    return pos


def moves(pos, old=False):
    pos.old_jump_rule = old
    try:
        return set(pawn_moves(pos, 0))
    finally:
        pos.old_jump_rule = False


# Mover (2,2) going North: C1 (2,3), C2 (2,4), C3 (2,5), D (1,3) / (3,3).
LINE = [(2, 2), (2, 3), (2, 4), (9, 9)]


def test_c2_pawn_gives_diagonals_and_denies_jump_by_distance():
    pos = mk(LINE, walls_v={(2, 1)})   # (2,1)v closes (3,2); (1,2) and (2,1) increase the distance
    m = moves(pos)
    assert {(1, 3), (3, 3)} <= m and (2, 5) not in m
    o = moves(pos, old=True)
    assert (1, 3) not in o and (3, 3) not in o and (2, 5) in o


def test_wall_c1_d_pawn_on_d():
    assert {(1, 3), (3, 3)} & moves(mk(LINE, walls_v={(2, 3)})) == {(1, 3)}
    assert {(1, 3), (3, 3)} & moves(mk([(2, 2), (2, 3), (2, 4), (1, 3)])) == {(3, 3)}


def test_wall_c1_c2_wall_c2_c3_pawn_on_c3():
    m = moves(mk(LINE, walls_h={(2, 3)}))            # wall C1|C2: diagonals as before, no jump
    assert {(1, 3), (3, 3)} <= m and (2, 5) not in m
    assert (2, 5) not in moves(mk(LINE, walls_h={(2, 4)}))      # wall C2|C3
    m = moves(mk([(2, 2), (2, 3), (2, 4), (2, 5)]))              # pawn on C3
    assert (2, 5) not in m and {(1, 3), (3, 3)} <= m


def test_c2_empty_eliminated_and_edge():
    m = moves(mk([(2, 2), (2, 3), (7, 7), (9, 9)]))
    assert (2, 4) in m and not {(1, 3), (3, 3)} & m
    m = moves(mk([(2, 2), (2, 3), None, (9, 9)]))                # eliminated pawn on C2's cell: not a pawn
    assert (2, 4) in m and not {(1, 3), (3, 3)} & m
    m = moves(mk([(2, 9), (2, 10), (9, 9), (8, 8)]))
    assert {(1, 10), (3, 10)} <= m


def test_distance_condition_with_diagonals_closed_and_diagonal_onto_centre():
    m = moves(mk(LINE, walls_v={(2, 1), (2, 3), (1, 3)}))
    assert not {(1, 3), (3, 3)} & m and (2, 5) in m
    pos = mk([(4, 4), (4, 5), (4, 6), (9, 9)])
    assert (5, 5) in moves(pos)
    assert play(pos, ('p', (5, 5))).winner == 0


def test_sgf_rule_version():
    rec = Q4Record(events=[{"a": "f2"}])
    line = rec.to_sgf_line()
    assert "jump=diag" in line and Q4Record.from_sgf_line(line).jump_diag
    old = line.replace("jump=diag,", "")
    back = Q4Record.from_sgf_line(old)
    assert not back.jump_diag and back.to_dict()["jumpDiag"] is False
    assert "jump=diag" not in back.to_sgf_line()
    assert back.replay().old_jump_rule and not Q4Record.from_sgf_line(line).replay().old_jump_rule


def test_differential_crowded_positions():
    cpp = Cpp(find_katago_bin())
    try:
        s = compare(cpp, random_positions(1500, seed=7))
    finally:
        cpp.close()
    assert s["mismatches"] == 0
    assert s["mover_line"] >= 100 and s["any_seat_changed"] >= 100
