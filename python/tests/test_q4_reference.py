"""Unit tests for the Q4 Python reference implementation (Plan §5.7 T1).

Each test case tests an individual rules clause with hand-constructed positions:
- Wall conflicts (overlap, end-to-end, crossing, T-junctions allowed, gaps allowed)
- Step blocked by a wall
- Straight jump
- Diagonal jump behind a wall
- Diagonal jump at the board edge
- NO diagonal when the cell behind is a pawn
- Two-pawn jump allowed only under distance restriction:
    * Allowed case
    * Denied case (an ordinary move keeps distance)
- Two-pawn jump blocked by wall between C1 and C2
- Two-pawn jump landing on center wins
- Win by straight jump and by diagonal jump into the center
- No-full-block with eliminated seat whose pawn would be enclosed (allowed)
- Last seat alive wins
- Elimination of seat to move passes the turn
- maxPlies boundary (win on ply maxPlies is a win, otherwise draw)
- N-fold repetition
"""

import pytest
from q4.reference import (
    Pos,
    action_to_index,
    action_to_str,
    compute_distances_to_center,
    eliminate,
    index_to_action,
    legal_moves,
    pawn_moves,
    play,
    str_to_action,
    wall_ok,
)


def test_start_position():
    pos = Pos()
    assert pos.pawn == [(5, 0), (0, 5), (5, 10), (10, 5)]
    assert pos.to_move == 0
    assert pos.walls_left == [7, 7, 7, 7]
    assert pos.alive == [True, True, True, True]
    assert not pos.is_terminal()
    # Seat 0 at (5, 0) can move to (4, 0), (6, 0), (5, 1) -> 3 pawn moves
    assert pawn_moves(pos, 0) == [(4, 0), (5, 1), (6, 0)]
    # All 200 walls are legal from start
    moves = legal_moves(pos)
    assert len(moves) == 3 + 200


def test_action_conversion_roundtrip():
    for idx in range(321):
        act = index_to_action(idx)
        assert action_to_index(act) == idx
        s = action_to_str(act)
        act2 = str_to_action(s)
        assert act == act2


def test_wall_conflicts():
    pos = Pos()
    # Place horizontal wall at e5h -> anchor (4, 4), orient 'h'
    pos.hwalls.add((4, 4))

    # Same orientation overlap at (4, 4)h is forbidden
    assert not wall_ok(pos, 4, 4, 'h')
    # Same orientation end-to-end adjacent (3, 4)h and (5, 4)h are forbidden
    assert not wall_ok(pos, 3, 4, 'h')
    assert not wall_ok(pos, 5, 4, 'h')

    # Crossing at (4, 4)v is forbidden
    assert not wall_ok(pos, 4, 4, 'v')

    # Perpendicular touching / T-junctions at different anchors are allowed:
    # e.g., (3, 4)v, (5, 4)v, (4, 3)v, (4, 5)v
    assert wall_ok(pos, 3, 4, 'v')
    assert wall_ok(pos, 5, 4, 'v')

    # Gaps of same orientation are allowed: (2, 4)h, (6, 4)h, (4, 2)h
    assert wall_ok(pos, 2, 4, 'h')
    assert wall_ok(pos, 6, 4, 'h')
    assert wall_ok(pos, 4, 2, 'h')


def test_step_blocked_by_wall():
    pos = Pos()
    # Seat 0 at (5, 0).
    # Normal moves are (4, 0), (6, 0), (5, 1).
    # Place horizontal wall at (5, 0)h: blocks (5, 0)-(5, 1) and (6, 0)-(6, 1)
    pos.hwalls.add((5, 0))
    moves = pawn_moves(pos, 0)
    assert (5, 1) not in moves
    assert moves == [(4, 0), (6, 0)]


def test_straight_jump():
    pos = Pos()
    # Move seat 0 to (5, 4), seat 1 to (5, 5) [wait, (5,5) is goal, so let's use another area]
    # Let's test at (5, 2) and (5, 3)
    pos.pawn[0] = (5, 2)
    pos.pawn[1] = (5, 3)
    pos.to_move = 0
    # No walls behind (5, 3)
    moves = pawn_moves(pos, 0)
    # (5, 3) is occupied, straight jump to (5, 4) is open
    assert (5, 4) in moves
    # Diagonal moves around (5, 3) to (4, 3) or (6, 3) should NOT be present
    assert (4, 3) not in moves
    assert (6, 3) not in moves


def test_diagonal_jump_behind_wall():
    pos = Pos()
    pos.pawn[0] = (5, 2)
    pos.pawn[1] = (5, 3)
    pos.to_move = 0
    # Place wall behind (5, 3) to block straight step (5, 3)-(5, 4).
    # Wall at (5, 3)h blocks (5, 3)-(5, 4) and (6, 3)-(6, 4).
    pos.hwalls.add((5, 3))
    moves = pawn_moves(pos, 0)
    # Straight jump (5, 4) is blocked by wall
    assert (5, 4) not in moves
    # Diagonal destinations around (5, 3): (4, 3) and (6, 3)
    assert (4, 3) in moves
    assert (6, 3) in moves


def test_diagonal_jump_at_board_edge():
    pos = Pos()
    # Put pawn 0 at (0, 1), pawn 1 at (0, 0)
    pos.pawn[0] = (0, 1)
    pos.pawn[1] = (0, 0)
    pos.to_move = 0
    # Behind (0, 0) going South would be (0, -1) which is off-board
    moves = pawn_moves(pos, 0)
    assert (0, -1) not in moves
    # Diagonal jump to (1, 0) should be available (perpendicular destination)
    assert (1, 0) in moves


def test_no_diagonal_when_cell_behind_is_pawn():
    pos = Pos()
    # Pawn 0 at (5, 1), Pawn 1 at (5, 2), Pawn 2 at (5, 3)
    pos.pawn[0] = (5, 1)
    pos.pawn[1] = (5, 2)
    pos.pawn[2] = (5, 3)
    pos.to_move = 0
    # No wall between (5, 2) and (5, 3)
    moves = pawn_moves(pos, 0)
    # Straight single jump to (5, 3) is blocked because (5, 3) is occupied.
    # Diagonal jumps to (4, 2) and (6, 2) MUST NOT be permitted because straight jump
    # was not blocked by wall or board edge!
    assert (4, 2) not in moves
    assert (6, 2) not in moves


def test_two_pawn_jump_allowed_and_denied_by_distance_restriction():
    # Board center is (5, 5).
    # Case A: Two-pawn jump is DENIED because an ordinary move keeps/decreases distance.
    pos = Pos()
    pos.pawn[0] = (5, 1)  # dist to (5, 5) = 4
    pos.pawn[1] = (5, 2)
    pos.pawn[2] = (5, 3)
    pos.to_move = 0
    # Ordinary moves from (5, 1): (4, 1) has dist 5 (increases), (6, 1) has dist 5 (increases),
    # but (5, 0) has dist 5. Wait! What if there are walls?
    # Without walls, from (5, 1):
    # (4, 1) dist is |4-5| + |1-5| = 1 + 4 = 5 > 4.
    # (6, 1) dist is 5 > 4.
    # (5, 0) dist is 5 > 4.
    # All ordinary moves increase distance from 4 to 5!
    # So without any walls, two-pawn jump to (5, 4) (dist 1) WOULD be allowed!
    moves = pawn_moves(pos, 0)
    assert (5, 4) in moves

    # Case B: Denied because an ordinary move does NOT strictly increase distance.
    # Put pawn 0 at (5, 2) with dist 3.
    # Pawn 1 at (6, 2) (dist 4), Pawn 2 at (7, 2) (dist 5).
    # Ordinary moves from (5, 2): (5, 3) [dist 2, closer!], (4, 2) [dist 4], (5, 1) [dist 4].
    # Here, moving to (5, 3) strictly decreases distance (2 < 3).
    # Even though straight line East over (6, 2) and (7, 2) to (8, 2) is geometrically open,
    # it must be DENIED because ordinary move (5, 3) does not increase distance!
    pos.pawn[0] = (5, 2)
    pos.pawn[1] = (6, 2)
    pos.pawn[2] = (7, 2)
    moves_denied = pawn_moves(pos, 0)
    assert (5, 3) in moves_denied
    assert (8, 2) not in moves_denied  # Two-pawn jump denied!


def test_two_pawn_jump_blocked_by_wall_between_c1_and_c2():
    pos = Pos()
    pos.pawn[0] = (5, 1)
    pos.pawn[1] = (5, 2)
    pos.pawn[2] = (5, 3)
    pos.to_move = 0
    # Wall between (5, 2) and (5, 3): wall at (5, 2)h blocks (5, 2)-(5, 3)
    pos.hwalls.add((5, 2))
    # With a wall between c1 and c2:
    # 1. Straight jump from (5, 2) to (5, 3) is blocked by wall, so DIAGONAL jumps around c1 (5, 2) become legal!
    # 2. Two-pawn jump over c1 and c2 is blocked because wall is between c1 and c2.
    moves = pawn_moves(pos, 0)
    assert (5, 4) not in moves  # Two-pawn jump blocked
    assert (4, 2) in moves      # Diagonal jump around c1 now legal
    assert (6, 2) in moves


def test_two_pawn_jump_landing_on_center_wins():
    pos = Pos()
    # Pawn 0 at (5, 3) (dist 2)
    # Pawn 1 at (5, 4)
    # Pawn 2 at (5, 5) wait, (5, 5) is center! So pawn 2 cannot be on center during the game.
    # What if approach is from (3, 5) -> (4, 5) -> (5, 5)?
    # Pawn 0 at (2, 5) (dist 3)
    # Pawn 1 at (3, 5)
    # Pawn 2 at (4, 5)
    # Landing cell is (5, 5) = CENTER!
    # Check ordinary moves from (2, 5): (2, 4) dist 4 > 3, (2, 6) dist 4 > 3, (1, 5) dist 5 > 3.
    # All ordinary moves increase distance, so two-pawn jump to (5, 5) is legal!
    pos.pawn[0] = (2, 5)
    pos.pawn[1] = (3, 5)
    pos.pawn[2] = (4, 5)
    pos.to_move = 0
    moves = pawn_moves(pos, 0)
    assert (5, 5) in moves

    # Play the move
    pos2 = play(pos, ('p', (5, 5)))
    assert pos2.is_terminal()
    assert pos2.winner == 0
    assert pos2.result_str() == "1+"


def test_win_by_straight_jump_and_diagonal_jump_into_center():
    # Win by straight jump into center:
    pos = Pos()
    pos.pawn[0] = (5, 3)
    pos.pawn[1] = (5, 4)
    pos.to_move = 0
    moves = pawn_moves(pos, 0)
    assert (5, 5) in moves
    p1 = play(pos, ('p', (5, 5)))
    assert p1.is_terminal()
    assert p1.winner == 0

    # Win by diagonal jump into center:
    # Pawn 0 at (4, 4), Pawn 1 at (4, 5).
    # Wall behind (4, 5) at (4, 5)h blocks straight jump (4, 5)-(4, 6).
    pos_diag = Pos()
    pos_diag.pawn[0] = (4, 4)
    pos_diag.pawn[1] = (4, 5)
    pos_diag.hwalls.add((4, 5))
    pos_diag.to_move = 0
    moves_diag = pawn_moves(pos_diag, 0)
    assert (5, 5) in moves_diag  # Diagonal jump into center
    p2 = play(pos_diag, ('p', (5, 5)))
    assert p2.is_terminal()
    assert p2.winner == 0


def test_no_full_block_with_eliminated_seat():
    pos = Pos()
    # Eliminate seat 1 (West, at a6 = (0, 5))
    pos = eliminate(pos, 1)
    assert not pos.alive[1]
    assert pos.pawn[1] is None

    # Even if we completely wall off a region around (0, 5), since seat 1 is eliminated,
    # wall placements that don't block alive seats 0, 2, 3 are legal.
    assert wall_ok(pos, 0, 4, 'h')


def test_last_seat_alive_wins():
    pos = Pos()
    pos = eliminate(pos, 1)
    assert not pos.is_terminal()
    pos = eliminate(pos, 2)
    assert not pos.is_terminal()
    pos = eliminate(pos, 3)
    assert pos.is_terminal()
    assert pos.winner == 0
    assert pos.result_str() == "1+"


def test_elimination_of_seat_to_move_passes_turn():
    pos = Pos()
    pos.to_move = 1
    # Eliminate seat 1
    pos = eliminate(pos, 1)
    assert not pos.alive[1]
    # Next alive clockwise is seat 2
    assert pos.to_move == 2


def test_max_plies_boundary():
    pos = Pos(max_plies=4)
    # Play 3 harmless moves
    pos = play(pos, ('p', (5, 1)))  # ply 1 (seat 0)
    pos = play(pos, ('p', (1, 5)))  # ply 2 (seat 1)
    pos = play(pos, ('p', (5, 9)))  # ply 3 (seat 2)
    assert not pos.is_terminal()
    assert pos.plies == 3

    # Move 4: non-winning move on maxPlies -> Draw!
    p_draw = play(pos, ('p', (9, 5)))  # ply 4 (seat 3)
    assert p_draw.is_terminal()
    assert p_draw.is_draw
    assert p_draw.result_str() == "Draw"

    # But if ply 4 lands on center -> Win!
    pos_win = pos.copy()
    pos_win.pawn[3] = (6, 5)
    pos_win.to_move = 3
    p_win = play(pos_win, ('p', (5, 5)))
    assert p_win.is_terminal()
    assert p_win.winner == 3
    assert p_win.result_str() == "4+"


def test_repetition_draw():
    # Test threefold repetition with repetition_draw_count=3
    pos = Pos(repetition_draw_count=3)
    # Seat 0 moves (5, 0) -> (4, 0) -> (5, 0) ...
    # All 4 seats do a cycle back to initial positions
    for _ in range(2):
        pos = play(pos, ('p', (4, 0)))
        pos = play(pos, ('p', (0, 4)))
        pos = play(pos, ('p', (4, 10)))
        pos = play(pos, ('p', (10, 4)))

        pos = play(pos, ('p', (5, 0)))
        pos = play(pos, ('p', (0, 5)))
        pos = play(pos, ('p', (5, 10)))
        pos = play(pos, ('p', (10, 5)))

    # Initial position has now occurred 3 times!
    assert pos.is_terminal()
    assert pos.is_draw
    assert pos.result_str() == "Draw"


def test_perft_pinned_numbers():
    from q4.reference import perft

    # 1. Start position
    start = Pos()
    assert perft(start, 1) == 203
    assert perft(start, 2) == 40445

    # 2. Pawn-heavy position matching C++ test (depths 1..6)
    p = Pos()
    p.walls_left = [0, 0, 0, 0]
    p.pawn[0] = (5, 4)
    p.pawn[1] = (4, 5)
    p.pawn[2] = (5, 6)
    p.pawn[3] = (6, 5)

    # Surrounding box of walls
    p.hwalls.add((3, 3))
    p.hwalls.add((5, 3))
    p.hwalls.add((3, 6))
    p.hwalls.add((5, 6))
    p.vwalls.add((3, 4))
    p.vwalls.add((6, 4))

    assert perft(p, 1) == 3
    assert perft(p, 2) == 7
    assert perft(p, 3) == 15
    assert perft(p, 4) == 35
    assert perft(p, 5) == 51
    assert perft(p, 6) == 111
