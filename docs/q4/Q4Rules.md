# Four at a Table (Q4): Rules and Mechanics Specification

This document provides the formal and unambiguous rules specification for the four-player Quoridor mode
"Four at a Table" (abbreviated **Q4**) in KataQuoridor, based on `GameRules.md` and the implementation decisions
in Section 4 of the Q4 Plan.

---

## 1. Board Geometry and Coordinates

1. **Board Size:** 11 columns × 11 rows ($11 \times 11$).
2. **Coordinates:**
   - Cartesian coordinates $(x, y)$ where $x \in [0, 10]$ and $y \in [0, 10]$, with $(0, 0)$ at the lower-left.
   - Text notation: column letters `a` through `k` ($x = 0 \dots 10$, letter `i` is never skipped), followed by 1-based row number `1` through `11` ($y = 0 \dots 10$).
   - Examples: lower-left is `a1` $(0, 0)$, upper-right is `k11` $(10, 10)$, center is `f6` $(5, 5)$.
   - Linear cell index: $c = y \times 11 + x \in [0, 120]$. The center cell `f6` is index $5 \times 11 + 5 = 60$.
3. **Wall Anchors:**
   - A wall anchor is an intersection $(ax, ay)$ where $ax \in [0, 9]$ and $ay \in [0, 9]$.
   - Linear anchor index: $a = ay \times 10 + ax \in [0, 99]$.
   - Orientation is either Vertical (`v`) or Horizontal (`h`).
   - Anchor text: anchor column `a` through `j`, anchor row `1` through `10`, followed by orientation, e.g., `e5h`, `j10v`.

---

## 2. Seats, Goals, and Turn Order

1. **Seats:** Exactly 4 seats, indexed $0 \dots 3$, corresponding to Players $1 \dots 4$:
   - **Seat 0 (Player 1, South):** Starts at `f1` $(5, 0)$, linear index 5.
   - **Seat 1 (Player 2, West):** Starts at `a6` $(0, 5)$, linear index 55.
   - **Seat 2 (Player 3, North):** Starts at `f11` $(5, 10)$, linear index 115.
   - **Seat 3 (Player 4, East):** Starts at `k6` $(10, 5)$, linear index 65.
2. **Goal:**
   - All players share a single, identical goal cell: the board center `f6` $(5, 5)$, linear index 60.
3. **Turn Order:**
   - Strictly **clockwise** around the board: Seat 0 $\to$ Seat 1 $\to$ Seat 2 $\to$ Seat 3 $\to$ Seat 0 $\dots$
   - Seat 0 takes the first turn.
   - Eliminated seats are skipped in the turn order.
4. **Initial Walls:**
   - Default: 7 walls per seat (configurable between 0 and 10 per seat via `initialWalls[4]`).

---

## 3. Walls and Conflict Rules

1. **Wall Span:**
   - A vertical wall at anchor $(ax, ay)$ blocks orthogonal passage across two cell boundaries:
     $$(ax, ay) \longleftrightarrow (ax+1, ay) \quad \text{and} \quad (ax, ay+1) \longleftrightarrow (ax+1, ay+1)$$
   - A horizontal wall at anchor $(ax, ay)$ blocks orthogonal passage across two cell boundaries:
     $$(ax, ay) \longleftrightarrow (ax, ay+1) \quad \text{and} \quad (ax+1, ay) \longleftrightarrow (ax+1, ay+1)$$
2. **Conflict Rules:** A newly placed wall must not conflict with any existing wall:
   - **Same Orientation:**
     - Horizontal walls conflict if they have the same $ay$ and $|ax_1 - ax_2| \le 1$.
     - Vertical walls conflict if they have the same $ax$ and $|ay_1 - ay_2| \le 1$.
     (Overlapping and direct end-to-end adjacent walls of the same orientation are forbidden).
   - **Opposite Orientation:**
     - A horizontal wall and a vertical wall conflict if and only if they share the exact same anchor $(ax, ay)$ (crossing is forbidden).
     - Perpendicular walls touching at T-junctions or corners with different anchors are legal.
3. **Wall Inventory:** A player must have at least one wall remaining to place a wall. When placed, their wall count decreases by 1.

---

## 4. Pawn Movement and Jumps

On their turn, the active player may move their pawn to an orthogonally adjacent cell or jump over pawns. No pawn may enter an occupied cell, move through the outer board boundary, or cross any wall.

Let $C_0$ be the current position of the active pawn.

### 4.1 Normal Step
- To an orthogonally adjacent cell $C_1$ (N, S, E, W): legal if $C_1$ is on the board, no wall blocks $C_0 \leftrightarrow C_1$, and $C_1$ is unoccupied.

### 4.2 Straight Single Jump
- If an adjacent cell $C_1$ is occupied by another pawn (any alive opponent), the player may attempt a straight jump to $C_2$ (the cell immediately beyond $C_1$ in the same direction):
  - Legal if and only if:
    1. No wall blocks $C_0 \leftrightarrow C_1$;
    2. $C_2$ is on the board;
    3. No wall blocks $C_1 \leftrightarrow C_2$;
    4. $C_2$ is unoccupied.

### 4.3 Diagonal Jump
- A diagonal jump around the adjacent pawn at $C_1$ is considered **only when** the straight jump to $C_2$ is blocked specifically by:
  - The **board edge** (i.e. $C_2$ is off-board); OR
  - A **wall** blocking $C_1 \leftrightarrow C_2$.
- **Crucial Rule on Pawn Obstruction:** If $C_2$ is on the board and no wall blocks $C_1 \leftrightarrow C_2$, but $C_2$ is **occupied by a pawn**, diagonal jumps are **NOT permitted**. (In that situation, only a two-pawn jump may apply if eligible).
- When eligible (straight jump blocked by edge or wall), the player may branch to either perpendicular cell adjacent to $C_1$ ($C_{diag}$):
  - $C_{diag}$ is legal if and only if:
    1. No wall blocks $C_0 \leftrightarrow C_1$;
    2. $C_{diag}$ is on the board;
    3. No wall blocks $C_1 \leftrightarrow C_{diag}$;
    4. $C_{diag}$ is unoccupied.

### 4.4 Four-Player Two-Pawn Jump
If two pawns consecutively occupy cells $C_1$ and $C_2$ in the same orthogonal direction directly in front of $C_0$:
- The destination cell is $C_3$, directly beyond $C_2$ in the same straight line.
- A two-pawn jump is subject to a strict bypass restriction:
  > **Two-Pawn Jump Restriction:** The two-pawn jump to $C_3$ is permitted if and only if:
  > 1. All geometric conditions are met:
  >    - $C_3$ is on the board;
  >    - No wall blocks $C_0 \leftrightarrow C_1$, $C_1 \leftrightarrow C_2$, and $C_2 \leftrightarrow C_3$;
  >    - $C_3$ is unoccupied.
  > 2. **Distance Condition:** **Every** ordinary legal pawn move (normal steps, straight single jumps, diagonal jumps) strictly increases the player's shortest-path distance to the goal, OR there are no ordinary legal pawn moves available at all.
  >
  > Formally: For all ordinary legal pawn moves with destination $D$:
  > $$\text{distToCenter}(D) > \text{distToCenter}(C_0)$$
  > where $\text{distToCenter}(c)$ is the shortest-path distance on the cell graph considering only walls and board edges (pawns are ignored for distance calculation).
- Diagonal branching off the second pawn ($C_2$) is never permitted.
- A two-pawn jump landing on the center `f6` is legal (if the restriction is satisfied) and immediately wins the game.

### 4.5 Move Deduplication
- Multiple jump routes might theoretically reach the same destination cell. Any destination cell is listed at most once in the legal action set.

---

## 5. No-Full-Block Rule

1. A wall placement is legal only if, after placement, every **currently alive** player still has at least one path to the central goal cell `f6` $(5, 5)$.
2. Paths are calculated on the grid considering walls and board boundaries; pawn positions do not block paths.
3. Eliminated players are completely ignored for this check. Enclosed board regions that contain no alive pawns are permitted.

---

## 6. Elimination

1. Elimination is an external protocol/arbitration event (`eliminate <seat>`), not a move made by a player.
2. Effects of eliminating a seat:
   - Its pawn is permanently removed from the board.
   - Its placed walls remain on the board and continue to obstruct movement.
   - Its unplaced walls are discarded.
   - Its pawn no longer obstructs any movement or jumps.
   - The seat no longer takes turns and is skipped in turn progression.
   - The seat is excluded from the no-full-block path requirement.
   - If the eliminated seat was currently the side to move, the turn immediately passes to the next alive seat in clockwise order.
3. Elimination does not advance the ply count.
4. An elimination event can be undone via the undo stack.
5. If eliminations reduce the number of alive seats to 1, that last remaining seat immediately wins the game.

---

## 7. KataQuoridor Rules and Terminal States

1. **Win Condition:**
   - A player whose pawn moves to `f6` $(5, 5)$ wins immediately.
   - If 3 players are eliminated, the sole remaining alive player wins immediately.
   - There is exactly one winner; all other players lose.
2. **Ply Limit (`maxPlies` = 400 default):**
   - Each legal pawn move or wall placement counts as 1 ply. (Eliminations do not count).
   - If the game reaches `maxPlies` plies without a winner, the game immediately ends in a **Draw**.
   - If a pawn reaches the goal on the `maxPlies`-th ply, that move wins the game (not a draw).
   - In a draw, all seats alive at that moment share the draw; any previously eliminated seats are counted as losses.
3. **Repetition Draw (`repetitionDrawCount` = N, default 0 = off):**
   - When enabled ($N \ge 2$, typically $N = 3$): the game immediately ends in a draw if the exact same position occurs for the $N$-th time.
   - A position consists of:
     - All 4 pawn coordinates (or $-1$ if eliminated);
     - The set of all placed walls;
     - The wall count of every seat;
     - The alive mask of seats;
     - The seat to move.
   - Because walls are never removed and eliminations are irreversible in forward play, only positions occurring since the last wall placement or elimination can repeat.
4. **Result Notation:**
   - `1+`, `2+`, `3+`, or `4+` when that respective player (Seat $0 \dots 3$) wins.
   - `Draw` when the game ends in a draw.
   - `none` when the game is ongoing.

---

## 8. Action Encoding

Total number of discrete actions: `Q4::NUM_ACTIONS = 321`.

| Action Index Range | Count | Type | Encoding / Text |
|:---:|:---:|:---:|:---|
| $0 \dots 120$ | 121 | Pawn Destination | Cell coordinate $c = y \times 11 + x$, written e.g. `f2`, `a6`, `f6`. |
| $121 \dots 220$ | 100 | Vertical Wall | Anchor $a = ay \times 10 + ax$, index $121 + a$, written e.g. `a1v`, `e5v`, `j10v`. |
| $221 \dots 320$ | 100 | Horizontal Wall | Anchor $a = ay \times 10 + ax$, index $221 + a$, written e.g. `a1h`, `e5h`, `j10h`. |
