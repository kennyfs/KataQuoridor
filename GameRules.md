# Quoridor — Complete Game Rules

> **KataQuoridor implements the Duel mode natively** (9 × 9, two players, 10 walls each). The Race and
> Four-at-a-Table sections below formalize the full rule family for reference and future variant support,
> with coordinates fully aligned with the engine's coordinate system.
>
> KataQuoridor adds rules of its own to the Duel: a **draw after 300 plies**, **komi**, a **fence handicap** and an
> optional **draw by repetition** (off in the standard game). They are described in
> [KataQuoridor rules](#kataquoridor-rules-draw-komi-and-fence-handicap) at the end; with the standard settings
> only the 300-ply draw changes the game.

## Game Overview

Quoridor is a turn-based, deterministic, perfect-information board game. Players move pawns across a rectangular grid while placing walls to obstruct routes.

On each turn, the player performs exactly **one action**:

1. Move their pawn to a legal destination square; or
2. Place one wall.

A player wins by reaching their designated goal. In the two-player modes (Duel and Race), this is the opposite goal row. In the four-player mode, all players share a single goal cell in the center of the board.

A wall may never completely cut off a player from their goal. Every legal wall placement must leave a route to the goal for every player who is still in the game.

## Board and Coordinates

The board consists of rectangular cells arranged in rows and columns.

Coordinates are defined using Cartesian coordinates where $y$ increases upward:

* The lower-left cell is `(0, 0)`.
* `x` increases to the right (from $0$ to $\text{columns}-1$).
* `y` increases upward (from $0$ to $\text{rows}-1$).
* A cell is therefore identified as `(x, y)`.

In KataQuoridor's text notation (QTP, SGF viewer), a cell is written as a column letter `a`, `b`, `c`, ... (`x` = 0, 1, 2, ...) followed by a 1-based row number `1`, `2`, `3`, ... (`y` = 0, 1, 2, ...).
For example, on a 9 × 9 board, `(0, 0)` is `a1` and `(4, 8)` is `e9`.

A wall is written as its anchor position `(x, y)` in the same notation (which ranges up to $\text{columns}-2$ and $\text{rows}-2$), followed by `h` (horizontal) or `v` (vertical): `e2h` is the horizontal wall at `(4, 1)`, between rows 2 and 3 under columns `e` and `f`.

### Screen / Matrix Coordinate Transformation

In engines or screen representations that index rows from top to bottom as `(r, c)` (where row index `r = 0` is the top row and increases downward, and column index `c = 0` is the leftmost column and increases rightward), the conversion to and from the canonical Cartesian `(x, y)` is:

$$r = (\text{rows} - 1) - y, \quad c = x$$

$$x = c, \quad y = (\text{rows} - 1) - r$$

All rules in this document are specified in canonical Cartesian `(x, y)` notation with `(0, 0)` at the lower-left.

Players may move only between orthogonally adjacent cells: up, down, left, or right. Moving outside the board is never legal.

## Walls

A wall occupies the boundary between adjacent cells and blocks movement across that boundary.

Every wall has:

* an anchor position `(x, y)`;
* an orientation: horizontal or vertical.

A wall position is valid only when:

$$
0 \le x \le \text{columns}-2
$$

and

$$
0 \le y \le \text{rows}-2.
$$

Each wall spans the equivalent of two adjacent cell boundaries.

### Vertical Wall

A vertical wall at `(x, y)` blocks both directions across these two boundaries:

* `(x, y) ↔ (x+1, y)`
* `(x, y+1) ↔ (x+1, y+1)`

Thus, a vertical wall separates the two columns at `x` and `x+1` over two consecutive rows.

### Horizontal Wall

A horizontal wall at `(x, y)` blocks both directions across:

* `(x, y) ↔ (x, y+1)`
* `(x+1, y) ↔ (x+1, y+1)`

Thus, a horizontal wall separates the two rows at `y` and `y+1` over two consecutive columns.

### Wall Conflicts

A newly placed wall must not conflict with any existing wall.

#### Same orientation

Two walls of the same orientation conflict when they overlap or are immediately adjacent along their shared span.

For horizontal walls, two walls conflict when they have the same `y` coordinate and their `x` coordinates differ by at most one:
$$y_1 = y_2 \quad \text{and} \quad |x_1 - x_2| \le 1$$

For vertical walls, two walls conflict when they have the same `x` coordinate and their `y` coordinates differ by at most one:
$$x_1 = x_2 \quad \text{and} \quad |y_1 - y_2| \le 1$$

Therefore, a same-orientation wall cannot overlap another wall or extend directly from it in a way that would create a continuous three-segment wall.

#### Opposite orientations

A horizontal wall and a vertical wall conflict only when they have the same `(x, y)` position. This represents the two walls crossing at their central intersection.

Perpendicular walls may otherwise touch one another. In particular, a wall may have an endpoint coincide with the midpoint of a perpendicular wall (forming a "T" junction), or two walls may meet at their endpoints.

For example, a horizontal wall at `(0, 0)` and a vertical wall at `(1, 0)` may coexist.

Walls of the same orientation may also be separated by an arbitrary gap; for example, horizontal walls at `(0, 0)` and `(0, 2)` may coexist.

### Wall Inventory

Each player has a limited number of walls. Once a player's wall supply reaches zero, that player can no longer place walls and must move their pawn instead.

## Pawn Movement

A normal pawn move moves the pawn to one orthogonally adjacent, unoccupied cell, provided that no wall blocks the boundary between the two cells.

A pawn cannot move through or over a wall.

Players cannot move through the edge of the board.

A player must choose either a pawn move or a wall placement on a turn; they cannot perform both.

## Jumping Over Pawns

A pawn may interact with another pawn occupying an adjacent cell.

### Direct Jump

If an opponent's pawn is immediately adjacent in the chosen direction, the player may normally jump directly over it.

The destination is the cell immediately beyond the opponent.

A direct jump is legal only when:

* that destination lies within the board;
* no wall blocks the boundary between the opponent and the destination;
* the destination is not occupied by another pawn.

### Diagonal Jump

If an adjacent opponent pawn cannot be jumped over directly because:

* the board edge lies immediately behind it, or
* a wall blocks the boundary immediately behind it,

the player may move diagonally around that pawn.

The possible diagonal destinations are the two cells adjacent to the opponent perpendicular to the original direction of approach.

A diagonal destination is legal only if:

* it lies within the board;
* there is no wall between the opponent's cell and the diagonal destination;
* the destination is not occupied by another pawn.

**Restriction:** If the direct straight jump is unobstructed, moving diagonally around the pawn is **not permitted**. Diagonal jumps are allowed only when jumping directly over is blocked.

## Four-Player Two-Pawn Jump

The four-player mode introduces an additional case that cannot occur in a two-player game.

If two other pawns stand consecutively in the same orthogonal direction directly in front of the active pawn, the active pawn may potentially jump over both in a single straight leap.

For example, if the active pawn at $C_0$ is followed in a straight line by two occupied cells $C_1$ and $C_2$, the player may jump over both to cell $C_3$ provided that:

* the two pawns are directly consecutive in a straight line;
* the cell beyond the second pawn ($C_3$) is on the board;
* no wall blocks any of the three boundaries ($C_0 \leftrightarrow C_1$, $C_1 \leftrightarrow C_2$, and $C_2 \leftrightarrow C_3$);
* the landing cell ($C_3$) is unoccupied.

However, this two-pawn jump is subject to a strict restriction:

> A two-pawn jump is permitted only when every ordinary legal move (normal step, single jump, or diagonal jump) would strictly increase the player's remaining shortest-path distance to the goal (or when no ordinary legal move exists).

In other words, the two-pawn jump is available only as a bypass when there is no ordinary move that keeps the player at least as close to the goal as before.

For the four-player mode, "distance to the goal" is measured by the shortest path through the current wall configuration to the central goal cell (pawns do not block distance paths).

A two-pawn jump must be straight; if the landing cell directly behind the second pawn is blocked, the player cannot branch diagonally off the second pawn.

## Goals and Winning

### Duel

Each player has a different goal side.

Player 1 (Black in KataQuoridor) starts on the top side, at `e9 = (4, 8)`, and must reach the opposite, bottom side (`y = 0`, row 1).

Player 2 (White) starts on the bottom side, at `e1 = (4, 0)`, and must reach the opposite, top side (`y = 8`, row 9).

A player wins immediately when their pawn enters any cell on their goal row.

### Race

Both players start on the bottom row (`y = 0`, row 1) and race toward the top row (`y = 12`, row 13).

A player wins immediately when their pawn reaches any cell on the top row.

### Four at a Table

All four players share exactly one goal:

> the single cell at the center of the board (`f6 = (5, 5)`).

A player wins immediately when their pawn reaches the central cell (including by a direct jump, diagonal jump, or two-pawn jump).

There is exactly one winner. The other three players lose. If three players are eliminated, the last remaining player wins immediately.

## No-Full-Block Rule

A wall placement is legal only if it does not completely eliminate every route from any player who is still in the game to that player's goal.

Before accepting a wall, the game considers the board as a graph:

* each cell is a graph vertex;
* two orthogonally adjacent cells are connected when no wall blocks their boundary;
* pawn positions do not block graph paths.

A wall may therefore be placed only when every active player still has at least one path to their goal.

### Duel (and Race)

After every wall placement, both players must still have a path to their respective goal rows.

### Four at a Table

After every wall placement, every player who is still alive must still have a path to the central goal cell.

An eliminated player no longer participates in this condition.

## Game Modes

### Duel

**Board:** 9 × 9  
**Players:** 2  
**Walls:** 10 per player  

Initial positions:

* Player 1 (Black): `(4, 8)`, i.e. `e9`, the middle of the top row ($y = 8$)
* Player 2 (White): `(4, 0)`, i.e. `e1`, the middle of the bottom row ($y = 0$)

Player 1 moves toward `y = 0` (row 1).

Player 2 moves toward `y = 8` (row 9).

Player 1 (Black) takes the first turn. There is no pass: the players strictly alternate.

The first player to reach their goal row wins.

### Race

**Board:** 9 × 13  
**Players:** 2  
**Walls:** 10 or 15 per player (default 15 in standard setup)  

Both players begin on the bottom row (`y = 0`, row 1), spaced two columns apart from the center column (`e`):

* Player 1: `(2, 0)`, i.e. `c1`
* Player 2: `(6, 0)`, i.e. `g1`

Both players move toward the top row, `y = 12` (row 13).

Player 1 takes the first turn.

The first player to reach any cell on the top row (`y = 12`) wins immediately.

The wall supply is selected when the game is configured and is the same for both players (15 walls each in standard play; 10 walls in short setup).

### Four at a Table

**Board:** 11 × 11  
**Players:** 4  
**Walls:** 7 per player  

Initial positions are at the midpoint of each side:

* Player 1, South: `(5, 0)`, i.e. `f1`
* Player 2, West: `(0, 5)`, i.e. `a6`
* Player 3, North: `(5, 10)`, i.e. `f11`
* Player 4, East: `(10, 5)`, i.e. `k6`

The common goal is the central cell:

* `(5, 5)`, i.e. `f6`

The turn order is strictly **clockwise** around the board:

1. Player 1 (South)
2. Player 2 (West)
3. Player 3 (North)
4. Player 4 (East)
5. repeat

Player 1 takes the first turn.

A player who reaches the center wins immediately.

#### Elimination

A player may be eliminated, for example because they miss their turn or exceed time limits.

When a player is eliminated:

* their pawn is removed from the board;
* their existing walls remain on the board;
* their unplaced walls are discarded;
* their walls continue to affect movement;
* their pawn no longer blocks movement by other players;
* they no longer take turns;
* they no longer need to retain a path to the goal for the no-full-block rule.

Turns continue among the remaining players in clockwise order.

If only one player remains alive, that player wins immediately.

## Turn Progression

At the end of a legal action, the turn passes to the next player in turn order.

In the two-player modes, this simply alternates between the two players.

In the four-player mode, eliminated players are skipped.

If a player reaches their goal as the result of a pawn move, the game ends immediately and the turn does not pass.

Once a winner has been determined, no further actions are legal.

## KataQuoridor rules: draw, komi and fence handicap

These rules are KataQuoridor's, not part of the published Quoridor rules. They are defined precisely in
[docs/QuoridorIOv2.md](docs/QuoridorIOv2.md); all quantities are from White's point of view.

### Draw after 300 plies

If the game reaches `maxPlies` plies (default **300**, counting both players' moves from the start of the game)
and no pawn has reached its goal row, the game ends in a **draw**. A pawn that reaches its goal on the 300th ply
still wins.

### Draw by repetition (optional, off in the standard game)

With `repetitionDrawCount` = N (0 = off, the default; e.g. 3 for threefold repetition), the game is a **draw the
moment a position occurs for the N-th time**. A position is both pawns, all walls, the walls left of each player
and whose turn it is (not the number of plies played). Since walls are never removed, only positions since the
last wall placement can repeat. The 300-ply draw still applies. KataQuoridor's own training games use N = 3; games
against other engines use the standard rules (off).

### Tempo and komi

When a pawn reaches its goal, let `d` be the other player's remaining shortest-path distance to its goal row
(walls only, at least 1). The **tempo** is

* `t = d` if White's pawn arrived (W+d), and
* `t = 1 − d` if Black's pawn arrived (B+d).

So `t = 0` is an equal race that Black, moving first, wins by one step (B+1), and one step of `t` is one move of
the race.

**Komi** is a half-integer (…, −1.5, −0.5, +0.5, +1.5, …; at most 20.5 either way). **White wins if and only if
`t + komi > 0`**, otherwise Black wins. The game still ends the moment a pawn arrives, so with komi the side whose
pawn arrived can lose. Komi never produces a draw.

* The **standard game has komi −0.5**: whoever's pawn arrives first wins, exactly as in the rules above.
* Positive komi favours White. Examples: with komi +0.5 an equal race (B+1) goes to White; with komi −1.5 White
  needs to arrive at least 2 steps ahead (W+2).

The **result** is `t + komi`, written like `W+0.5` or `B+2.5` (SGF `RE`), or `0` for a draw.

### Fence handicap

Each player starts with a configurable number of walls, 0 to 10 (default 10 each in Duel; up to 15 in Race). Everything else is unchanged:
a player whose walls are used up must move the pawn.

## Canonical State

At any point in the game, the complete board state is determined by:

* the game mode;
* board dimensions;
* every player's pawn position;
* every existing wall and its orientation and position;
* every player's remaining wall count;
* in KataQuoridor, the number of plies played (for the 300-ply draw) and the komi;
* in KataQuoridor with the repetition draw on, how often each position since the last wall placement has occurred;
* which players are alive, in four-player mode;
* the goal definition;
* whose turn it is;
* whether a winner has already been determined.

The game contains no hidden information.

All players can observe the complete current game state.

## Legal Action

A legal action is exactly one of:

### Pawn action

Move the current player's pawn to one of the destinations allowed by the movement and jumping rules.

### Wall action

Place one wall from the set of valid wall positions and orientations, provided that:

1. the player has at least one wall remaining;
2. the wall position lies within the board's wall-placement range;
3. the wall does not conflict with an existing wall;
4. the placement leaves every currently active player with at least one route to their goal.

No other action is legal.

The game is therefore fully determined by the current state and the selected legal action.
