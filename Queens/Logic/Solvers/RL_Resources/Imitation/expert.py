"""
The expert that DAgger queries: backtracking, extended to solve from a
partial state (queens already on the board), plus a full-solution verifier.

The original naive_backtracking.backtracking is untouched; its is_valid
check is reused here.
"""

from typing import List, Set, Tuple

from Logic.Solvers.naive_backtracking import is_valid


def solve_from(
    board: List[List[int]],
    n: int,
    placed: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Function: solve_from completes a board from a partial placement

    Args:
        board: List of Lists of Integers between [0, N-1]
        n: int board size
        placed: List of (row, col) tuples for queens already on the board

    Description: Seeds the backtracking search with the given queens and
    solves the remaining rows. Returns [] if the partial state itself breaks
    a rule or cannot be completed — which is how DAgger detects that the
    student policy has gone off the rails.

    Returns: List[Tuple[int, int]] -> full solution including the seeded
    queens, or [] if unsolvable
    """
    queens: Set[Tuple[int, int]] = set()
    used_cols: Set[int] = set()
    used_colours: Set[int] = set()
    used_rows: Set[int] = set()

    # Validate and absorb the seed placements one by one
    for row, col in placed:
        if row in used_rows:
            return []
        if not is_valid(row, col, board, queens, used_cols, used_colours):
            return []
        queens.add((row, col))
        used_cols.add(col)
        used_colours.add(board[row][col])
        used_rows.add(row)

    def _backtrack(row: int) -> List[Tuple[int, int]]:
        if row == n:
            return list(queens)
        if row in used_rows:
            return _backtrack(row + 1)

        for col in range(n):
            if is_valid(row, col, board, queens, used_cols, used_colours):
                queens.add((row, col))
                used_cols.add(col)
                used_colours.add(board[row][col])

                result = _backtrack(row + 1)
                if result:
                    return result

                queens.remove((row, col))
                used_cols.remove(col)
                used_colours.remove(board[row][col])

        return []

    return _backtrack(0)


def verify_solution(
    board: List[List[int]],
    n: int,
    queens: List[Tuple[int, int]],
) -> bool:
    """
    Function: verify_solution checks a full proposed solution against the rules

    Args:
        board: List of Lists of Integers between [0, N-1]
        n: int board size
        queens: List of (row, col) tuples, one proposed queen per region

    Description: Checks exactly one queen per row, column and colour region,
    and no two queens in adjacent (including diagonal) cells.

    Returns: Boolean -> True if the solution is valid
    """
    if len(queens) != n:
        return False

    rows = {r for r, _ in queens}
    cols = {c for _, c in queens}
    colours = {board[r][c] for r, c in queens}
    if len(rows) != n or len(cols) != n or len(colours) != n:
        return False

    queen_set = set(queens)
    for r, c in queens:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in queen_set:
                    return False

    return True
