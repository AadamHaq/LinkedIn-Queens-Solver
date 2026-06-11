"""
Board encoding for the imitation learning policy.

The network is given only raw facts about the position:
    - which region each cell belongs to (one-hot, labels randomly permutable)
    - where queens currently sit
    - which cells are actually on the board (when padded for batching)

No rule-derived features (occupied columns, used regions, adjacency) are
provided — the network has to discover those patterns from the expert's play.

Sizing: the network is fully convolutional, so the spatial size is not fixed.
Boards are only padded (to the largest size in a training batch) for batching;
at inference a board is encoded at its native size. The one baked-in limit is
REGION_CHANNELS — the number of one-hot region planes — which is given
headroom above today's largest board (11x11) so future bigger boards still
run through the same architecture.
"""

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

REGION_CHANNELS = 16  # headroom: largest board seen so far is 11 regions
NUM_CHANNELS = REGION_CHANNELS + 2  # region one-hots + queens + on-board mask


def random_region_permutation(n: int) -> List[int]:
    """
    Function: random_region_permutation creates a shuffled relabelling of regions

    Args:
        n: int number of regions on the board

    Description: Region ids are arbitrary labels, so training samples are
    augmented by shuffling them. This forces the network to learn board
    structure rather than memorising label identities.

    Returns: List[int] -> perm[old_region_id] = new_region_id
    """
    perm = list(range(n))
    random.shuffle(perm)
    return perm


def encode(
    board: List[List[int]],
    queens: Sequence[Tuple[int, int]],
    perm: Optional[List[int]] = None,
    pad_to: Optional[int] = None,
) -> np.ndarray:
    """
    Function: encode converts a board state into a network input tensor

    Args:
        board: List of Lists of Integers between [0, N-1]
        queens: Sequence of (row, col) tuples for queens already placed
        perm: Optional region relabelling from random_region_permutation
        pad_to: Optional spatial size to pad to (for batching); defaults to N

    Description: Builds a (NUM_CHANNELS, S, S) float32 array where S is
    pad_to or the board size. Channels 0..REGION_CHANNELS-1 one-hot the
    (optionally permuted) region of each cell, the next channel marks placed
    queens and the last channel marks cells that are on the board.

    Returns: np.ndarray -> shape (NUM_CHANNELS, S, S)
    """
    n = len(board)
    if n > REGION_CHANNELS:
        raise ValueError(
            f"Board has {n} regions but the encoding supports at most "
            f"{REGION_CHANNELS}; increase REGION_CHANNELS and retrain"
        )

    size = pad_to if pad_to is not None else n
    x = np.zeros((NUM_CHANNELS, size, size), dtype=np.float32)

    for r in range(n):
        for c in range(n):
            region = board[r][c]
            if perm is not None:
                region = perm[region]
            x[region, r, c] = 1.0
            x[REGION_CHANNELS + 1, r, c] = 1.0  # on-board mask

    for r, c in queens:
        x[REGION_CHANNELS, r, c] = 1.0

    return x


def legal_action_mask(
    n: int,
    queens: Sequence[Tuple[int, int]],
    pad_to: Optional[int] = None,
) -> np.ndarray:
    """
    Function: legal_action_mask marks which cells a queen may be placed on

    Args:
        n: int board size
        queens: Sequence of (row, col) tuples for queens already placed
        pad_to: Optional spatial size matching the encoded tensor

    Description: Only off-board and already-occupied cells are masked out.
    Rule-breaking cells (shared row/column/region, adjacency) deliberately
    stay available — avoiding them is what the network must learn.

    Returns: np.ndarray -> bool array of shape (S*S,), True = allowed
    """
    size = pad_to if pad_to is not None else n
    mask = np.zeros(size * size, dtype=bool)
    for r in range(n):
        for c in range(n):
            mask[r * size + c] = True
    for r, c in queens:
        mask[r * size + c] = False
    return mask


def action_to_cell(action: int, size: int) -> Tuple[int, int]:
    """
    Function: action_to_cell converts a flat action index to a board cell

    Args:
        action: int between [0, size*size - 1]
        size: int spatial size of the encoded tensor

    Returns: Tuple[int, int] -> (row, col)
    """
    return divmod(action, size)
