"""
Builds the behavioural cloning dataset from the solved metadata boards.

A solution is a *set* of placements, so each board is replayed under many
random placement orderings; every prefix of an ordering becomes one training
state. Because LinkedIn Queens boards have a unique solution, the target for
a state is the full set of remaining solution cells (a soft distribution),
which keeps training order-invariant.
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from Logic.Solvers.RL_Resources.Imitation.board_encoding import (
    encode,
    legal_action_mask,
    random_region_permutation,
)
from torch.utils.data import Dataset

Board = List[List[int]]
Cell = Tuple[int, int]
# (board_idx, queens placed so far, remaining solution cells)
Sample = Tuple[int, Tuple[Cell, ...], Tuple[Cell, ...]]
# (input tensor, target distribution, legal-action mask)
Item = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def load_metadata_boards(directory: str) -> List[Tuple[Board, List[Cell]]]:
    """
    Function: load_metadata_boards reads every solved board in the metadata dir

    Args:
        directory: path to Training_Data/Metadata

    Description: Each .txt file assigns queens_board and solved_board (the
    format create_training_data.py writes). Files missing either are skipped.

    Returns: List[Tuple[Board, List[Cell]]] -> (board, solution) pairs
    """
    boards = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), "r") as file:
            content = file.read()
        local_vars: Dict = {}
        exec(content, {}, local_vars)
        board = local_vars.get("queens_board")
        solution = local_vars.get("solved_board")
        if board and solution:
            boards.append((board, solution))
        else:
            print(f"Skipping {filename}: missing queens_board or solved_board")
    return boards


def build_samples(
    boards: List[Tuple[Board, List[Cell]]],
    orderings_per_board: int,
) -> List[Sample]:
    """
    Function: build_samples expands solved boards into training states

    Args:
        boards: (board, solution) pairs from load_metadata_boards
        orderings_per_board: how many random placement orderings to replay

    Description: For each ordering, every prefix (including the empty board)
    becomes a state whose target is the remaining solution cells. States are
    deduplicated per board, since different orderings share prefixes.

    Returns: List[Sample]
    """
    samples: List[Sample] = []
    seen = set()
    for board_idx, (_, solution) in enumerate(boards):
        order = list(solution)
        for _ in range(orderings_per_board):
            random.shuffle(order)
            for prefix_len in range(len(order)):
                placed = frozenset(order[:prefix_len])
                key = (board_idx, placed)
                if key in seen:
                    continue
                seen.add(key)
                remaining = tuple(sorted(set(order) - placed))
                samples.append((board_idx, tuple(sorted(placed)), remaining))
    return samples


class QueensImitationDataset(Dataset[Item]):
    """
    Function: QueensImitationDataset serves encoded states to the trainer

    Description: Each item is encoded on the fly with a fresh random region
    relabelling (augmentation), padded to the largest board in the dataset.
    Returns (input tensor, target distribution over cells, legal-action mask).
    """

    def __init__(
        self,
        boards: List[Tuple[Board, List[Cell]]],
        samples: List[Sample],
    ) -> None:
        self.boards = [board for board, _ in boards]
        self.samples = samples
        self.pad_size = max(len(board) for board in self.boards)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Item:
        board_idx, placed, remaining = self.samples[index]
        board = self.boards[board_idx]
        n = len(board)
        size = self.pad_size

        perm = random_region_permutation(n)
        x = torch.from_numpy(encode(board, placed, perm=perm, pad_to=size))

        target = np.zeros(size * size, dtype=np.float32)
        for r, c in remaining:
            target[r * size + c] = 1.0 / len(remaining)

        mask = legal_action_mask(n, placed, pad_to=size)
        return x, torch.from_numpy(target), torch.from_numpy(mask)
