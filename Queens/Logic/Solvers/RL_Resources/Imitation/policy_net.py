"""
The imitation policy network: a small fully convolutional CNN.

Fully convolutional means no dense layers — the head is a 1x1 conv emitting
one logit per cell — so the same weights run on any board size. Only the
input channel count (board_encoding.NUM_CHANNELS) is fixed.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from Logic.Solvers.RL_Resources.Imitation.board_encoding import (
    NUM_CHANNELS,
    action_to_cell,
    encode,
    legal_action_mask,
)

HIDDEN_CHANNELS = 64
NUM_HIDDEN_LAYERS = 6  # receptive field ~13, covers an 11x11 board


class PolicyNet(nn.Module):
    """
    Function: PolicyNet predicts where the expert would place the next queen

    Description: A stack of 3x3 conv layers with batch norm, finished by a
    1x1 conv producing one logit per cell. forward returns logits flattened
    to (batch, S*S); illegal cells are masked by the caller.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = NUM_CHANNELS
        for _ in range(NUM_HIDDEN_LAYERS):
            layers += [
                nn.Conv2d(in_channels, HIDDEN_CHANNELS, 3, padding=1),
                nn.BatchNorm2d(HIDDEN_CHANNELS),
                nn.ReLU(),
            ]
            in_channels = HIDDEN_CHANNELS
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(HIDDEN_CHANNELS, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.body(x))  # (batch, 1, S, S)
        return logits.flatten(1)  # (batch, S*S)


@torch.no_grad()
def greedy_rollout(
    net: PolicyNet,
    board: List[List[int]],
    n: int,
    prefix: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Function: greedy_rollout plays a full board with the policy

    Args:
        net: trained PolicyNet
        board: List of Lists of Integers between [0, N-1]
        n: int board size
        prefix: Optional queens already on the board to continue from

    Description: Repeatedly encodes the position at native size and places a
    queen on the cell with the highest legal logit, until N queens are down.
    No rule checking happens here — the result may be wrong and must be
    verified by the caller.

    Returns: List[Tuple[int, int]] -> all N placements including the prefix,
    in placement order
    """
    net.eval()
    queens: List[Tuple[int, int]] = list(prefix) if prefix else []
    while len(queens) < n:
        x = torch.from_numpy(encode(board, queens)).unsqueeze(0)
        logits = net(x).squeeze(0)
        mask = torch.from_numpy(legal_action_mask(n, queens))
        logits[~mask] = -torch.inf
        action = int(torch.argmax(logits).item())
        queens.append(action_to_cell(action, n))
    return queens
