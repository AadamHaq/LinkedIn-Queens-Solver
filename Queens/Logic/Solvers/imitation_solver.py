"""
Daily-run adapter for the imitation learning policy.

Drop-in replacement for naive_backtracking.backtracking: same signature,
same return format. Loads the trained weights once, plays the board with a
greedy rollout, then verifies the proposed solution against the rules. If
the weights are missing or the network gets the board wrong, it falls back
to backtracking so the daily run never fails.

Train/refresh the weights with RL_Resources/Imitation/train_imitation.py.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from Logic.Solvers.naive_backtracking import backtracking
from Logic.Solvers.RL_Resources.Imitation.expert import verify_solution

if TYPE_CHECKING:
    from Logic.Solvers.RL_Resources.Imitation.policy_net import PolicyNet

WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "RL_Resources" / "Weights" / "imitation_policy.pt"
)

_net: Optional["PolicyNet"] = None  # loaded once, reused across calls


def _load_net() -> Optional["PolicyNet"]:
    global _net
    if _net is not None:
        return _net
    if not WEIGHTS_PATH.exists():
        return None

    import torch
    from Logic.Solvers.RL_Resources.Imitation.policy_net import PolicyNet

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    net = PolicyNet()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    _net = net
    return _net


def imitation(board: List[List[int]], N: int = 0) -> List[Tuple[int, int]]:
    """
    Function: imitation solves the board with the trained policy network

    Args:
        board: List of Lists of Integers between [0, N-1]
        N: int for length of board. Default is 0

    Description: Greedy rollout of the imitation policy, verified against
    the game rules. Falls back to backtracking if no weights are present or
    the network's proposal is invalid.

    Returns: List[Tuple(int, int)] -> rows and columns of queen solutions
    """
    N = N or len(board)

    net = _load_net()
    if net is None:
        print(f"No imitation weights at {WEIGHTS_PATH} - using backtracking")
        return backtracking(board, N)

    from Logic.Solvers.RL_Resources.Imitation.policy_net import greedy_rollout

    attempt = greedy_rollout(net, board, N)
    if verify_solution(board, N, attempt):
        print("Imitation policy solved the board")
        return sorted(attempt)

    print("Imitation policy proposal was invalid - falling back to backtracking")
    return backtracking(board, N)
