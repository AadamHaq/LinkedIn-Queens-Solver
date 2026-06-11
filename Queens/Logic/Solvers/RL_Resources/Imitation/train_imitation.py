"""
Trains the imitation policy: a behavioural cloning pass over the solved
metadata boards, followed by DAgger rounds where the student's own rollouts
are corrected by the backtracking expert.

Run from anywhere:
    uv run python Queens/Logic/Solvers/RL_Resources/Imitation/train_imitation.py

The metric that matters is the full-board greedy solve rate on validation
boards the network has never seen — per-move accuracy is not reported.
Best weights go to RL_Resources/Weights/imitation_policy.pt.
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from Logic.Solvers.RL_Resources.Imitation.board_encoding import (
    REGION_CHANNELS,  # noqa: E402
)
from Logic.Solvers.RL_Resources.Imitation.dataset import (  # noqa: E402
    Board,
    Cell,
    QueensImitationDataset,
    Sample,
    build_samples,
    load_metadata_boards,
)
from Logic.Solvers.RL_Resources.Imitation.expert import (  # noqa: E402
    solve_from,
    verify_solution,
)
from Logic.Solvers.RL_Resources.Imitation.policy_net import (  # noqa: E402
    PolicyNet,
    greedy_rollout,
)
from torch.utils.data import DataLoader

# Make `Logic.*` imports work no matter where the script is launched from
QUEENS_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(QUEENS_DIR))

METADATA_DIR = (
    QUEENS_DIR / "Logic" / "Solvers" / "RL_Resources" / "Training_Data" / "Metadata"
)
WEIGHTS_DIR = QUEENS_DIR / "Logic" / "Solvers" / "RL_Resources" / "Weights"


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Function: masked_soft_cross_entropy is the training loss

    Args:
        logits: (batch, S*S) raw network outputs
        target: (batch, S*S) distribution over remaining solution cells
        mask: (batch, S*S) bool, True where a queen may be placed

    Description: Illegal cells are excluded from the softmax, then the loss
    is the cross entropy against the soft target distribution.

    Returns: torch.Tensor -> scalar loss
    """
    logits = logits.masked_fill(~mask, -1e9)
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def train_epochs(
    net: PolicyNet,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    label: str,
) -> None:
    net.train()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for x, target, mask in loader:
            optimiser.zero_grad()
            loss = masked_soft_cross_entropy(net(x), target, mask)
            loss.backward()
            optimiser.step()
            total += loss.item()
            count += 1
        print(f"  {label} epoch {epoch + 1}/{epochs} loss {total / count:.4f}")


def evaluate(
    net: PolicyNet,
    boards: List[Tuple[Board, List[Cell]]],
) -> Tuple[float, dict]:
    """
    Function: evaluate measures the full-board greedy solve rate

    Args:
        net: the policy under test
        boards: (board, solution) pairs the network has never trained on

    Returns: Tuple[float, dict] -> overall solve rate, {size: (solved, total)}
    """
    by_size: dict = defaultdict(lambda: [0, 0])
    solved = 0
    for board, _ in boards:
        n = len(board)
        attempt = greedy_rollout(net, board, n)
        ok = verify_solution(board, n, attempt)
        solved += ok
        by_size[n][0] += ok
        by_size[n][1] += 1
    return solved / len(boards), dict(by_size)


def collect_dagger_samples(
    net: PolicyNet,
    boards: List[Tuple[Board, List[Cell]]],
    seen: set,
) -> List[Sample]:
    """
    Function: collect_dagger_samples gathers expert corrections on student states

    Args:
        net: the current student policy
        boards: training (board, solution) pairs
        seen: dedup keys of every sample already in the dataset (mutated)

    Description: Rolls each board out with the student's own greedy actions.
    At every state visited, the expert is asked to complete the position; its
    remaining placements become the label. When the student has wrecked the
    position (expert returns []), the rollout stops — every later state would
    be unsolvable and unlabellable.

    Returns: List[Sample] -> newly seen states with expert labels
    """
    new_samples: List[Sample] = []
    for board_idx, (board, _) in enumerate(boards):
        n = len(board)
        queens: List[Cell] = []
        for _ in range(n):
            completion = solve_from(board, n, queens)
            if not completion:
                break
            key = (board_idx, frozenset(queens))
            if key not in seen:
                seen.add(key)
                remaining = tuple(sorted(set(completion) - set(queens)))
                new_samples.append((board_idx, tuple(sorted(queens)), remaining))
            queens.append(greedy_rollout_step(net, board, n, queens))
    return new_samples


def greedy_rollout_step(
    net: PolicyNet,
    board: Board,
    n: int,
    queens: List[Cell],
) -> Cell:
    """One greedy policy action from the given state."""
    continuation = greedy_rollout(net, board, n, prefix=queens)
    return continuation[len(queens)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the imitation policy")
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--dagger-epochs", type=int, default=5)
    parser.add_argument("--orderings", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    boards = load_metadata_boards(str(METADATA_DIR))
    print(f"Loaded {len(boards)} solved boards")

    indices = list(range(len(boards)))
    random.shuffle(indices)
    val_count = max(1, int(len(boards) * args.val_frac))
    val_boards = [boards[i] for i in indices[:val_count]]
    train_boards = [boards[i] for i in indices[val_count:]]
    print(f"Split: {len(train_boards)} train / {len(val_boards)} validation")

    samples = build_samples(train_boards, args.orderings)
    seen = {(idx, frozenset(placed)) for idx, placed, _ in samples}
    print(f"Behavioural cloning dataset: {len(samples)} states")

    net = PolicyNet()
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)

    report = [
        f"Training run {datetime.now():%Y-%m-%d %H:%M}",
        f"Args: {vars(args)}",
        f"Boards: {len(train_boards)} train / {len(val_boards)} val",
    ]

    def make_loader() -> DataLoader:
        ds = QueensImitationDataset(train_boards, samples)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("Behavioural cloning...")
    train_epochs(net, make_loader(), optimiser, args.bc_epochs, "BC")
    best_rate, by_size = evaluate(net, val_boards)
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    line = f"After BC: val solve rate {best_rate:.1%} {format_by_size(by_size)}"
    print(line)
    report.append(line)

    for round_idx in range(1, args.dagger_rounds + 1):
        print(f"DAgger round {round_idx}...")
        new_samples = collect_dagger_samples(net, train_boards, seen)
        samples.extend(new_samples)
        print(f"  collected {len(new_samples)} new states (dataset now {len(samples)})")
        train_epochs(
            net,
            make_loader(),
            optimiser,
            args.dagger_epochs,
            f"DAgger {round_idx}",
        )
        rate, by_size = evaluate(net, val_boards)
        line = (
            f"After DAgger {round_idx}: val solve rate {rate:.1%} "
            f"{format_by_size(by_size)} (+{len(new_samples)} states)"
        )
        print(line)
        report.append(line)
        if rate >= best_rate:
            best_rate = rate
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = WEIGHTS_DIR / "imitation_policy.pt"
    torch.save(
        {
            "state_dict": best_state,
            "val_solve_rate": best_rate,
            "region_channels": REGION_CHANNELS,
            "trained": datetime.now().isoformat(timespec="minutes"),
            "args": vars(args),
        },
        checkpoint_path,
    )
    report.append(f"Best val solve rate: {best_rate:.1%}")
    report.append(f"Weights saved to {checkpoint_path}")
    (WEIGHTS_DIR / "training_report.txt").write_text("\n".join(report) + "\n")
    print(f"Saved best weights (val solve rate {best_rate:.1%}) to {checkpoint_path}")


def format_by_size(by_size: dict) -> str:
    parts = [f"{n}x{n}: {s}/{t}" for n, (s, t) in sorted(by_size.items())]
    return "[" + ", ".join(parts) + "]"


if __name__ == "__main__":
    main()
