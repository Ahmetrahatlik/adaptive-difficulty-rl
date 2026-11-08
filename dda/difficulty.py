"""Dynamic difficulty: predict a player's chance of winning a level from their history,
then pick the level whose predicted win chance is closest to a target."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

from dda.env import LevelConfig, generate_level
from dda.player import Session, SimulatedPlayer

FEATURES = ["history_win_rate", "history_trap_rate", "history_efficiency", "history_length", "size", "trap_ratio"]


def features(history: list[Session], config: LevelConfig) -> list[float]:
    """Only past sessions are used, never the one being predicted.
    The rates use a small prior so a new player starts in the middle.
    Efficiency (shortest route / steps taken, on won levels) separates careful
    players from lucky ones better than the win rate alone."""
    n = len(history)
    won = [s for s in history if s.won]
    traps = sum(s.outcome == "trap" for s in history)
    efficiency = (sum(min(1.0, s.optimal_steps / s.steps) for s in won) + 0.5) / (len(won) + 1)
    return [
        (len(won) + 1) / (n + 2),
        (traps + 0.5) / (n + 2),
        efficiency,
        min(n, 20) / 20,
        (config.size - 4) / 4,
        config.trap_ratio,
    ]


def random_config(rng: np.random.Generator) -> LevelConfig:
    return LevelConfig(size=int(rng.integers(4, 9)), trap_ratio=float(rng.uniform(0.0, 0.4)))


def collect(players: list[SimulatedPlayer], levels_each: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every player plays random levels. Returns features, outcomes and player ids."""
    rows, labels, owners = [], [], []
    for player in players:
        history: list[Session] = []
        for _ in range(levels_each):
            config = random_config(rng)
            session = player.play(generate_level(config, rng), rng)
            rows.append(features(history, config))
            labels.append(int(session.won))
            owners.append(player.player_id)
            history.append(session)
    return np.array(rows, dtype=np.float32), np.array(labels), np.array(owners)


class WinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self(torch.as_tensor(x, dtype=torch.float32))).numpy()


def train_win_model(x: np.ndarray, y: np.ndarray, epochs: int = 300, seed: int = 0) -> WinModel:
    torch.manual_seed(seed)
    model = WinModel()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    inputs, targets = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


def train_baseline(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(x, y)


def split_by_player(owners: np.ndarray, test_share: float, rng: np.random.Generator) -> np.ndarray:
    """Test players never appear in training, otherwise their history would leak."""
    ids = np.unique(owners)
    test_ids = rng.choice(ids, size=max(1, int(len(ids) * test_share)), replace=False)
    return np.isin(owners, test_ids)


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


@dataclass
class Director:
    """Chooses the next level for a player. predict(features) -> win probabilities."""

    predict: object
    target: float = 0.7
    sizes: tuple[int, ...] = (4, 5, 6, 7, 8)
    trap_ratios: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)

    def next_level(self, history: list[Session]) -> LevelConfig:
        configs = [LevelConfig(s, t) for s in self.sizes for t in self.trap_ratios]
        chances = self.predict(np.array([features(history, c) for c in configs], dtype=np.float32))
        return configs[int(np.argmin(np.abs(chances - self.target)))]


def simulate(players: list[SimulatedPlayer], levels_each: int, choose, rng: np.random.Generator) -> dict[int, float]:
    """choose(history) -> LevelConfig. Returns each player's win rate."""
    rates = {}
    for player in players:
        history: list[Session] = []
        for _ in range(levels_each):
            history.append(player.play(generate_level(choose(history), rng), rng))
        rates[player.player_id] = float(np.mean([s.won for s in history]))
    return rates
