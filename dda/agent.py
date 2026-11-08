"""Double DQN agent (PyTorch) that learns to play Gem Run on randomly generated levels."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dda.env import ACTIONS, OBS_SIZE, GemRun, LevelConfig, generate_level


class QNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class TrainConfig:
    episodes: int = 4000
    gamma: float = 0.97
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 30_000
    warmup: int = 1_000
    target_sync: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.6  # share of episodes over which epsilon decays


def greedy_action(net: QNet, obs: np.ndarray) -> int:
    with torch.no_grad():
        return int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())


def train_dqn(levels: list[LevelConfig], cfg: TrainConfig = TrainConfig(), seed: int = 0, log_every: int = 0) -> tuple[QNet, list[float]]:
    """Train on a new random level every episode so the agent learns a general policy,
    not one memorised map. Returns the network and the reward of every episode."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    online, target = QNet(), QNet()
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer: deque = deque(maxlen=cfg.buffer_size)
    rewards: list[float] = []
    step_count = 0

    for episode in range(cfg.episodes):
        level_cfg = levels[rng.integers(len(levels))]
        game = GemRun(generate_level(level_cfg, rng))
        obs = game.reset()
        progress = min(1.0, episode / max(1, cfg.episodes * cfg.eps_fraction))
        epsilon = cfg.eps_start + progress * (cfg.eps_end - cfg.eps_start)
        total = 0.0
        done = False
        while not done:
            action = int(rng.integers(len(ACTIONS))) if rng.random() < epsilon else greedy_action(online, obs)
            next_obs, reward, done, info = game.step(action)
            # A timeout is not a real terminal state, so we keep bootstrapping from it.
            terminal = done and info["outcome"] != "timeout"
            buffer.append((obs, action, reward, next_obs, terminal))
            obs, total = next_obs, total + reward
            step_count += 1

            if len(buffer) >= cfg.warmup:
                _learn(online, target, optimiser, random.sample(buffer, cfg.batch_size), cfg.gamma)
            if step_count % cfg.target_sync == 0:
                target.load_state_dict(online.state_dict())
        rewards.append(total)
        if log_every and (episode + 1) % log_every == 0:
            print(f"episode {episode + 1}: mean reward (last {log_every}) {np.mean(rewards[-log_every:]):.2f}, epsilon {epsilon:.2f}")
    return online, rewards


def _learn(online: QNet, target: QNet, optimiser, batch, gamma: float) -> None:
    obs, actions, rewards, next_obs, terminal = zip(*batch)
    obs = torch.from_numpy(np.stack(obs))
    next_obs = torch.from_numpy(np.stack(next_obs))
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminal = torch.tensor(terminal, dtype=torch.float32)

    q = online(obs).gather(1, actions).squeeze(1)
    with torch.no_grad():
        # Double DQN: the online net picks the next action, the target net scores it.
        best = online(next_obs).argmax(dim=1, keepdim=True)
        next_q = target(next_obs).gather(1, best).squeeze(1)
        goal = rewards + gamma * next_q * (1 - terminal)
    loss = nn.functional.smooth_l1_loss(q, goal)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10)
    optimiser.step()


def evaluate(policy, config: LevelConfig, episodes: int, seed: int) -> dict[str, float]:
    """policy(game, obs, rng) -> action. Every policy is scored on the same levels."""
    rng = np.random.default_rng(seed)
    outcomes = {"win": 0, "trap": 0, "timeout": 0}
    gems = 0
    for _ in range(episodes):
        game = GemRun(generate_level(config, rng))
        obs, done = game.reset(), False
        while not done:
            obs, _, done, info = game.step(policy(game, obs, rng))
        outcomes[info["outcome"]] += 1
        gems += game.gems_collected
    return {
        "win_rate": outcomes["win"] / episodes,
        "trap_rate": outcomes["trap"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "avg_gems": gems / episodes,
    }


def random_policy(game, obs, rng) -> int:
    return int(rng.integers(len(ACTIONS)))


def dqn_policy(net: QNet):
    return lambda game, obs, rng: greedy_action(net, obs)
