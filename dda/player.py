"""Simulated players with different skill and play styles, used to create player data."""

from dataclasses import dataclass

import numpy as np

from dda.env import ACTIONS, Cell, GemRun, Level, safe_path


@dataclass(frozen=True)
class Session:
    player_id: int
    size: int
    trap_ratio: float
    outcome: str
    steps: int
    gems: int
    optimal_steps: int  # length of the shortest safe route to the exit

    @property
    def won(self) -> bool:
        return self.outcome == "win"


@dataclass(frozen=True)
class SimulatedPlayer:
    """skill: chance of making a sensible move instead of a random one.
    greed: share of the gems the player tries to collect before heading to the exit."""

    player_id: int
    skill: float
    greed: float = 0.0

    def play(self, level: Level, rng: np.random.Generator) -> Session:
        game = GemRun(level)
        game.reset()
        wanted = round(self.greed * len(level.gems))
        done = False
        while not done:
            _, _, done, info = game.step(self._choose(game, wanted, rng))
        optimal = len(safe_path(level.config.size, level.traps, level.start, level.exit)) - 1
        return Session(self.player_id, level.config.size, level.config.trap_ratio, info["outcome"], game.steps, game.gems_collected, optimal)

    def _choose(self, game: GemRun, wanted: int, rng: np.random.Generator) -> int:
        if rng.random() >= self.skill:
            return int(rng.integers(len(ACTIONS)))
        size = game.level.config.size
        goal = game.level.exit
        if game.gems_collected < wanted and game.gems_left:
            paths = [p for g in game.gems_left if (p := safe_path(size, game.level.traps, game.pos, g))]
            if paths:
                goal = min(paths, key=len)[-1]
        path = safe_path(size, game.level.traps, game.pos, goal)
        return _direction(game.pos, path[1]) if path and len(path) > 1 else int(rng.integers(len(ACTIONS)))


def _direction(a: Cell, b: Cell) -> int:
    return ACTIONS.index((b[0] - a[0], b[1] - a[1]))


def random_players(n: int, rng: np.random.Generator, first_id: int = 0) -> list[SimulatedPlayer]:
    return [
        SimulatedPlayer(first_id + i, skill=float(rng.uniform(0.5, 0.95)), greed=float(rng.choice([0.0, 0.34, 0.67, 1.0])))
        for i in range(n)
    ]
