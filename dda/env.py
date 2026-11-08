"""Gem Run: a small grid game used as the environment for RL and difficulty tuning.

The player starts in the top-left corner and has to reach the exit in the bottom-right
corner. Gems give points, traps end the run. Every generated level is solvable.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
VIEW = 2  # the agent sees a (2*VIEW+1) x (2*VIEW+1) window around itself
CHANNELS = 5  # trap, gem, wall, exit, visited
OBS_SIZE = CHANNELS * (2 * VIEW + 1) ** 2 + 2

Cell = tuple[int, int]


@dataclass(frozen=True)
class LevelConfig:
    size: int = 6
    trap_ratio: float = 0.15
    gems: int = 3

    @property
    def max_steps(self) -> int:
        return self.size * 4


@dataclass(frozen=True)
class Level:
    config: LevelConfig
    traps: frozenset[Cell]
    gems: frozenset[Cell]

    start: Cell = (0, 0)

    @property
    def exit(self) -> Cell:
        return (self.config.size - 1, self.config.size - 1)


def neighbours(cell: Cell, size: int):
    for dr, dc in ACTIONS:
        r, c = cell[0] + dr, cell[1] + dc
        if 0 <= r < size and 0 <= c < size:
            yield (r, c)


def safe_path(size: int, traps: frozenset[Cell] | set[Cell], start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest path that avoids traps (breadth-first search), or None if there is none."""
    previous: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = previous[cell]
            return path[::-1]
        for nxt in neighbours(cell, size):
            if nxt not in previous and nxt not in traps:
                previous[nxt] = cell
                queue.append(nxt)
    return None


def generate_level(config: LevelConfig, rng: np.random.Generator) -> Level:
    size = config.size
    start, goal = (0, 0), (size - 1, size - 1)
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in (start, goal)]
    n_traps = round(config.trap_ratio * len(free))
    for _ in range(200):
        picked = rng.choice(len(free), size=n_traps, replace=False)
        traps = frozenset(free[i] for i in picked)
        path = safe_path(size, traps, start, goal)
        if path is None or len(path) - 1 > config.max_steps:
            continue  # no way through, or no way through in time
        reachable = [c for c in free if c not in traps and safe_path(size, traps, start, c)]
        n_gems = min(config.gems, len(reachable))
        gems = frozenset(reachable[i] for i in rng.choice(len(reachable), size=n_gems, replace=False))
        return Level(config, traps, gems)
    raise ValueError(f"Could not generate a solvable level for {config}")


class GemRun:
    def __init__(self, level: Level):
        self.level = level
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.level.start
        self.visited = {self.pos}
        self.gems_left = set(self.level.gems)
        self.steps = 0
        self.done = False
        self.outcome: str | None = None
        return self.observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is over, call reset()")
        size = self.level.config.size
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        reward = -0.05
        if 0 <= r < size and 0 <= c < size:
            self.pos = (r, c)
        else:
            reward -= 0.25  # bumped into the wall
        self.steps += 1
        self.visited.add(self.pos)

        if self.pos in self.level.traps:
            reward -= 5
            self.done, self.outcome = True, "trap"
        elif self.pos == self.level.exit:
            reward += 5
            self.done, self.outcome = True, "win"
        elif self.pos in self.gems_left:
            self.gems_left.remove(self.pos)
            reward += 1
        if not self.done and self.steps >= self.level.config.max_steps:
            self.done, self.outcome = True, "timeout"
        return self.observe(), reward, self.done, {"outcome": self.outcome}

    @property
    def gems_collected(self) -> int:
        return len(self.level.gems) - len(self.gems_left)

    def observe(self) -> np.ndarray:
        """Egocentric view: trap / gem / wall / exit / visited channels around the agent,
        plus the direction to the exit. A local view generalises across random levels,
        and the visited channel gives the agent enough memory to stop walking in circles."""
        size = self.level.config.size
        width = 2 * VIEW + 1
        grid = np.zeros((CHANNELS, width, width), dtype=np.float32)
        for i in range(width):
            for j in range(width):
                cell = (self.pos[0] + i - VIEW, self.pos[1] + j - VIEW)
                if not (0 <= cell[0] < size and 0 <= cell[1] < size):
                    grid[2, i, j] = 1
                elif cell in self.level.traps:
                    grid[0, i, j] = 1
                elif cell in self.gems_left:
                    grid[1, i, j] = 1
                if cell == self.level.exit:
                    grid[3, i, j] = 1
                if cell in self.visited:
                    grid[4, i, j] = 1
        exit_r, exit_c = self.level.exit
        direction = np.array([(exit_r - self.pos[0]) / size, (exit_c - self.pos[1]) / size], dtype=np.float32)
        return np.concatenate([grid.ravel(), direction])

    def render(self) -> str:
        size = self.level.config.size
        rows = []
        for r in range(size):
            row = ""
            for c in range(size):
                cell = (r, c)
                if cell == self.pos:
                    row += "@"
                elif cell == self.level.exit:
                    row += "E"
                elif cell in self.level.traps:
                    row += "x"
                elif cell in self.gems_left:
                    row += "*"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)
