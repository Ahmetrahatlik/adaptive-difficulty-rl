import numpy as np
import pytest

from dda.env import OBS_SIZE, GemRun, Level, LevelConfig, generate_level, safe_path

UP, DOWN, LEFT, RIGHT = range(4)


def level(traps=(), gems=(), size=4):
    return Level(LevelConfig(size, 0.0, len(gems)), frozenset(traps), frozenset(gems))


@pytest.mark.parametrize("size,ratio", [(5, 0.0), (6, 0.2), (7, 0.35), (8, 0.4)])
def test_generated_levels_are_solvable_and_seeded(size, ratio):
    config = LevelConfig(size, ratio)
    a = generate_level(config, np.random.default_rng(42))
    b = generate_level(config, np.random.default_rng(42))
    assert a == b
    assert safe_path(size, a.traps, a.start, a.exit) is not None
    assert a.start not in a.traps and a.exit not in a.traps
    assert not (a.gems & a.traps)
    assert len(a.traps) == round(ratio * (size * size - 2))


def test_walls_block_and_cost_extra():
    game = GemRun(level())
    _, reward, done, _ = game.step(UP)
    assert game.pos == (0, 0) and not done
    assert reward == pytest.approx(-0.3)


def test_trap_ends_the_run():
    game = GemRun(level(traps=[(0, 1)]))
    _, reward, done, info = game.step(RIGHT)
    assert done and info["outcome"] == "trap" and reward < -4


def test_gems_count_once_and_exit_wins():
    game = GemRun(level(gems=[(0, 1)], size=2))
    _, reward, _, _ = game.step(RIGHT)
    assert reward == pytest.approx(0.95) and game.gems_collected == 1
    game.step(LEFT)
    _, reward, _, _ = game.step(RIGHT)
    assert reward == pytest.approx(-0.05)
    _, reward, done, info = game.step(DOWN)
    assert done and info["outcome"] == "win" and reward == pytest.approx(4.95)
    with pytest.raises(RuntimeError):
        game.step(DOWN)


def test_running_out_of_steps_is_a_timeout():
    game = GemRun(level(size=3))
    for _ in range(12):
        _, _, done, info = game.step(UP)
    assert done and info["outcome"] == "timeout"


def test_observation_is_egocentric():
    obs = GemRun(level(traps=[(1, 1)])).observe()
    assert obs.shape == (OBS_SIZE,)
    grid = obs[:-2].reshape(5, 5, 5)
    assert grid[2, 0, 0] == 1 and grid[2, 2, 2] == 0  # outside the board is a wall, own cell is not
    assert grid[0, 3, 3] == 1  # the trap one step down-right
    assert grid[4, 2, 2] == 1 and grid[4].sum() == 1  # only the start cell is visited
    assert obs[-2:] == pytest.approx([0.75, 0.75])
