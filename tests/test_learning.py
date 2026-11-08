import numpy as np
import pytest
import torch

from dda.agent import TrainConfig, dqn_policy, evaluate, random_policy, train_dqn
from dda.difficulty import Director, features, scores, split_by_player
from dda.env import LevelConfig, generate_level
from dda.player import Session, SimulatedPlayer


def test_dqn_learns_a_small_level_better_than_random():
    torch.set_num_threads(1)
    config = LevelConfig(4, 0.1, gems=1)
    net, rewards = train_dqn([config], TrainConfig(episodes=250, warmup=200, target_sync=200, eps_fraction=0.5), seed=0)
    assert np.mean(rewards[-50:]) > np.mean(rewards[:50])
    learned = evaluate(dqn_policy(net), config, 100, seed=1)["win_rate"]
    random = evaluate(random_policy, config, 100, seed=1)["win_rate"]
    assert learned > random + 0.3


def test_perfect_player_always_wins_and_skill_matters():
    rng = np.random.default_rng(0)
    levels = [generate_level(LevelConfig(6, 0.3), rng) for _ in range(60)]
    expert, novice = SimulatedPlayer(0, 1.0), SimulatedPlayer(1, 0.4)
    assert all(expert.play(lvl, rng).won for lvl in levels)
    assert np.mean([novice.play(lvl, rng).won for lvl in levels]) < 0.7


def test_features_only_use_the_past():
    config = LevelConfig(6, 0.2)
    assert features([], config)[:4] == [0.5, 0.25, 0.5, 0.0]
    history = [Session(1, 6, 0.2, "win", 20, 0, 10)] * 3
    win_rate, _, efficiency, _, _, _ = features(history, config)
    assert win_rate == 0.8 and efficiency == pytest.approx((1.5 + 0.5) / 4)


def test_split_keeps_players_apart():
    owners = np.repeat(np.arange(20), 5)
    test = split_by_player(owners, 0.25, np.random.default_rng(0))
    assert not set(owners[test]) & set(owners[~test])
    assert len(set(owners[test])) == 5


def test_director_gives_weak_players_easier_levels():
    # Toy model: win chance falls with traps and rises with past win rate.
    model = lambda x: np.clip(x[:, 0] * 1.2 - x[:, 5] * 1.5 - x[:, 4] * 0.3 + 0.2, 0, 1)
    director = Director(model, target=0.7)
    weak = [Session(1, 6, 0.2, "trap", 5, 0, 10)] * 6
    strong = [Session(2, 6, 0.2, "win", 10, 0, 10)] * 6
    easy, hard = director.next_level(weak), director.next_level(strong)
    difficulty = lambda c: c.size * 0.3 / 4 + c.trap_ratio * 1.5  # same weights as the toy model
    assert difficulty(easy) < difficulty(hard)


def test_scores_reward_better_predictions():
    y = np.array([0, 0, 1, 1])
    assert scores(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert scores(y, np.array([0.1, 0.2, 0.8, 0.9]))["log_loss"] < scores(y, np.full(4, 0.5))["log_loss"]
