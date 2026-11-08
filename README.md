# Adaptive Difficulty RL

[![CI](https://github.com/Ahmetrahatlik/adaptive-difficulty-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmetrahatlik/adaptive-difficulty-rl/actions/workflows/ci.yml)

A small game, a reinforcement learning agent that learns to play it, and a difficulty system that uses player data to keep each player at a target win rate. Built with PyTorch, NumPy and scikit-learn.

**Gem Run** is a grid game. The player starts in the top-left corner and has to reach the exit in the bottom-right corner. Gems give points and traps end the run. Every level is generated randomly and is guaranteed to be solvable. Difficulty comes from the board size and the share of trap cells.

## What is in the project

1. **Double DQN agent (PyTorch)**
   - It trains on a new random level every episode, so it learns a general way to play rather than memorising one map.
   - It sees only a 5x5 window around itself plus the direction of the exit, which helps it handle levels it has never seen.
   - Training uses a replay buffer, a target network, the Huber loss and gradient clipping.
   - Hitting the step limit is not treated as the end of the game, so the agent keeps learning from the value of later moves.
2. **Player data**
   - Simulated players have different skill levels (how often they make a sensible move) and play styles (how many gems they chase before heading to the exit).
   - Each play session records the outcome, the number of steps and the gems collected.
3. **Win prediction**
   - A PyTorch network predicts whether a player will win a level. It uses the player's past win and trap rates and the level's size and trap ratio.
   - It is compared with scikit-learn logistic regression and with a model that always predicts the average.
   - The train/test split is by player, so the test players' histories never leak into training.
4. **Dynamic difficulty**
   - Before each level, a director picks the trap ratio whose predicted win chance for this player is closest to a target (70%).
   - It is compared with a fixed difficulty on new players.

## Results

The full numbers, with an example game, are in **[results.md](results.md)**. They come from `python experiment.py` and are not edited by hand.

| What | Result |
|---|---|
| DQN agent, win rate on new levels | easy 96%, medium 86%, hard 56% (random play: 3%, 0%, 0%) |
| Traps hit by the DQN agent | 0% at every difficulty; its failures are all timeouts |
| Win prediction, ROC-AUC on unseen players | PyTorch 0.779, logistic regression 0.775, constant 0.500 |
| Dynamic difficulty, mean distance from the 70% target | fixed 0.24, adaptive **0.10** |
| Lower-skill players' win rate | fixed 29%, adaptive 56% |

What the numbers say:

- The agent learned to avoid traps completely. When it loses, it is because it wanders, often after collecting gems, until the step limit runs out. That is the price of seeing only a 5x5 window.
- The neural network is only slightly better than logistic regression. The relationship between the features and winning is mostly linear here, so the simpler model would be a fine choice in production.
- The director brings most players much closer to the target. Lower-skill players still stay under 70%, because even the easiest level (4x4, no traps) is not easy enough for some of them.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                         # 15 tests, under a minute
python experiment.py              # full run (~7 min on CPU), writes results.md and models/dqn.pt
python experiment.py --quick      # small version, runs in CI
```

## Project structure

```
dda/env.py          Gem Run game, level generator, safe-path search
dda/agent.py        Double DQN, training loop, evaluation
dda/player.py       simulated players and play sessions
dda/difficulty.py   features, win model, logistic regression baseline, director
experiment.py       trains everything and writes results.md
tests/              rules of the game, learning smoke test, leakage and director tests
```

## Limits and next steps

- The players are simulated. With real players, the same pipeline would read session logs instead of `SimulatedPlayer`.
- The director only changes the trap ratio. Board size, gem count or time limit could be tuned the same way.
- The DQN agent could serve as a level tester: if a trained agent rarely wins a generated level, the level is probably too hard.
- A PPO agent and a convolutional network over the whole board are natural comparisons.

---

## Türkçe özet

Bu projede küçük bir oyun, onu oynamayı kendi kendine öğrenen bir yapay zeka ajanı ve oyuncu verisinden zorluk ayarlayan bir sistem var.

- **Oyun:** Izgarada mücevher toplayıp tuzaklardan kaçarak çıkışa ulaşmaya çalışırsın. Her bölüm rastgele üretilir ve her zaman çözülebilir.
- **Ajan:** Pekiştirmeli öğrenmeyle (Double DQN, PyTorch) deneme yanılarak oynamayı öğrenir. Her denemede yeni bir harita görür, bu yüzden hiç görmediği bölümleri de oynayabilir.
- **Oyuncu verisi:** Farklı beceri ve oyun tarzında sanal oyuncular oynar, her oyunun sonucu kaydedilir.
- **Kazanma tahmini:** Bir sinir ağı, oyuncunun geçmişine bakarak bir bölümü kazanma ihtimalini tahmin eder. scikit-learn ile karşılaştırılır.
- **Dinamik zorluk:** Her bölümden önce, oyuncunun kazanma ihtimalini %70'e en yakın tutacak tuzak oranı seçilir. Zayıf oyuncu kolay bölüm, güçlü oyuncu zor bölüm alır.

MIT License.
