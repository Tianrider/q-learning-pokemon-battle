# Pokemon Battle AI - Q-Learning with Pokemon Showdown

Recreating the paper "Optimal Battle Strategy in Pokemon using Reinforcement Learning" (Kalose, Kaya, Kim) using the **real Pokemon Showdown simulator** instead of a custom deterministic engine.

## Key Design Choice

- **Our team is FIXED** (Alakazam, Snorlax, Tauros, Starmie, Exeggutor, Chansey) — removes randomness from our side
- **Opponent teams are RANDOM** — generated from a pool of 50 Gen 1 Pokemon each battle
- **Format: gen1ou** — allows custom teams, uses full Gen 1 mechanics

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Local Pokemon Showdown Server

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

The server must be running on `localhost:8000` before training.

### 3. Train the Agent

```bash
python src/train.py
```

Trains a softmax Q-Learning agent for 5000 battles against a random opponent, with progress logged every 10 battles. Then evaluates the trained agent over 500 battles.

### 4. Evaluate

```bash
python src/evaluate.py
```

Loads saved Q-tables and evaluates against 1000 random opponents.

### 5. Play Against the Bot

```bash
python src/play_human.py
```

Starts the trained bot on the local Showdown server. Open `http://localhost:8000` in your browser, pick a username, and challenge the bot (`bottianrider`) to a Gen 1 OU battle.

### 6. Visualize Training Results

```bash
python src/visualize_training.py --input results/q_table_softmax.pkl
```

You can also use a CSV training log (if available):

```bash
python src/visualize_training.py --input results/training_log.csv
```

Plots are saved under `results/plots/plot_[input_name]/`.

## Hyperparameter Tuning

You can run hyperparameter sweeps using `src/tune_hyperparams.py`. This will train and evaluate multiple configurations, logging results to `results/tuning_results.csv` and `results/tuning_log.txt`. The results can be visualized using `src/plot_tuning.py` to generate a summary plot of all configurations as `results/tuning_charts.png`.

## Architecture

```
src/
├── q_learning_agent.py   # Tabular Q-Learning with softmax exploration
├── teams.py              # Fixed team + random Gen 1 teambuilder
├── train.py              # Training vs random (logs every 10 battles)
├── evaluate.py           # Load & evaluate trained models
├── visualize_training.py # Generate visualize plot for training (pkl)
├── tune_hyperparams.py   # Run hyperparameter tuning sweeps
├── plot_tuning.py        # Generate summary plot for tuning results
└── play_human.py         # Play against the trained bot in browser
results/
├── *.pkl                 # Saved Q-tables
└── plots/                # Generated visualization outputs
```

## Algorithm (from paper)

**Q-Learning Update:**

```
Q(s, a) ← Q(s, a) + α(r + γ·max_a' Q(s', a') - Q(s, a))
```

**Hyperparameters:** α = 0.10, γ = 0.95, softmax λ = 1.0

**State Vector:**

| Feature            | Description                         |
| ------------------ | ----------------------------------- |
| player_hp_bucket   | HP of active Pokemon (0-9 bucket)   |
| opponent_hp_bucket | HP of opponent Pokemon (0-9 bucket) |
| player_type_1      | Primary type of our Pokemon         |
| player_type_2      | Secondary type of our Pokemon       |
| opponent_type_1    | Primary type of opponent            |
| opponent_type_2    | Secondary type of opponent          |

**Actions:** Move index (0-3), choosing from available moves

**Exploration:** Softmax — P(action) ∝ exp(λ · normalized_Q)

## Paper Results (reference)

| Method              | Win Rate vs Random |
| ------------------- | ------------------ |
| Softmax (5k games)  | 65%                |
| Softmax (20k games) | 70%                |

## Our Results

| Method             | Training | Win Rate vs Random  |
| ------------------ | -------- | ------------------- |
| Softmax (5k games) | 571.3s   | **93.4%** (467/500) |

- Q-table states discovered: 7311
- Training win rate: 86.4% (4320 wins / 678 losses)

Significantly outperforms the paper's 65% (5k games) and 70% (20k games) thanks to the fixed team removing state space noise from our side.

## Differences from Paper

| Paper                          | This Project                           |
| ------------------------------ | -------------------------------------- |
| Custom deterministic simulator | Real Pokemon Showdown (stochastic)     |
| Random teams for both sides    | Fixed team for us, random for opponent |
| No accuracy/crit RNG           | Full Gen 1 RNG (accuracy, crits, etc.) |
| 151 Pokemon, 165 moves         | Full Gen 1 via Showdown engine         |

## Future Improvements

1. **Increase opponent Pokemon team pool** — Expand the random opponent pool beyond the current 50 Gen 1 Pokemon to include all 151. This will expose the agent to a wider variety of type matchups and force it to generalize better, making the learned policy more robust against unseen teams.

2. **Add switch Pokemon action with type-effectiveness reward** — Currently the agent only chooses between its 4 available moves. Adding the ability to switch Pokemon as an action (expanding action space from 4 to up to 9) would allow the agent to learn strategic switches. A bonus reward should be given when switching to a Pokemon with a type advantage against the opponent's active Pokemon, encouraging the agent to learn defensive/offensive pivoting.
