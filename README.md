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

**Hyperparameters (baseline):** α = 0.10, γ = 0.95, softmax λ = 1.0  
**Best tuned:** α = 0.10, γ = 0.95, softmax λ = 2.0 → 96.8% eval win rate

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

### Baseline

| Metric | Value |
|---|---|
| Training win rate | 86.8% (4340 W / 660 L) |
| Eval win rate (greedy, 500 battles) | 94.8% |
| Q-table states discovered | 9,772 |
| Training time | ~706s |

### Hyperparameter Tuning Results

Grid search sweep over α, γ, and λ (one-at-a-time). All reward shaping parameters locked to defaults.

| ID | α | γ | λ | Train% | Eval% | States |
|---|---|---|---|---|---|---|
| lam_2.0 | 0.10 | 0.95 | **2.0** | 90.4% | **96.8%** | 9,459 |
| gamma_0.99 | 0.10 | **0.99** | 1.0 | 86.1% | **96.8%** | 9,742 |
| gamma_0.90 | 0.10 | **0.90** | 1.0 | 86.4% | 96.4% | 9,807 |
| lam_5.0 | 0.10 | 0.95 | **5.0** | 92.9% | 96.2% | 8,933 |
| lam_0.5 | 0.10 | 0.95 | **0.5** | 82.9% | 96.0% | 9,977 |
| alpha_0.05 | **0.05** | 0.95 | 1.0 | 87.5% | 96.0% | 9,738 |
| baseline | 0.10 | 0.95 | 1.0 | 86.8% | 94.8% | 9,772 |
| gamma_0.80 | 0.10 | **0.80** | 1.0 | 86.6% | 94.8% | 9,833 |
| alpha_0.20 | **0.20** | 0.95 | 1.0 | 86.1% | 93.2% | 9,808 |
| alpha_0.30 | **0.30** | 0.95 | 1.0 | 84.0% | 93.0% | 9,847 |

**Best config:** `lam_2.0` — α=0.10, γ=0.95, λ=2.0 → **96.8% eval win rate**

**Key findings:**
- λ is the most sensitive parameter — raising it from 1.0 to 2.0 reduces suboptimal exploration during training
- γ = 0.99 ties for best eval; higher discount fits Pokemon's 10–20 turn battle horizon
- α is least sensitive — higher values (0.20, 0.30) hurt performance by overreacting to Gen 1 RNG
- All configs exceed 93%, suggesting robustness comes largely from the fixed OU team

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
