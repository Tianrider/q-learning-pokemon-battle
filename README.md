# Pokemon Battle AI - Q-Learning with Pokemon Showdown

Recreating the paper "Optimal Battle Strategy in Pokemon using Reinforcement Learning" (Kalose, Kaya, Kim) using the **real Pokemon Showdown simulator** instead of a custom deterministic engine.

## Key Design Choice

- **Our team is FIXED** (Alakazam, Snorlax, Tauros, Starmie, Exeggutor, Chansey) — removes randomness from our side
- **Opponent teams are RANDOM** — generated from a pool of 30 Gen 1 Pokemon each battle
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
cd src
python train.py
```

This runs:

1. **Self-play training** (5000 battles) — softmax agent vs epsilon-greedy agent
2. **Exploration comparison** — epsilon-greedy vs softmax, both trained for 5000 battles

### 4. Evaluate

```bash
cd src
python evaluate.py
```

Loads saved Q-tables and evaluates each against 1000 random opponents.

## Architecture

```
src/
├── q_learning_agent.py   # Tabular Q-Learning with epsilon-greedy & softmax
├── teams.py              # Fixed team + random Gen 1 teambuilder
├── train.py              # Self-play & vs-random training
└── evaluate.py           # Load & evaluate trained models
results/
└── *.pkl                 # Saved Q-tables
```

## Algorithm (from paper)

**Q-Learning Update:**

```
Q(s, a) ← Q(s, a) + α(r + γ·max_a' Q(s', a') - Q(s, a))
```

**Hyperparameters:** α = 0.10, γ = 0.95, ε = 0.10

**State Vector:**
| Feature | Description |
|---------|-------------|
| player_hp_bucket | HP of active Pokemon (0-9 bucket) |
| opponent_hp_bucket | HP of opponent Pokemon (0-9 bucket) |
| player_type_1 | Primary type of our Pokemon |
| player_type_2 | Secondary type of our Pokemon |
| opponent_type_1 | Primary type of opponent |
| opponent_type_2 | Secondary type of opponent |

**Actions:** Move index (0-3), choosing from available moves

**Exploration:**

- Epsilon-greedy: random move 10% of the time
- Softmax: P(action) ∝ exp(λ · normalized_Q)

## Paper Results (reference)

| Method                    | Win Rate vs Random |
| ------------------------- | ------------------ |
| Epsilon-greedy (5k games) | 60%                |
| Softmax (5k games)        | 65%                |
| Softmax (20k games)       | 70%                |

## Differences from Paper

| Paper                          | This Project                           |
| ------------------------------ | -------------------------------------- |
| Custom deterministic simulator | Real Pokemon Showdown (stochastic)     |
| Random teams for both sides    | Fixed team for us, random for opponent |
| No accuracy/crit RNG           | Full Gen 1 RNG (accuracy, crits, etc.) |
| 151 Pokemon, 165 moves         | Full Gen 1 via Showdown engine         |
