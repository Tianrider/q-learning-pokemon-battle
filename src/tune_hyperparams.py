import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.player import RandomPlayer
from q_learning_agent import QLearningAgent
from teams import FIXED_TEAM, FixedTeambuilder, RandomGen1Teambuilder

BATTLE_FORMAT = "gen1ou"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "tuning_results.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "tuning_log.txt")

N_TRAINING_BATTLES = 8000
N_EVAL_BATTLES = 500
LOG_INTERVAL = 100  

# Reward shaping defaults
REWARD_DEFAULTS = dict(
    k_d=40.0,
    k_t=30.0,
    KO_bonus=20.0,
    faint_penalty=20.0,
    status_bonus=5.0,
    time_penalty=0.1,
    w_hp=1.0,
    w_alive=8.0,
    w_type=2.0,
)


TUNING_CONFIGS = [
    # Baseline 
    {"id": "baseline",   "alpha": 0.10, "gamma": 0.95, "softmax_lambda": 1.0},

    # Lambda sweep (alpha=0.10, gamma=0.95) 
    {"id": "lam_0.5",    "alpha": 0.10, "gamma": 0.95, "softmax_lambda": 0.5},
    {"id": "lam_2.0",    "alpha": 0.10, "gamma": 0.95, "softmax_lambda": 2.0},
    {"id": "lam_5.0",    "alpha": 0.10, "gamma": 0.95, "softmax_lambda": 5.0},

    # Alpha sweep (gamma=0.95, lambda=1.0) 
    {"id": "alpha_0.05", "alpha": 0.05, "gamma": 0.95, "softmax_lambda": 1.0},
    {"id": "alpha_0.20", "alpha": 0.20, "gamma": 0.95, "softmax_lambda": 1.0},
    {"id": "alpha_0.30", "alpha": 0.30, "gamma": 0.95, "softmax_lambda": 1.0},

    # Gamma sweep (alpha=0.10, lambda=1.0) 
    {"id": "gamma_0.80", "alpha": 0.10, "gamma": 0.80, "softmax_lambda": 1.0},
    {"id": "gamma_0.90", "alpha": 0.10, "gamma": 0.90, "softmax_lambda": 1.0},
    {"id": "gamma_0.99", "alpha": 0.10, "gamma": 0.99, "softmax_lambda": 1.0},
]

def log(msg: str, log_file=None):
    print(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def write_csv_row(result: dict, write_header: bool):
    fieldnames = [
        "id", "alpha", "gamma", "softmax_lambda",
        "train_win_rate", "eval_win_rate", "q_states",
        "train_time_s", "eval_wins", "timestamp",
    ]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(result)


# Single Config Run

async def run_single_config(cfg: dict, log_file) -> dict:
    cfg_id = cfg["id"]
    alpha  = cfg["alpha"]
    gamma  = cfg["gamma"]
    lam    = cfg["softmax_lambda"]

    log(f"\n{'='*65}", log_file)
    log(f"[{cfg_id}]  alpha={alpha}  gamma={gamma}  lambda={lam}", log_file)
    log(f"  reward shaping: locked to defaults {REWARD_DEFAULTS}", log_file)
    log(f"{'='*65}", log_file)

    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        softmax_lambda=lam,
        training=True,
        battle_format=BATTLE_FORMAT,
        team=FixedTeambuilder(FIXED_TEAM),
        max_concurrent_battles=1,
        **REWARD_DEFAULTS,
    )

    # Training 
    train_opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=RandomGen1Teambuilder(),
        max_concurrent_battles=1,
    )

    log(f"  [TRAIN] Starting {N_TRAINING_BATTLES} battles...", log_file)
    start = time.time()
    battles_done = 0
    train_curve = []

    while battles_done < N_TRAINING_BATTLES:
        batch = min(LOG_INTERVAL, N_TRAINING_BATTLES - battles_done)
        await agent.battle_against(train_opponent, n_battles=batch)
        battles_done += batch

        stats = agent.get_stats()
        wr = stats["training_win_rate"] * 100
        elapsed = time.time() - start
        train_curve.append({"battle": battles_done, "win_rate": round(wr, 2)})
        log(
            f"    [{battles_done:>5}/{N_TRAINING_BATTLES}]  "
            f"win={wr:.1f}%  states={stats['q_table_size']}  t={elapsed:.0f}s",
            log_file,
        )

    stats = agent.get_stats()
    train_elapsed = time.time() - start
    train_wr = stats["training_win_rate"] * 100
    q_states = stats["q_table_size"]

    log(
        f"  [TRAIN DONE]  win={train_wr:.1f}%  "
        f"states={q_states}  time={train_elapsed:.1f}s",
        log_file,
    )

    curve_path = os.path.join(RESULTS_DIR, f"curve_{cfg_id}.json")
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(train_curve, f)

    # Evaluation
    agent.training = False
    agent._prev_state = None
    agent._prev_action = None

    eval_opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=RandomGen1Teambuilder(),
        max_concurrent_battles=1,
    )

    log(f"  [EVAL]  Running {N_EVAL_BATTLES} greedy battles...", log_file)
    eval_wins_before = agent.n_won_battles
    await agent.battle_against(eval_opponent, n_battles=N_EVAL_BATTLES)
    eval_wins = agent.n_won_battles - eval_wins_before
    eval_wr = eval_wins / N_EVAL_BATTLES * 100

    log(f"  [EVAL DONE]  win={eval_wr:.1f}%  ({eval_wins}/{N_EVAL_BATTLES})", log_file)

    qtable_path = os.path.join(RESULTS_DIR, f"q_table_tune_{cfg_id}.pkl")
    agent.save_q_table(qtable_path)

    return {
        "id":             cfg_id,
        "alpha":          alpha,
        "gamma":          gamma,
        "softmax_lambda": lam,
        "train_win_rate": round(train_wr, 2),
        "eval_win_rate":  round(eval_wr, 2),
        "q_states":       q_states,
        "train_time_s":   round(train_elapsed, 1),
        "eval_wins":      eval_wins,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }

# Main

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_exists = os.path.exists(CSV_PATH)
    completed_ids: set[str] = set()
    if csv_exists:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed_ids.add(row["id"])

    configs_to_run = [c for c in TUNING_CONFIGS if c["id"] not in completed_ids]
    skipped = len(TUNING_CONFIGS) - len(configs_to_run)

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log(f"\n{'='*65}", log_file)
        log(f"# Hyperparameter Tuning Run  --  {datetime.now()}", log_file)
        log(f"# Sweep: alpha / gamma / softmax_lambda", log_file)
        log(f"# Reward shaping: locked to defaults", log_file)
        log(
            f"# Configs total: {len(TUNING_CONFIGS)}  |  "
            f"To run: {len(configs_to_run)}  |  Skipped (done): {skipped}",
            log_file,
        )
        log(f"# Training battles per config : {N_TRAINING_BATTLES}", log_file)
        log(f"# Eval battles per config     : {N_EVAL_BATTLES}", log_file)
        log(f"{'='*65}", log_file)

        all_results = []
        for i, cfg in enumerate(configs_to_run, 1):
            log(f"\n>>> Config {i}/{len(configs_to_run)}: {cfg['id']}", log_file)
            result = await run_single_config(cfg, log_file)
            all_results.append(result)

            write_header = not csv_exists and i == 1
            write_csv_row(result, write_header=write_header)
            csv_exists = True

            log(f"\n  Saved -> {CSV_PATH}", log_file)

        # Summary 
        if all_results:
            log(f"\n\n{'='*65}", log_file)
            log("TUNING SUMMARY (this run)", log_file)
            log(f"{'='*65}", log_file)
            log(
                f"  {'ID':<14} {'alpha':>6} {'gamma':>6} {'lambda':>7}  "
                f"{'Train%':>8} {'Eval%':>8}  {'States':>7}",
                log_file,
            )
            log("-" * 65, log_file)
            for r in sorted(all_results, key=lambda x: -x["eval_win_rate"]):
                log(
                    f"  {r['id']:<14} {r['alpha']:>6} {r['gamma']:>6} "
                    f"{r['softmax_lambda']:>7}  "
                    f"{r['train_win_rate']:>7.1f}%  {r['eval_win_rate']:>7.1f}%  "
                    f"{r['q_states']:>7}",
                    log_file,
                )

            best = max(all_results, key=lambda x: x["eval_win_rate"])
            log(
                f"\n  Best: [{best['id']}]  "
                f"eval={best['eval_win_rate']}%  "
                f"(alpha={best['alpha']}, gamma={best['gamma']}, "
                f"lambda={best['softmax_lambda']})",
                log_file,
            )

    print(f"\nAll results saved to : {CSV_PATH}")
    print(f"Full log saved to    : {LOG_PATH}")
    print(f"\nRun  python src/plot_tuning.py  to generate charts.")


if __name__ == "__main__":
    asyncio.run(main())
