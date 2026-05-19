"""
Training script for the Q-Learning Pokemon battle agent.

Trains using a fixed team for our agent and random teams for opponents.
Uses softmax exploration with Q-Learning (best approach from the paper).
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.player import RandomPlayer
from q_learning_agent import QLearningAgent
from teams import FIXED_TEAM, FixedTeambuilder, RandomGen1Teambuilder


BATTLE_FORMAT = "gen1ou"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


async def train(
    n_training_battles: int = 5000,
    n_eval_battles: int = 500,
    log_interval: int = 10,
    save_path: str | None = None,
):
    """
    Train Q-Learning agent (softmax) against a random opponent.

    Our agent uses a FIXED team.
    The opponent uses RANDOM Gen 1 teams each battle.
    Logs progress every `log_interval` battles.
    """
    print("=" * 60)
    print("Training Q-Learning Agent (Softmax)")
    print(f"  Training battles: {n_training_battles}")
    print(f"  Eval battles:     {n_eval_battles}")
    print(f"  Log interval:     every {log_interval} battles")
    print(f"  Format:           {BATTLE_FORMAT}")
    print(f"  Our team:         FIXED (Alakazam/Snorlax/Tauros/Starmie/Exeggutor/Chansey)")
    print(f"  Opponent team:    RANDOM (from Gen 1 pool)")
    print("=" * 60)

    # Create our Q-Learning agent with FIXED team
    agent = QLearningAgent(
        alpha=0.10,
        gamma=0.95,
        softmax_lambda=1.0,
        training=True,
        battle_format=BATTLE_FORMAT,
        team=FixedTeambuilder(FIXED_TEAM),
        max_concurrent_battles=1,
    )

    # Create random opponent with random Gen 1 teams
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=RandomGen1Teambuilder(),
        max_concurrent_battles=1,
    )

    # --- TRAINING PHASE (with progress logging) ---
    print(f"\n[TRAINING] Starting...")
    start_time = time.time()
    battles_done = 0

    while battles_done < n_training_battles:
        batch = min(log_interval, n_training_battles - battles_done)
        await agent.battle_against(opponent, n_battles=batch)
        battles_done += batch

        stats = agent.get_stats()
        elapsed = time.time() - start_time
        win_rate = stats["training_win_rate"] * 100
        print(
            f"  [{battles_done:>5}/{n_training_battles}] "
            f"Win rate: {win_rate:.1f}% | "
            f"Q-states: {stats['q_table_size']} | "
            f"Time: {elapsed:.1f}s"
        )

    stats = agent.get_stats()
    elapsed = time.time() - start_time
    print(f"\n[TRAINING COMPLETE] in {elapsed:.1f}s")
    print(f"  Q-table states discovered: {stats['q_table_size']}")
    print(f"  Training win rate: {stats['training_win_rate']*100:.1f}%")
    print(f"  Wins: {stats['training_wins']}, Losses: {stats['training_losses']}")

    # --- EVALUATION PHASE ---
    print(f"\n[EVALUATION] Running {n_eval_battles} battles (greedy policy)...")
    agent.training = False
    agent._prev_state = None
    agent._prev_action = None

    eval_opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=RandomGen1Teambuilder(),
        max_concurrent_battles=1,
    )

    eval_wins_before = agent.n_won_battles
    await agent.battle_against(eval_opponent, n_battles=n_eval_battles)
    eval_wins = agent.n_won_battles - eval_wins_before

    win_rate = eval_wins / n_eval_battles * 100
    print(f"\n[RESULTS]")
    print(f"  Win rate vs Random: {win_rate:.1f}% ({eval_wins}/{n_eval_battles})")

    # --- SAVE Q-TABLE ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save_q_table(save_path)
        print(f"  Q-table saved to: {save_path}")

    return agent, win_rate


async def main():
    """Run training."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    await train(
        n_training_battles=5000,
        n_eval_battles=500,
        log_interval=10,
        save_path=os.path.join(RESULTS_DIR, "q_table_softmax.pkl"),
    )


if __name__ == "__main__":
    asyncio.run(main())
