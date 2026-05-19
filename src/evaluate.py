import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.player import RandomPlayer
from q_learning_agent import QLearningAgent
from teams import FIXED_TEAM, FixedTeambuilder, RandomGen1Teambuilder


BATTLE_FORMAT = "gen1ou"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


async def evaluate(q_table_path: str, n_battles: int = 1000):
    """Load trained agent and evaluate against random opponent."""
    print(f"Loading Q-table from: {q_table_path}")

    agent = QLearningAgent(
        training=False,
        battle_format=BATTLE_FORMAT,
        team=FixedTeambuilder(FIXED_TEAM),
        max_concurrent_battles=1,
    )
    agent.load_q_table(q_table_path)

    print(f"Q-table loaded with {len(agent.q_table)} states")
    print(f"\nEvaluating over {n_battles} battles vs Random (random Gen 1 teams)...")

    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=RandomGen1Teambuilder(),
        max_concurrent_battles=1,
    )

    await agent.battle_against(opponent, n_battles=n_battles)

    wins = agent.n_won_battles
    win_rate = wins / n_battles * 100
    print(f"\nResults:")
    print(f"  Wins: {wins}/{n_battles}")
    print(f"  Win rate: {win_rate:.1f}%")

    return win_rate


async def main():
    # Evaluate all saved Q-tables
    q_tables = [
        ("Self-Play (Softmax)", "q_table_self_play.pkl"),
        ("Epsilon-Greedy", "q_table_epsilon_greedy.pkl"),
        ("Softmax", "q_table_softmax.pkl"),
    ]

    results = {}
    for name, filename in q_tables:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            rate = await evaluate(path, n_battles=1000)
            results[name] = rate
        else:
            print(f"\nSkipping {name} - {path} not found (train first)")

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, rate in results.items():
            print(f"  {name:30s}: {rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
