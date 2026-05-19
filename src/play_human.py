import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env import AccountConfiguration, ServerConfiguration
from q_learning_agent import QLearningAgent
from teams import FIXED_TEAM, FixedTeambuilder


BATTLE_FORMAT = "gen1ou"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Local server configuration (no authentication needed with --no-security)
LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?"
)


async def main():
    # Pick Q-table to load
    q_table_path = os.path.join(RESULTS_DIR, "q_table_softmax.pkl")
    if not os.path.exists(q_table_path):
        q_table_path = os.path.join(RESULTS_DIR, "q_table_self_play.pkl")
    if not os.path.exists(q_table_path):
        print("ERROR: No trained Q-table found in results/")
        print("Run 'python src/train.py' first to train the agent.")
        return

    bot_name = "bottianrider"

    print("=" * 60)
    print("  HUMAN vs AI - Pokemon Gen 1 OU Battle")
    print("=" * 60)
    print(f"\n  Loading Q-table from: {q_table_path}")

    agent = QLearningAgent(
        training=False,
        battle_format=BATTLE_FORMAT,
        team=FixedTeambuilder(FIXED_TEAM),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(bot_name, None),
        server_configuration=LOCAL_SERVER,
    )
    agent.load_q_table(q_table_path)

    print(f"  Q-table loaded: {len(agent.q_table)} states")
    print(f"\n  Bot username: {bot_name}")
    print(f"  Format: {BATTLE_FORMAT}")
    print(f"\n  INSTRUCTIONS:")
    print(f"  1. Open http://localhost:8000 in your browser")
    print(f"  2. Click 'Choose Name' and pick any username")
    print(f"  3. Click 'Find a user' (magnifying glass) and search: {bot_name}")
    print(f"  4. Click 'Challenge' -> select 'Gen 1 OU' format")
    print(f"  5. Build/select a Gen 1 team and send the challenge")
    print(f"\n  Waiting for challenge...")
    print("=" * 60)

    # Accept challenges indefinitely
    await agent.accept_challenges(None, n_challenges=1)

    # Print result
    for battle_id, battle in agent.battles.items():
        if battle.won:
            print(f"\n  Result: Bot WON!")
        elif battle.lost:
            print(f"\n  Result: You WON! (Bot lost)")
        else:
            print(f"\n  Result: TIE")

    print("\n  Want to play again? Re-run this script.")


if __name__ == "__main__":
    asyncio.run(main())
