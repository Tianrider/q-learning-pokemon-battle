import numpy as np
from collections import defaultdict
from poke_env.player import Player
from poke_env.battle import AbstractBattle


def encode_state(battle: AbstractBattle) -> tuple:
    # Player HP bucket (0-9)
    if battle.active_pokemon and not battle.active_pokemon.fainted:
        player_hp = battle.active_pokemon.current_hp_fraction
    else:
        player_hp = 0.0
    player_hp_bucket = min(int(player_hp * 10), 9)

    # Opponent HP bucket (0-9)
    if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction
    else:
        opp_hp = 0.0
    opp_hp_bucket = min(int(opp_hp * 10), 9)

    # Player types
    if battle.active_pokemon:
        p_type1 = battle.active_pokemon.type_1.name if battle.active_pokemon.type_1 else "NONE"
        p_type2 = battle.active_pokemon.type_2.name if battle.active_pokemon.type_2 else "NONE"
    else:
        p_type1 = "NONE"
        p_type2 = "NONE"

    # Opponent types
    if battle.opponent_active_pokemon:
        o_type1 = battle.opponent_active_pokemon.type_1.name if battle.opponent_active_pokemon.type_1 else "NONE"
        o_type2 = battle.opponent_active_pokemon.type_2.name if battle.opponent_active_pokemon.type_2 else "NONE"
    else:
        o_type1 = "NONE"
        o_type2 = "NONE"

    return (player_hp_bucket, opp_hp_bucket, p_type1, p_type2, o_type1, o_type2)


class QLearningAgent(Player):
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.95,
        softmax_lambda: float = 1.0,
        training: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.gamma = gamma
        self.softmax_lambda = softmax_lambda
        self.training = training

        # Track previous state/action for Q-value updates
        self._prev_state: tuple | None = None
        self._prev_action: int | None = None
        self._prev_battle_tag: str | None = None

        # Statistics
        self.training_wins = 0
        self.training_losses = 0

    def choose_move(self, battle: AbstractBattle):
        # Check if battle ended (handle terminal state from previous action)
        if battle.finished:
            self._handle_battle_end(battle)
            return self.choose_default_move()

        state = encode_state(battle)
        n_moves = len(battle.available_moves)
        n_switches = len(battle.available_switches)

        # If no moves available, must switch
        if n_moves == 0:
            if n_switches > 0:
                self._prev_state = None
                self._prev_action = None
                return self.create_order(battle.available_switches[0])
            return self.choose_default_move()

        # Update Q-values from previous step (same battle)
        if (
            self.training
            and self._prev_state is not None
            and self._prev_action is not None
            and self._prev_battle_tag == battle.battle_tag
        ):
            reward = self._compute_intermediate_reward(battle)
            self._update_q(self._prev_state, self._prev_action, reward, state, n_moves)

        # Choose action
        if self.training:
            action = self._select_action(state, n_moves)
        else:
            # Greedy in evaluation
            action = int(np.argmax(self.q_table[state][:n_moves]))

        # Clamp to valid range
        action = min(action, n_moves - 1)

        # Store for next update
        self._prev_state = state
        self._prev_action = action
        self._prev_battle_tag = battle.battle_tag

        return self.create_order(battle.available_moves[action])

    def _select_action(self, state: tuple, n_moves: int) -> int:
        q_vals = self.q_table[state][:n_moves].copy()
        # Normalize Q-values for numerical stability
        q_range = np.max(q_vals) - np.min(q_vals)
        if q_range > 0:
            normalized = (q_vals - np.min(q_vals)) / q_range
        else:
            normalized = np.zeros(n_moves)
        # Softmax with temperature lambda
        exp_q = np.exp(self.softmax_lambda * normalized)
        probs = exp_q / exp_q.sum()
        return int(np.random.choice(n_moves, p=probs))

    def _compute_intermediate_reward(self, battle: AbstractBattle) -> float:
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            my_hp = battle.active_pokemon.current_hp_fraction
        else:
            my_hp = 0.0

        if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
            opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        else:
            opp_hp = 0.0

        return (my_hp - opp_hp) * 5.0

    def _update_q(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        n_next_actions: int,
    ):
        old_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state][: max(n_next_actions, 1)])
        self.q_table[state][action] = old_q + self.alpha * (
            reward + self.gamma * max_next_q - old_q
        )

    def _handle_battle_end(self, battle: AbstractBattle):
        if not self.training:
            return

        if self._prev_state is not None and self._prev_action is not None:
            if battle.won:
                reward = 100.0
                self.training_wins += 1
            elif battle.lost:
                reward = -100.0
                self.training_losses += 1
            else:
                reward = 0.0  # tie

            # Terminal update (no next state)
            old_q = self.q_table[self._prev_state][self._prev_action]
            self.q_table[self._prev_state][self._prev_action] = old_q + self.alpha * (
                reward - old_q
            )

        self._prev_state = None
        self._prev_action = None
        self._prev_battle_tag = None

    def _battle_finished_callback(self, battle: AbstractBattle):
        self._handle_battle_end(battle)

    def get_stats(self) -> dict:
        total = self.training_wins + self.training_losses
        return {
            "q_table_size": len(self.q_table),
            "training_wins": self.training_wins,
            "training_losses": self.training_losses,
            "training_win_rate": self.training_wins / total if total > 0 else 0,
        }

    def save_q_table(self, filepath: str):
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filepath: str):
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(4), data)
