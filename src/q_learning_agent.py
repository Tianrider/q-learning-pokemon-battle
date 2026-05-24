import numpy as np
from collections import defaultdict
from poke_env.player import Player
from poke_env.battle import AbstractBattle


TYPE_CHART: dict[str, dict[str, set[str]]] = {
    "NORMAL": {"super": set(), "not_very": {"ROCK", "STEEL"}, "immune": {"GHOST"}},
    "FIRE": {
        "super": {"GRASS", "ICE", "BUG", "STEEL"},
        "not_very": {"FIRE", "WATER", "ROCK", "DRAGON"},
        "immune": set(),
    },
    "WATER": {
        "super": {"FIRE", "GROUND", "ROCK"},
        "not_very": {"WATER", "GRASS", "DRAGON"},
        "immune": set(),
    },
    "ELECTRIC": {
        "super": {"WATER", "FLYING"},
        "not_very": {"ELECTRIC", "GRASS", "DRAGON"},
        "immune": {"GROUND"},
    },
    "GRASS": {
        "super": {"WATER", "GROUND", "ROCK"},
        "not_very": {"FIRE", "GRASS", "POISON", "FLYING", "BUG", "DRAGON", "STEEL"},
        "immune": set(),
    },
    "ICE": {
        "super": {"GRASS", "GROUND", "FLYING", "DRAGON"},
        "not_very": {"FIRE", "WATER", "ICE", "STEEL"},
        "immune": set(),
    },
    "FIGHTING": {
        "super": {"NORMAL", "ICE", "ROCK", "DARK", "STEEL"},
        "not_very": {"POISON", "FLYING", "PSYCHIC", "BUG", "FAIRY"},
        "immune": {"GHOST"},
    },
    "POISON": {
        "super": {"GRASS", "FAIRY"},
        "not_very": {"POISON", "GROUND", "ROCK", "GHOST"},
        "immune": {"STEEL"},
    },
    "GROUND": {
        "super": {"FIRE", "ELECTRIC", "POISON", "ROCK", "STEEL"},
        "not_very": {"GRASS", "BUG"},
        "immune": {"FLYING"},
    },
    "FLYING": {
        "super": {"GRASS", "FIGHTING", "BUG"},
        "not_very": {"ELECTRIC", "ROCK", "STEEL"},
        "immune": set(),
    },
    "PSYCHIC": {
        "super": {"FIGHTING", "POISON"},
        "not_very": {"PSYCHIC", "STEEL"},
        "immune": {"DARK"},
    },
    "BUG": {
        "super": {"GRASS", "PSYCHIC", "DARK"},
        "not_very": {
            "FIRE",
            "FIGHTING",
            "POISON",
            "FLYING",
            "GHOST",
            "STEEL",
            "FAIRY",
        },
        "immune": set(),
    },
    "ROCK": {
        "super": {"FIRE", "ICE", "FLYING", "BUG"},
        "not_very": {"FIGHTING", "GROUND", "STEEL"},
        "immune": set(),
    },
    "GHOST": {
        "super": {"PSYCHIC", "GHOST"},
        "not_very": {"DARK"},
        "immune": {"NORMAL"},
    },
    "DRAGON": {"super": {"DRAGON"}, "not_very": {"STEEL"}, "immune": {"FAIRY"}},
    "DARK": {
        "super": {"PSYCHIC", "GHOST"},
        "not_very": {"FIGHTING", "DARK", "FAIRY"},
        "immune": set(),
    },
    "STEEL": {
        "super": {"ICE", "ROCK", "FAIRY"},
        "not_very": {"FIRE", "WATER", "ELECTRIC", "STEEL"},
        "immune": set(),
    },
    "FAIRY": {
        "super": {"FIGHTING", "DRAGON", "DARK"},
        "not_very": {"FIRE", "POISON", "STEEL"},
        "immune": set(),
    },
}


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
        k_d: float = 40.0,
        k_t: float = 30.0,
        KO_bonus: float = 20.0,
        faint_penalty: float = 20.0,
        status_bonus: float = 5.0,
        time_penalty: float = 0.1,
        w_hp: float = 1.0,
        w_alive: float = 8.0,
        w_type: float = 2.0,
        training: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.gamma = gamma
        self.softmax_lambda = softmax_lambda
        self.training = training

        # Reward shaping hyperparameters
        self.k_d = k_d
        self.k_t = k_t
        self.KO_bonus = KO_bonus
        self.faint_penalty = faint_penalty
        self.status_bonus = status_bonus
        self.time_penalty = time_penalty
        self.w_hp = w_hp
        self.w_alive = w_alive
        self.w_type = w_type

        # Track previous state/action for Q-value updates
        self._prev_state: tuple | None = None
        self._prev_action: int | None = None
        self._prev_battle_tag: str | None = None
        self._reward_trackers: dict[str, dict[str, float]] = {}

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
        else:
            # Initialize tracker when this battle is first seen.
            self._reward_trackers.setdefault(
                battle.battle_tag, self._extract_battle_snapshot(battle)
            )

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
        self._reward_trackers[battle.battle_tag] = self._extract_battle_snapshot(battle)

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
        battle_tag = battle.battle_tag
        prev = self._reward_trackers.get(battle_tag)
        current = self._extract_battle_snapshot(battle)

        if prev is None:
            self._reward_trackers[battle_tag] = current
            return 0.0

        delta_opp_hp = max(prev["opp_active_hp"] - current["opp_active_hp"], 0.0)
        delta_my_hp = max(prev["my_active_hp"] - current["my_active_hp"], 0.0)

        ko_events = max(int(current["opp_fainted_count"] - prev["opp_fainted_count"]), 0)
        faint_events = max(int(current["my_fainted_count"] - prev["my_fainted_count"]), 0)
        status_events = max(
            int(current["opp_statused_count"] - prev["opp_statused_count"]), 0
        )

        r_damage = self.k_d * delta_opp_hp
        r_taken = -self.k_t * delta_my_hp
        r_ko = self.KO_bonus * ko_events
        r_faint = -self.faint_penalty * faint_events
        r_status = self.status_bonus * status_events
        r_time = -self.time_penalty
        r_potential = self.gamma * current["potential"] - prev["potential"]

        reward = r_damage + r_taken + r_ko + r_faint + r_status + r_time + r_potential
        reward = float(np.clip(reward, -50.0, 50.0))

        self._reward_trackers[battle_tag] = current
        return reward

    def _extract_battle_snapshot(self, battle: AbstractBattle) -> dict[str, float]:
        my_active_hp = self._hp_fraction(battle.active_pokemon)
        opp_active_hp = self._hp_fraction(battle.opponent_active_pokemon)

        my_team = self._team_members(battle.team)
        opp_team = self._team_members(battle.opponent_team)

        my_fainted_count = float(sum(1 for mon in my_team if getattr(mon, "fainted", False)))
        opp_fainted_count = float(
            sum(1 for mon in opp_team if getattr(mon, "fainted", False))
        )
        opp_statused_count = float(sum(1 for mon in opp_team if self._has_status(mon)))

        potential = self._state_potential(battle)
        return {
            "my_active_hp": my_active_hp,
            "opp_active_hp": opp_active_hp,
            "my_fainted_count": my_fainted_count,
            "opp_fainted_count": opp_fainted_count,
            "opp_statused_count": opp_statused_count,
            "potential": potential,
        }

    def _state_potential(self, battle: AbstractBattle) -> float:
        my_team = self._team_members(battle.team)
        opp_team = self._team_members(battle.opponent_team)

        sum_my_hp_frac = sum(self._hp_fraction(mon) for mon in my_team)
        sum_opp_hp_frac = sum(self._hp_fraction(mon) for mon in opp_team)
        alive_my = sum(1 for mon in my_team if not getattr(mon, "fainted", False))
        alive_opp = sum(1 for mon in opp_team if not getattr(mon, "fainted", False))

        type_term = self.type_advantage_score(
            battle.active_pokemon, battle.opponent_active_pokemon
        )

        return (
            self.w_hp * (sum_my_hp_frac - sum_opp_hp_frac)
            + self.w_alive * (alive_my - alive_opp)
            + self.w_type * type_term
        )

    def type_advantage_score(self, active_my, active_opp) -> float:
        if not active_my or not active_opp:
            return 0.0

        my_attack = self._best_attack_multiplier(active_my, active_opp)
        opp_attack = self._best_attack_multiplier(active_opp, active_my)

        if my_attack > opp_attack + 0.25:
            return 1.0
        if opp_attack > my_attack + 0.25:
            return -1.0
        return 0.0

    def _best_attack_multiplier(self, attacker, defender) -> float:
        atk_types = self._pokemon_types(attacker)
        if not atk_types:
            return 1.0

        def_types = self._pokemon_types(defender)
        if not def_types:
            return 1.0

        multipliers = [self._type_multiplier(atk_type, def_types) for atk_type in atk_types]
        return max(multipliers) if multipliers else 1.0

    def _type_multiplier(self, atk_type: str, def_types: list[str]) -> float:
        chart = TYPE_CHART.get(atk_type)
        if not chart:
            return 1.0

        multiplier = 1.0
        for def_type in def_types:
            if def_type in chart["immune"]:
                return 0.0
            if def_type in chart["super"]:
                multiplier *= 2.0
            elif def_type in chart["not_very"]:
                multiplier *= 0.5
        return multiplier

    def _pokemon_types(self, pokemon) -> list[str]:
        types: list[str] = []
        t1 = self._type_name(getattr(pokemon, "type_1", None))
        t2 = self._type_name(getattr(pokemon, "type_2", None))

        if t1:
            types.append(t1)
        if t2 and t2 != t1:
            types.append(t2)
        return types

    def _type_name(self, pokemon_type) -> str | None:
        if pokemon_type is None:
            return None
        if hasattr(pokemon_type, "name"):
            return str(pokemon_type.name).upper()
        return str(pokemon_type).upper()

    def _team_members(self, team) -> list:
        if not team:
            return []
        if hasattr(team, "values"):
            return list(team.values())
        return list(team)

    def _hp_fraction(self, pokemon) -> float:
        if pokemon is None or getattr(pokemon, "fainted", False):
            return 0.0
        hp = getattr(pokemon, "current_hp_fraction", 0.0)
        if hp is None:
            return 0.0
        return float(max(0.0, min(1.0, hp)))

    def _has_status(self, pokemon) -> bool:
        if pokemon is None:
            return False
        status = getattr(pokemon, "status", None)
        return status is not None

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
            self._reward_trackers.pop(battle.battle_tag, None)
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
        self._reward_trackers.pop(battle.battle_tag, None)

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
