import random
from poke_env.teambuilder import Teambuilder

FIXED_TEAM = """
Alakazam
Ability: No Ability
Level: 100
EVs: 252 SpA / 4 SpD / 252 Spe
- Psychic
- Recover
- Thunder Wave
- Seismic Toss

Snorlax
Ability: No Ability
Level: 100
EVs: 252 HP / 252 Atk / 4 SpD
- Body Slam
- Earthquake
- Rest
- Self-Destruct

Tauros
Ability: No Ability
Level: 100
EVs: 252 Atk / 4 SpD / 252 Spe
- Body Slam
- Hyper Beam
- Earthquake
- Blizzard

Starmie
Ability: No Ability
Level: 100
EVs: 252 SpA / 4 SpD / 252 Spe
- Surf
- Blizzard
- Thunder Wave
- Recover

Exeggutor
Ability: No Ability
Level: 100
EVs: 252 SpA / 252 HP / 4 SpD
- Psychic
- Sleep Powder
- Stun Spore
- Mega Drain

Chansey
Ability: No Ability
Level: 100
EVs: 252 HP / 252 SpD / 4 Def
- Soft-Boiled
- Thunder Wave
- Ice Beam
- Thunderbolt
"""

GEN1_POKEMON_POOL = [
    """Venusaur
Ability: No Ability
Level: 100
- Razor Leaf
- Sleep Powder
- Body Slam
- Swords Dance""",
    """Charizard
Ability: No Ability
Level: 100
- Fire Blast
- Earthquake
- Body Slam
- Swords Dance""",
    """Blastoise
Ability: No Ability
Level: 100
- Surf
- Blizzard
- Body Slam
- Rest""",
    """Pikachu
Ability: No Ability
Level: 100
- Thunderbolt
- Thunder Wave
- Surf
- Body Slam""",
    """Nidoking
Ability: No Ability
Level: 100
- Earthquake
- Blizzard
- Thunderbolt
- Body Slam""",
    """Nidoqueen
Ability: No Ability
Level: 100
- Earthquake
- Blizzard
- Thunderbolt
- Body Slam""",
    """Clefable
Ability: No Ability
Level: 100
- Blizzard
- Body Slam
- Thunder Wave
- Thunderbolt""",
    """Arcanine
Ability: No Ability
Level: 100
- Fire Blast
- Body Slam
- Hyper Beam
- Rest""",
    """Machamp
Ability: No Ability
Level: 100
- Submission
- Earthquake
- Hyper Beam
- Body Slam""",
    """Golem
Ability: No Ability
Level: 100
- Earthquake
- Rock Slide
- Body Slam
- Explosion""",
    """Slowbro
Ability: No Ability
Level: 100
- Surf
- Psychic
- Thunder Wave
- Rest""",
    """Gengar
Ability: No Ability
Level: 100
- Thunderbolt
- Hypnosis
- Night Shade
- Explosion""",
    """Hypno
Ability: No Ability
Level: 100
- Psychic
- Hypnosis
- Thunder Wave
- Rest""",
    """Rhydon
Ability: No Ability
Level: 100
- Earthquake
- Rock Slide
- Body Slam
- Substitute""",
    """Lapras
Ability: No Ability
Level: 100
- Blizzard
- Thunderbolt
- Body Slam
- Rest""",
    """Jynx
Ability: No Ability
Level: 100
- Blizzard
- Lovely Kiss
- Psychic
- Rest""",
    """Zapdos
Ability: No Ability
Level: 100
- Thunderbolt
- Drill Peck
- Thunder Wave
- Rest""",
    """Articuno
Ability: No Ability
Level: 100
- Blizzard
- Ice Beam
- Hyper Beam
- Rest""",
    """Moltres
Ability: No Ability
Level: 100
- Fire Blast
- Hyper Beam
- Agility
- Sky Attack""",
    """Dragonite
Ability: No Ability
Level: 100
- Blizzard
- Thunderbolt
- Body Slam
- Hyper Beam""",
    """Persian
Ability: No Ability
Level: 100
- Slash
- Hyper Beam
- Bubble Beam
- Thunder""",
    """Jolteon
Ability: No Ability
Level: 100
- Thunderbolt
- Thunder Wave
- Pin Missile
- Body Slam""",
    """Vaporeon
Ability: No Ability
Level: 100
- Surf
- Blizzard
- Rest
- Body Slam""",
    """Flareon
Ability: No Ability
Level: 100
- Fire Blast
- Body Slam
- Hyper Beam
- Quick Attack""",
    """Kangaskhan
Ability: No Ability
Level: 100
- Body Slam
- Hyper Beam
- Earthquake
- Rest""",
    """Dodrio
Ability: No Ability
Level: 100
- Drill Peck
- Body Slam
- Hyper Beam
- Agility""",
    """Victreebel
Ability: No Ability
Level: 100
- Razor Leaf
- Sleep Powder
- Stun Spore
- Body Slam""",
    """Poliwrath
Ability: No Ability
Level: 100
- Surf
- Hypnosis
- Body Slam
- Earthquake""",
    """Sandslash
Ability: No Ability
Level: 100
- Earthquake
- Rock Slide
- Body Slam
- Swords Dance""",
    """Dewgong
Ability: No Ability
Level: 100
- Surf
- Blizzard
- Rest
- Body Slam""",
]


class FixedTeambuilder(Teambuilder):
    def __init__(self, team_str: str):
        super().__init__()
        self._team = self.join_team(self.parse_showdown_team(team_str))

    def yield_team(self) -> str:
        return self._team


class RandomGen1Teambuilder(Teambuilder):
    def __init__(self, pool: list[str] | None = None):
        super().__init__()
        self.pool = pool or GEN1_POKEMON_POOL

    def yield_team(self) -> str:
        team_mons = random.sample(self.pool, 6)
        team_str = "\n\n".join(team_mons)
        return self.join_team(self.parse_showdown_team(team_str))
