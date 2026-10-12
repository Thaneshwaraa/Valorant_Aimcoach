"""Valorant vocabulary.

Used for three things: query understanding (which map/agent is this about?),
the domain gate (is this even a Valorant question?), and BM25 tokenisation
sanity. Kept in one module so adding a new agent is a one-line change.
"""

from __future__ import annotations

MAPS = [
    "ascent", "bind", "haven", "split", "icebox", "breeze", "fracture",
    "pearl", "lotus", "sunset", "abyss", "corrode", "summit",
]

AGENTS_BY_ROLE: dict[str, list[str]] = {
    "duelist": ["jett", "raze", "reyna", "phoenix", "yoru", "neon", "iso", "waylay"],
    "controller": ["omen", "brimstone", "viper", "astra", "harbor", "clove"],
    "initiator": ["sova", "breach", "skye", "kayo", "fade", "gekko", "tejo"],
    "sentinel": ["killjoy", "cypher", "sage", "chamber", "deadlock", "vyse", "veto"],
}

AGENTS = sorted({a for agents in AGENTS_BY_ROLE.values() for a in agents})
ROLES = list(AGENTS_BY_ROLE)

ROLE_OF_AGENT: dict[str, str] = {
    agent: role for role, agents in AGENTS_BY_ROLE.items() for agent in agents
}

# Spellings players actually type, mapped to the canonical name above.
AGENT_ALIASES: dict[str, str] = {
    "kay/o": "kayo", "kay-o": "kayo", "kay o": "kayo",
    "kj": "killjoy", "brim": "brimstone", "phx": "phoenix",
}


def normalize_agent(name: str) -> str | None:
    """Canonical agent name, or None if it is not a known agent."""
    n = " ".join((name or "").strip().lower().split())
    n = AGENT_ALIASES.get(n, n)
    return n if n in ROLE_OF_AGENT else None


def normalize_map(name: str) -> str | None:
    """Canonical map name, or None if it is not a known map."""
    n = (name or "").strip().lower()
    return n if n in MAPS else None

RANKS = [
    "iron", "bronze", "silver", "gold", "platinum",
    "diamond", "ascendant", "immortal", "radiant",
]

# Coarse bands are what the knowledge base tags itself with. Advice that is
# right for Silver is actively wrong for Immortal, so this mapping is what
# makes rank-aware retrieval possible.
RANK_BANDS: dict[str, str] = {
    "iron": "low", "bronze": "low", "silver": "low",
    "gold": "mid", "platinum": "mid", "diamond": "mid",
    "ascendant": "high", "immortal": "high", "radiant": "high",
}

BAND_ALIASES: dict[str, list[str]] = {
    "low": ["low", "iron", "bronze", "silver"],
    "mid": ["mid", "gold", "platinum", "plat", "diamond"],
    "high": ["high", "ascendant", "immortal", "radiant"],
}

# Domain terms. Any of these appearing in a query is strong evidence the user
# is talking about Valorant rather than something else.
GAME_TERMS = [
    "valorant", "spike", "plant", "defuse", "retake", "execute", "eco", "econ",
    "economy", "bonus round", "force buy", "full buy", "save", "rotate",
    "rotation", "lineup", "lineups", "crosshair", "aim", "peek", "peeking",
    "entry", "entry frag", "trade", "trading", "utility", "util", "ult",
    "ultimate", "orb", "ability", "abilities", "attack", "attacking", "defense",
    "defence", "defending", "site", "post plant", "postplant", "clutch",
    "flank", "flanking", "smoke", "smokes", "flash", "flashes", "molly",
    "molotov", "recon", "wallbang", "operator", "vandal", "phantom", "sheriff",
    "odin", "judge", "marshal", "outlaw", "guardian", "ares", "spectre", "bulldog",
    "headshot", "hs", "kd", "acs", "adr", "rank", "ranked", "rr", "elo",
    "comp", "competitive", "team comp", "composition", "callout", "callouts",
    # Generic match language. Included because coaching questions are often
    # phrased entirely in these words ("why do we keep losing rounds?").
    "buy", "buying", "credits", "creds", "shields", "armor", "armour",
    "loadout", "weapon", "gun", "pistol", "rifle", "afford",
    "round", "rounds", "losing", "lose", "win", "winning", "match", "game",
    "teammate", "teammates", "enemy", "enemies", "push", "hold", "duel",
    "spray", "recoil", "movement", "strafe", "counter", "improve", "climb",
    "practice", "warmup", "vod", "review", "solo queue", "soloq", "queue",
    "heaven", "hell", "catwalk", "mid", "tree", "market", "showers", "garage",
    "riot", "episode", "act", "agent", "duelist", "controller", "initiator",
    "sentinel", "radianite", "deathmatch", "aim trainer", "range",
    # Questions about the player themselves ("check my profile, give me a
    # tip"). The coach only coaches Valorant, so these are always on-topic.
    "profile", "tip", "tips", "coach", "coaching", "strength", "strengths",
    "weakness", "weaknesses", "playstyle", "main", "mains", "hardstuck",
    "my rank", "my agent", "my agents", "my role",
    # Aim and mouse setup.
    "dpi", "edpi", "sens", "sensitivity", "mouse", "mousepad", "flick",
    "flicking", "flicks", "tracking", "overshoot", "undershoot", "aimlab",
    "aim lab", "kovaaks", "first bullet",
]

DOMAIN_TERMS = set(MAPS) | set(AGENTS) | set(ROLES) | set(RANKS) | set(GAME_TERMS)

# The domain gate matches single tokens, so a multi-word entry such as
# "force buy" can never match a token and would be silently dead. Split the
# vocabulary by shape and let the gate handle each with the right strategy.
SINGLE_TERMS = {t for t in DOMAIN_TERMS if " " not in t and "-" not in t}
MULTIWORD_TERMS = sorted(t for t in DOMAIN_TERMS if " " in t or "-" in t)

SIDE_TERMS = {
    "attack": [
        "attack", "attacking", "offense", "offence",
        "t side", "t-side", "execute", "entry",
    ],
    "defense": [
        "defense", "defence", "defending", "defend",
        "ct side", "ct-side", "retake", "setup", "hold",
    ],
}
