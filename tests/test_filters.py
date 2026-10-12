"""Query understanding: map, side and agent extraction, plus profile merging."""

from __future__ import annotations

from app.core.filters import build_context
from app.schemas.profile import ProfileIn, ProfileOut

GOLD_JETT = ProfileOut(
    user_id="u1",
    username="tester",
    current_rank="gold",
    target_rank="platinum",
    preferred_role="duelist",
    playstyle="aggressive",
    main_agents=["jett"],
    favorite_maps=["ascent"],
)


def test_extracts_map_and_side() -> None:
    ctx = build_context("Give me an attack strategy for Ascent", GOLD_JETT)
    assert ctx.maps == ["ascent"]
    assert ctx.side == "attack"


def test_extracts_defense_side() -> None:
    ctx = build_context("How do I hold B site on Split when defending?", None)
    assert ctx.maps == ["split"]
    assert ctx.side == "defense"
    assert ctx.site == "B"


def test_query_agent_beats_profile_agent() -> None:
    ctx = build_context("How do I play Killjoy on Ascent?", GOLD_JETT)
    assert ctx.agents == ["killjoy"]
    assert "sentinel" in ctx.roles


def test_profile_agent_used_when_query_names_none() -> None:
    ctx = build_context("Give me an attack strategy for Ascent", GOLD_JETT)
    assert ctx.agents == ["jett"]
    assert ctx.rank_band == "mid"
    assert ctx.playstyle == "aggressive"


def test_where_filters_map_and_side_but_keeps_untagged() -> None:
    ctx = build_context("attack strategy for Ascent", None)
    where = ctx.where()
    assert where == {
        "$and": [
            {"map": {"$in": ["ascent", ""]}},
            {"side": {"$in": ["attack", ""]}},
        ]
    }


def test_where_is_none_without_signals() -> None:
    assert build_context("how do I manage my economy", None).where() is None


def test_allows_rejects_wrong_side() -> None:
    ctx = build_context("attack strategy for Ascent", None)
    assert ctx.allows("ascent", "attack")
    assert ctx.allows("", "")  # untagged general strategy stays eligible
    assert not ctx.allows("ascent", "defense")
    assert not ctx.allows("bind", "attack")


def test_allows_rejects_wrong_site() -> None:
    ctx = build_context("How do I hold B site on Ascent?", None)
    assert ctx.site == "B"
    assert ctx.allows("ascent", "defense", "B")
    assert ctx.allows("ascent", "defense", "")  # whole-map notes stay eligible
    assert not ctx.allows("ascent", "defense", "A")


def test_article_a_is_not_site_a() -> None:
    """'take a site' is the English article, not A site."""
    assert build_context("How do I take a site on Ascent?", None).site == ""
    assert build_context("what is the best way to attack a site", None).site == ""
    # Explicit mentions still resolve, in either case for B and C.
    assert build_context("Give me an A site execute", None).site == "A"
    assert build_context("how do i hold b site", None).site == "B"
    assert build_context("C site on Haven", None).site == "C"


def test_where_includes_site_when_named() -> None:
    where = build_context("A site execute on Ascent", None).where()
    assert {"site": {"$in": ["A", ""]}} in where["$and"]


def test_rank_normalisation() -> None:
    p = ProfileIn(current_rank="Gold 2", target_rank="PLATINUM")
    assert p.current_rank == "gold"
    assert p.target_rank == "platinum"
