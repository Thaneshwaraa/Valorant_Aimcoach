from app.schemas.profile import ProfileIn


def test_main_agents_normalised_and_deduped():
    p = ProfileIn(main_agents=["Jett", " KAY/O ", "kj", "jett", "Omen"])
    assert p.main_agents == ["jett", "kayo", "killjoy", "omen"]


def test_unknown_agents_dropped():
    p = ProfileIn(main_agents=["jet", "sova", "", "notanagent"])
    assert p.main_agents == ["sova"]


def test_every_role_agent_accepted():
    from app.core.vocab import AGENTS

    assert ProfileIn(main_agents=AGENTS).main_agents == AGENTS


def test_favorite_maps_normalised():
    p = ProfileIn(favorite_maps=["Ascent", " SUMMIT ", "ascent", "dust2"])
    assert p.favorite_maps == ["ascent", "summit"]


def test_skills_cleaned_but_free_text_allowed():
    p = ProfileIn(strengths=["Aim", " game  sense ", "aim"], weaknesses=["my own thing"])
    assert p.strengths == ["aim", "game sense"]
    assert p.weaknesses == ["my own thing"]


def test_edpi_derived_from_mouse_settings():
    assert ProfileIn(dpi=800, sensitivity=0.35).edpi == 280
    assert ProfileIn(dpi=800).edpi == 0
    assert ProfileIn().edpi == 0
