"""Turn a natural-language question plus a user profile into retrieval hints.

Two kinds of hint come out of this:

  hard filters  - passed to Chroma's `where` clause. Only used when the query
                  names something unambiguous (a map). Wrong hard filters are
                  worse than none, so the bar is high.
  soft signals  - used to re-rank candidates. The user's rank, main agents and
                  playstyle live here: they should tilt results, never exclude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.vocab import AGENTS, MAPS, RANK_BANDS, ROLE_OF_AGENT, SIDE_TERMS
from app.schemas.profile import ProfileOut


@dataclass
class QueryContext:
    query: str
    maps: list[str] = field(default_factory=list)
    side: str = ""
    site: str = ""
    agents: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    rank_band: str = ""
    playstyle: str = ""

    def where(self) -> dict | None:
        """Hard filter for the vector store.

        Only map and side are ever hard filters, because both are categorical
        rather than a matter of degree: a Bind chunk cannot help an Ascent
        question, and defensive setups are not an attacking plan. Everything
        else (agent, rank, playstyle) stays a soft boost.

        Untagged documents -- economy, trading, general strategy -- carry "" in
        these fields and remain eligible through the OR.
        """
        clauses: list[dict] = []
        if len(self.maps) == 1:
            clauses.append({"map": {"$in": [self.maps[0], ""]}})
        if self.side:
            clauses.append({"side": {"$in": [self.side, ""]}})
        if self.site:
            clauses.append({"site": {"$in": [self.site, ""]}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def allows(self, chunk_map: str, chunk_side: str, chunk_site: str = "") -> bool:
        """Same rule as `where`, for candidates that did not come from Chroma."""
        if self.maps and chunk_map and chunk_map not in self.maps:
            return False
        if self.side and chunk_side and chunk_side != self.side:
            return False
        # Asking about B site and being handed the A site section is the exact
        # failure this prevents. Untagged chunks (site == "") stay eligible
        # because they apply to the whole map.
        if self.site and chunk_site and chunk_site != self.site:
            return False
        return True


def _find_terms(text: str, vocabulary: list[str]) -> list[str]:
    found = []
    for term in vocabulary:
        if re.search(rf"\b{re.escape(term)}\b", text):
            found.append(term)
    return found


def build_context(query: str, profile: ProfileOut | None) -> QueryContext:
    q = query.lower()
    ctx = QueryContext(query=query)

    ctx.maps = _find_terms(q, MAPS)
    ctx.agents = _find_terms(q, AGENTS)

    for side, terms in SIDE_TERMS.items():
        if any(re.search(rf"\b{re.escape(t)}\b", q) for t in terms):
            ctx.side = side
            break

    # Site detection runs on the ORIGINAL text, not the lowercased copy,
    # because "a" is also the English article: "how do I take a site" must not
    # be read as A site. B and C have no such ambiguity, so they match in any
    # case; A only counts when the writer capitalised it.
    m = re.search(r"\b([BbCc])\s*(?:site|main|long|short)\b", query) or re.search(
        r"\bA\s*(?:site|main|long|short)\b", query
    )
    if m:
        ctx.site = (m.group(1) if m.groups() else "A").upper()

    if profile:
        # Profile agents supplement, never override, agents named in the query.
        if not ctx.agents and profile.main_agents:
            ctx.agents = [a for a in profile.main_agents if a in AGENTS]
        ctx.playstyle = profile.playstyle or ""
        if profile.current_rank:
            ctx.rank_band = RANK_BANDS.get(profile.current_rank, "")

    ctx.roles = sorted({ROLE_OF_AGENT[a] for a in ctx.agents if a in ROLE_OF_AGENT})
    if profile and profile.preferred_role and profile.preferred_role not in ctx.roles:
        ctx.roles.append(profile.preferred_role)

    return ctx
