"""Resolve which AgentProfile a request belongs to and produce the matching
adapter pipeline.

Matching rules (first match wins, ``default: true`` profiles are tried last):

    header   + contains   -> case-insensitive substring match on header value
    path_prefix           -> request.url.path startswith
    query_param           -> matches if `?<param>=...` is present in the URL
    default: true         -> catch-all
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..adapters import AdapterPipeline
from ..adapters.registry import build_pipeline
from ..config.schema import AdapterConfig, AgentProfileConfig
from ..core.context import RequestContext


@dataclass
class CompiledAgent:
    name: str
    profile: AgentProfileConfig
    pipeline: AdapterPipeline


class AgentResolver:
    def __init__(
        self,
        profiles: List[AgentProfileConfig],
        adapter_cfg: Dict[str, AdapterConfig],
    ) -> None:
        self._explicit: List[CompiledAgent] = []
        self._default: Optional[CompiledAgent] = None
        for p in profiles:
            compiled = CompiledAgent(p.name, p, build_pipeline(p, adapter_cfg))
            if p.match.default:
                self._default = compiled
            else:
                self._explicit.append(compiled)

    def resolve(
        self,
        *,
        path: str,
        headers: Dict[str, str],
        query_params: Dict[str, str],
    ) -> Optional[CompiledAgent]:
        # case-insensitive header lookup
        h_lower = {k.lower(): v for k, v in headers.items()}

        for ag in self._explicit:
            m = ag.profile.match
            if m.path_prefix and path.startswith(m.path_prefix):
                return ag
            if m.header and m.contains:
                v = h_lower.get(m.header.lower(), "")
                if m.contains.lower() in v.lower():
                    return ag
            if m.query_param and m.query_param in query_params:
                return ag
        return self._default

    def update_context(self, ctx: RequestContext, agent: Optional[CompiledAgent]) -> None:
        if agent is not None:
            ctx.agent_name = agent.name
