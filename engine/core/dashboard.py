from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WidgetSpec:
    """Minimal dashboard widget contract for Stage 5 surfaces."""

    section: str
    title: str
    kind: str
    value: Any
    unit: str = ""
    status: str = "info"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardPayload:
    """Five-block dashboard payload aligned with the Stage 1-5 PRD."""

    quality: list[WidgetSpec] = field(default_factory=list)
    modeling: list[WidgetSpec] = field(default_factory=list)
    optimization: list[WidgetSpec] = field(default_factory=list)
    mpc: list[WidgetSpec] = field(default_factory=list)
    report: list[WidgetSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "quality": [asdict(item) for item in self.quality],
            "modeling": [asdict(item) for item in self.modeling],
            "optimization": [asdict(item) for item in self.optimization],
            "mpc": [asdict(item) for item in self.mpc],
            "report": [asdict(item) for item in self.report],
        }

