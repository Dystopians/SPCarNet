from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .edit_types import MeshEdit


@dataclass(frozen=True)
class PortfolioItem:
    edit: MeshEdit
    expected_debt_reduction: float
    render_cost: float
    topology_cost: float
    free_space_risk: float
    uncertainty: float
    prior_only_flag: bool = False

    def score(self) -> float:
        denom = 1.0 + max(self.render_cost, 0.0) + max(self.topology_cost, 0.0)
        penalty = self.free_space_risk + 0.5 * self.uncertainty + (2.0 if self.prior_only_flag else 0.0)
        return self.expected_debt_reduction / denom - penalty

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edit"] = self.edit.to_dict()
        data["score"] = self.score()
        return data


def rank_portfolio(items: list[PortfolioItem]) -> list[PortfolioItem]:
    return sorted(items, key=lambda x: x.score(), reverse=True)
