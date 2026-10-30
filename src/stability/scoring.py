"""Aggregate BRENK hits into per-axis stability scores."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .config import AXES


class Scorer:
    """Turn matched BRENK descriptions into scores and a tier."""

    def __init__(self, config: dict):
        self.severities: Dict[str, List[Tuple[str, int]]] = config["severities"]
        agg = config["aggregation"]
        self.extra_hit_increment = float(agg["extra_hit_increment"])
        self.axis_cap = float(agg["axis_cap"])
        self.cross_axis_increment = float(agg["cross_axis_increment"])
        self.total_cap = float(agg["total_cap"])
        self.tiers: Sequence[Tuple[float, str]] = sorted(config["tiers"])

    def validate_against(self, descriptions: Iterable[str]) -> List[str]:
        """Return config keys absent from the catalog, e.g. typos in a custom config."""
        known = set(descriptions)
        return sorted(k for k in self.severities if k not in known)

    def unweighted(self, descriptions: Iterable[str]) -> List[str]:
        """Return catalog entries missing from the config; they score 0."""
        return sorted(d for d in descriptions if d not in self.severities)

    def score(self, hits: Sequence[str]) -> Dict[str, float]:
        """Score one molecule from its list of matched BRENK descriptions."""
        per_axis = {axis: [] for axis in AXES}
        for description in hits:
            for axis, severity in self.severities.get(description, []):
                if severity > 0:
                    per_axis[axis].append(float(severity))

        scores = {
            axis: self._aggregate(values, self.extra_hit_increment, self.axis_cap)
            for axis, values in per_axis.items()
        }
        active = [v for v in scores.values() if v > 0]
        scores["total"] = self._aggregate(
            active, self.cross_axis_increment, self.total_cap
        )
        return scores

    @staticmethod
    def _aggregate(values: List[float], increment: float, cap: float) -> float:
        """Return the worst value plus `increment` per extra value, capped."""
        if not values:
            return 0.0
        return min(max(values) + increment * (len(values) - 1), cap)

    def tier(self, total: float) -> str:
        """Return the label of the highest tier threshold reached by `total`."""
        label = self.tiers[0][1]
        for threshold, name in self.tiers:
            if total >= threshold:
                label = name
        return label
