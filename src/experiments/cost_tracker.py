from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    records: list[dict] = field(default_factory=list)

    def add(self, agent_name: str, example_id: str, cost_usd: float, usage: dict) -> None:
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        self.total_cost_usd += float(cost_usd or 0.0)
        self.total_tokens += total_tokens
        self.records.append(
            {
                "agent_name": agent_name,
                "example_id": example_id,
                "cost_usd": float(cost_usd or 0.0),
                "total_tokens": total_tokens,
            }
        )

    def summary(self) -> dict:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "num_records": len(self.records),
            "records": self.records,
        }
