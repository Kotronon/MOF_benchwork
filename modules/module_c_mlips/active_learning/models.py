"""Serializable models for the Module C active-learning state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.config import save_benchmark_data


@dataclass(frozen=True)
class ActiveLearningSettings:
    base_model: str = "mace_mp_0a_small"
    committee_size: int = 3
    initial_training_configurations: int = 30
    test_configurations: int = 20
    rounds: int = 4
    configurations_per_round: int = 25
    maximum_configurations: int = 150
    stable_rounds_required: int = 2
    energy_pass_mae_ev: float = 0.043
    force_pass_mae_ev_per_A: float = 0.10
    energy_warning_mae_ev: float = 0.10
    force_warning_mae_ev_per_A: float = 0.20
    candidate_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "framework": 0.20,
            "single_adsorbate": 0.40,
            "multiple_adsorbates": 0.40,
        }
    )
    cp2k: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveLearningSettings":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in data.items() if key in fields}
        return cls(**values)

    @property
    def bootstrap_count(self) -> int:
        return self.initial_training_configurations + self.test_configurations


@dataclass
class ActiveLearningState:
    campaign_id: str
    material_id: str
    status: str = "new"
    iteration: int = 0
    stable_rounds: int = 0
    candidate_count: int = 0
    labeled_count: int = 0
    training_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    pending_ids: list[str] = field(default_factory=list)
    baseline_ids: list[str] = field(default_factory=list)
    model_paths: list[str] = field(default_factory=list)
    selected_cutoff_Ry: int | None = None
    validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveLearningState":
        values = dict(data)
        values.pop("schema_version", None)
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "ActiveLearningState":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        save_benchmark_data(path, self.to_dict())
