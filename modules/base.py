from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SelectionScore = Callable[[dict[str, Any]], int]


@dataclass(frozen=True)
class BenchmarkModule:
    """Shared interface for benchmark modules registered in the pipeline."""

    id: str
    name: str
    slug: str
    task_aliases: tuple[str, ...]
    default_metrics: tuple[str, ...]
    selection_score: SelectionScore = field(repr=False)

    def metadata(self) -> dict[str, str]:
        """Return the public JSON metadata used in run plans."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
        }

    def matches(self, config: dict[str, Any]) -> bool:
        """Return whether this module is a candidate for automatic selection."""
        return self.selection_score(config) > 0

    def assess(self, config: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
        """Assess whether this module can run the requested benchmark."""
        from pipeline.applicability import assess_applicability

        return assess_applicability(config, self.metadata(), resolved)

    def prepare(self, run_plan: dict[str, Any]) -> dict[str, Any]:
        """Build a side-effect-free preparation plan using the shared pipeline."""
        from pipeline.prepare import prepare_benchmark

        return prepare_benchmark(run_plan)

    def materialize(self, prepare_plan: dict[str, Any]) -> dict[str, Any]:
        """Create files for a prepared benchmark using the shared pipeline."""
        from pipeline.materialization import materialize_benchmark

        return materialize_benchmark(prepare_plan)

    def run_test(self, materialized_plan: dict[str, Any]) -> dict[str, Any]:
        """Run the module's short technical smoke test."""
        from pipeline.runners import run_benchmark

        return run_benchmark(materialized_plan)

    def run(self, materialized_plan: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run the module's production benchmark path."""
        from pipeline.runners import run_isotherm

        return run_isotherm(materialized_plan, **kwargs)

    def evaluate(self, run_directory: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Evaluate an existing run directory."""
        from pipeline.evaluate import evaluate_run_directory

        return evaluate_run_directory(run_directory, **kwargs)


class UnsupportedModule(BenchmarkModule):
    """Registered placeholder for planned modules that are not executable yet."""

    def prepare(self, run_plan: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"Module {self.id} is registered but not implemented.")

    def materialize(self, prepare_plan: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"Module {self.id} is registered but not implemented.")

    def run_test(self, materialized_plan: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"Module {self.id} is registered but not implemented.")

    def run(self, materialized_plan: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(f"Module {self.id} is registered but not implemented.")

    def evaluate(self, run_directory: str | Path, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(f"Module {self.id} is registered but not implemented.")
