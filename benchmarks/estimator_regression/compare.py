"""Base/head comparator and Markdown report rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml

SCHEMA_VERSION = 1


class Tolerance(NamedTuple):
    """Regression threshold for one metric."""

    direction: str
    absolute: float
    relative: float


EXACT_MATCH = Tolerance("same", 0.0, 0.0)

# Metrics not listed here must match exactly.
TOLERANCES: dict[str, Tolerance] = {
    "accuracy": Tolerance("higher", 0.002, 0.005),
    "cross_entropy": Tolerance("lower", 0.005, 0.01),
    "exact_match": Tolerance("higher", 0.002, 0.005),
    "f1": Tolerance("higher", 0.002, 0.005),
    "log_loss": Tolerance("lower", 0.005, 0.01),
    "macro_accuracy": Tolerance("higher", 0.002, 0.005),
    "macro_f1": Tolerance("higher", 0.002, 0.005),
    "mae": Tolerance("lower", 0.005, 0.01),
    "micro_f1": Tolerance("higher", 0.002, 0.005),
    "multioutput_mae": Tolerance("lower", 0.005, 0.01),
    "multioutput_rmse": Tolerance("lower", 0.005, 0.01),
    "n_clusters": EXACT_MATCH,
    "n_predictions": EXACT_MATCH,
    "n_unique_predictions": EXACT_MATCH,
    "r2": Tolerance("higher", 0.005, 0.01),
    "rmse": Tolerance("lower", 0.005, 0.01),
    "rolling_roc_auc": Tolerance("higher", 0.002, 0.005),
    "score_separation": Tolerance("higher", 0.01, 0.02),
}


@dataclass(frozen=True)
class MetricComparison:
    """A single base/head metric comparison result.

    Attributes:
        scenario_id: Scenario the metric belongs to.
        metric_name: Metric name within the scenario.
        base: Base-branch value.
        head: Head-branch value (``None`` if missing on head).
        direction: Comparison direction used to interpret the delta.
        regression: Signed regression magnitude (positive is worse).
        tolerance: Effective tolerance applied.
        status: One of ``regressed``, ``improved``, ``unchanged``,
            ``missing_head``.
    """

    scenario_id: str
    metric_name: str
    base: float
    head: float | None
    direction: str
    regression: float
    tolerance: float
    status: str


@dataclass
class ComparisonReport:
    """Aggregate comparison result across all scenarios."""

    regressions: list[MetricComparison] = field(default_factory=list)
    improvements: list[MetricComparison] = field(default_factory=list)
    new_scenarios: list[str] = field(default_factory=list)
    removed_scenarios: list[str] = field(default_factory=list)
    errored: list[dict[str, Any]] = field(default_factory=list)
    scenarios_compared: int = 0
    metrics_compared: int = 0
    schema_mismatches: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Whether any blocking issue should fail CI."""

        return bool(
            self.regressions or self.removed_scenarios or self.errored or self.schema_mismatches
        )


def _load_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _signed_regression(direction: str, base: float, head: float) -> float:
    """Positive regression means the head got worse for ``direction``."""

    if direction == "higher":
        return base - head
    if direction == "lower":
        return head - base
    return abs(head - base)


def _classify_status(regression: float, tolerance: float, direction: str) -> str:
    if regression > tolerance:
        return "regressed"
    if regression < -tolerance and direction != "same":
        return "improved"
    return "unchanged"


def _compare_metric(
    scenario_id: str, metric_name: str, base: float, head: float
) -> MetricComparison:
    tolerance_rule = TOLERANCES.get(metric_name, EXACT_MATCH)
    tolerance = max(tolerance_rule.absolute, abs(base) * tolerance_rule.relative)
    regression = _signed_regression(tolerance_rule.direction, base, head)
    return MetricComparison(
        scenario_id=scenario_id,
        metric_name=metric_name,
        base=base,
        head=head,
        direction=tolerance_rule.direction,
        regression=regression,
        tolerance=tolerance,
        status=_classify_status(regression, tolerance, tolerance_rule.direction),
    )


def _schema_mismatches(scenario_id: str, base_result: dict, head_result: dict) -> list[str]:
    mismatches: list[str] = []
    for field_name in ("workload", "harness", "n_samples"):
        if base_result.get(field_name) == head_result.get(field_name):
            continue
        mismatches.append(
            f"{field_name} changed for {scenario_id}: "
            f"base={base_result.get(field_name)} head={head_result.get(field_name)}"
        )
    return mismatches


def compare(base_path: str | Path, head_path: str | Path) -> ComparisonReport:
    """Compare two YAML result files and return a structured report."""

    base_data = _load_results(base_path)
    head_data = _load_results(head_path)
    report = ComparisonReport()
    if base_data.get("schema_version") != head_data.get("schema_version"):
        report.schema_mismatches.append(
            f"schema_version mismatch: base={base_data.get('schema_version')} "
            f"head={head_data.get('schema_version')}"
        )
        return report
    base_results = base_data.get("results", {}) or {}
    head_results = head_data.get("results", {}) or {}

    for scenario_id, head_result in sorted(head_results.items()):
        if head_result.get("status") == "error":
            report.errored.append({"id": scenario_id, **head_result})
            continue
        if scenario_id not in base_results:
            report.new_scenarios.append(scenario_id)
            continue

        base_result = base_results[scenario_id]
        mismatches = _schema_mismatches(scenario_id, base_result, head_result)
        if mismatches:
            report.schema_mismatches.extend(mismatches)
            continue

        report.scenarios_compared += 1
        base_metrics = base_result.get("metrics", {}) or {}
        head_metrics = head_result.get("metrics", {}) or {}
        for name in sorted(set(base_metrics) & set(head_metrics)):
            report.metrics_compared += 1
            metric = _compare_metric(
                scenario_id,
                name,
                float(base_metrics[name]),
                float(head_metrics[name]),
            )
            if metric.status == "regressed":
                report.regressions.append(metric)
            if metric.status == "improved":
                report.improvements.append(metric)
        for name in sorted(set(base_metrics) - set(head_metrics)):
            report.regressions.append(
                MetricComparison(
                    scenario_id=scenario_id,
                    metric_name=name,
                    base=float(base_metrics[name]),
                    head=None,
                    direction=TOLERANCES.get(name, EXACT_MATCH).direction,
                    regression=float("inf"),
                    tolerance=0.0,
                    status="missing_head",
                )
            )

    for scenario_id in sorted(set(base_results) - set(head_results)):
        report.removed_scenarios.append(scenario_id)
    return report


def comparison_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """Serialize a comparison report to plain YAML-friendly data."""

    return {
        "failed": report.failed,
        "scenarios_compared": report.scenarios_compared,
        "metrics_compared": report.metrics_compared,
        "regressions": [asdict(r) for r in report.regressions],
        "improvements": [asdict(r) for r in report.improvements],
        "new_scenarios": report.new_scenarios,
        "removed_scenarios": report.removed_scenarios,
        "errored": report.errored,
        "schema_mismatches": report.schema_mismatches,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def render_markdown(report: ComparisonReport) -> str:
    """Render a compact Markdown summary suitable for a PR comment."""

    status = "failed" if report.failed else "passed"
    lines = [
        f"# Estimator regression: {status}",
        "",
        f"Compared {report.scenarios_compared} scenarios and {report.metrics_compared} metrics.",
        f"{len(report.regressions)} metric regressions beyond tolerance.",
        f"{len(report.errored)} scenarios errored.",
        f"{len(report.new_scenarios)} scenarios are new and not yet gated.",
        "",
    ]
    if report.schema_mismatches:
        lines.extend(["## Schema Mismatches", ""])
        lines.extend(f"- {mismatch}" for mismatch in report.schema_mismatches)
        lines.append("")
    if report.regressions:
        lines.extend(["## Blocking Regressions", ""])
        for result in report.regressions:
            lines.append(
                f"- `{result.scenario_id}` **{result.metric_name}** "
                f"base={_fmt(result.base)} head={_fmt(result.head)} "
                f"regression={result.regression:.6f} tolerance={result.tolerance:.6f} "
                f"({result.status})"
            )
        lines.append("")
    if report.errored:
        lines.extend(["## Errored Scenarios", ""])
        for error in report.errored:
            message = error.get("error", {}).get("message", "unknown error")
            lines.append(f"- `{error.get('id')}`: {message}")
        lines.append("")
    if report.new_scenarios:
        lines.extend(["## New Scenarios", ""])
        lines.extend(f"- `{scenario_id}`" for scenario_id in report.new_scenarios)
        lines.append("")
    if report.removed_scenarios:
        lines.extend(["## Removed Scenarios", ""])
        lines.extend(f"- `{scenario_id}`" for scenario_id in report.removed_scenarios)
        lines.append("")
    if report.improvements:
        lines.extend(["## Largest Improvements", ""])
        top = sorted(report.improvements, key=lambda result: result.regression)[:10]
        for result in top:
            lines.append(
                f"- `{result.scenario_id}` **{result.metric_name}** "
                f"base={_fmt(result.base)} head={_fmt(result.head)} "
                f"improvement={-result.regression:.6f}"
            )
        lines.append("")
    lines.extend(["## Artifacts", "", "- `metrics.base.yml`", "- `metrics.head.yml`", ""])
    return "\n".join(lines)
