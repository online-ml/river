"""Command-line entry point for the estimator regression suite."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

from benchmarks.estimator_regression import workloads
from benchmarks.estimator_regression.compare import (
    SCHEMA_VERSION,
    compare,
    comparison_to_dict,
    render_markdown,
)
from benchmarks.estimator_regression.discovery import discover
from benchmarks.estimator_regression.run import HARNESSES, run_scenario
from benchmarks.estimator_regression.scenarios import EXCLUSIONS, SCENARIOS, Scenario
from river import __version__ as river_version


def _scenario_to_result(
    scenario: Scenario, metrics: dict[str, float], diagnostics: dict[str, float]
) -> dict:
    return {
        "estimator": scenario.estimator,
        "harness": scenario.harness,
        "workload": scenario.workload,
        "n_samples": scenario.n_samples,
        "status": "passed",
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _error_result(scenario: Scenario, exc: BaseException) -> dict:
    return {
        "estimator": scenario.estimator,
        "harness": scenario.harness,
        "workload": scenario.workload,
        "n_samples": scenario.n_samples,
        "status": "error",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _filter_scenarios(scenarios: Sequence[Scenario], keyword: str | None) -> list[Scenario]:
    if not keyword:
        return list(scenarios)
    kw = keyword.lower()
    return [
        s
        for s in scenarios
        if kw in s.id.lower() or kw in s.harness.lower() or kw in s.workload.lower()
    ]


def cmd_discover(args: argparse.Namespace) -> int:
    """List concrete public estimators discovered from the River API."""

    estimators = discover()
    print(f"# Discovered {len(estimators)} concrete public estimators")
    module_counts = Counter(est.module for est in estimators)
    for module, count in sorted(module_counts.items()):
        print(f"{module}: {count}")
    if args.verbose:
        for est in estimators:
            print(f"- {est.id}  ({est.qualname})")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Verify every discovered estimator is covered or explicitly excluded."""

    covered = {s.estimator for s in SCENARIOS}
    excluded = set(EXCLUSIONS)
    discovered = {f"river.{e.module}.{e.class_name}" for e in discover()}
    missing = sorted(discovered - covered - excluded)
    stale = sorted((covered | excluded) - discovered)
    problems: list[str] = []
    scenario_id_counts = Counter(s.id for s in SCENARIOS)
    for scenario_id, count in sorted(scenario_id_counts.items()):
        if count > 1:
            problems.append(f"DUPLICATE_ID: {scenario_id} appears {count} times in SCENARIOS")
    for est in missing:
        problems.append(f"MISSING: {est} is discovered but not in SCENARIOS or EXCLUSIONS")
    for est in stale:
        problems.append(f"STALE: {est} is listed but can no longer be imported")
    for estimator, reason in EXCLUSIONS.items():
        if not reason:
            problems.append(f"NO_REASON: exclusion for {estimator} has no reason")
    for scenario in SCENARIOS:
        if scenario.harness not in HARNESSES:
            problems.append(f"UNKNOWN_HARNESS: {scenario.id} uses {scenario.harness}")
        if not hasattr(workloads, scenario.workload):
            problems.append(f"UNKNOWN_WORKLOAD: {scenario.id} uses {scenario.workload}")
    for line in problems:
        print(line)
    print(
        f"# Audit: {len(discovered)} discovered, {len(covered)} covered, {len(excluded)} excluded, {len(problems)} problems"
    )
    return 1 if problems else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run scenarios and write the YAML metrics artifact."""

    scenarios = _filter_scenarios(SCENARIOS, args.keyword)
    results: dict[str, dict] = {}
    n_passed = 0
    n_failed = 0
    metrics_total = 0
    for scenario in scenarios:
        try:
            metrics, diagnostics = run_scenario(scenario)
            result = _scenario_to_result(scenario, metrics, diagnostics)
            n_passed += 1
            metrics_total += len(metrics)
        except Exception as exc:  # noqa: BLE001 - record every scenario failure
            result = _error_result(scenario, exc)
            n_failed += 1
        print(f"[{result['status']}] {scenario.id}")
        results[scenario.id] = result
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "river_version": river_version,
            "git_sha": _git_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "n_samples": workloads.N_SAMPLES,
            "seed": workloads.SEED,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        },
        "summary": {
            "scenarios_total": len(results),
            "scenarios_passed": n_passed,
            "scenarios_failed": n_failed,
            "metrics_total": metrics_total,
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True)
    print(f"# Wrote {out}: {n_passed} passed, {n_failed} failed")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare base and head YAML artifacts and write a Markdown report."""

    report = compare(args.base, args.head)
    markdown = render_markdown(report)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(markdown)
    if args.details:
        with Path(args.details).open("w") as fh:
            yaml.safe_dump(comparison_to_dict(report), fh, sort_keys=True)
    print(markdown)
    return 1 if report.failed else 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "HEAD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks.estimator_regression.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_discover = sub.add_parser("discover", help="list discovered estimators")
    p_discover.add_argument("-v", "--verbose", action="store_true")
    p_discover.set_defaults(func=cmd_discover)
    p_audit = sub.add_parser("audit", help="verify inventory coverage")
    p_audit.set_defaults(func=cmd_audit)
    p_run = sub.add_parser("run", help="run scenarios and emit metrics YAML")
    p_run.add_argument("--output", required=True)
    p_run.add_argument("-k", "--keyword", default=None, help="filter by scenario, harness, or workload")
    p_run.set_defaults(func=cmd_run)
    p_compare = sub.add_parser("compare", help="compare base/head YAML files")
    p_compare.add_argument("--base", required=True)
    p_compare.add_argument("--head", required=True)
    p_compare.add_argument("--output", required=True)
    p_compare.add_argument("--details", default=None, help="optional YAML comparison output")
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
