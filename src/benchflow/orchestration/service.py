from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..cluster import CommandError, create_manifest
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..kueue import (
    create_reservation_workload,
    delete_reservation_workload,
    link_reservation_to_execution,
    queue_name_from_labels,
    requested_gpus_from_labels,
    reservation_required_for_labels,
    wait_for_reservation,
)
from ..loaders import load_run_plan_data, load_run_plan_file
from ..setup import load_setup_state
from ..toolbox import setup_platform, teardown_platform
from ..ui import detail, step, success, warning
from .tekton import TektonOrchestrator

DEFAULT_EXECUTION_NAME = "benchflow-e2e"
DEFAULT_MATRIX_EXECUTION_NAME = "benchflow-matrix"

_TEKTON = TektonOrchestrator()


def load_run_plan_from_sources(
    *,
    run_plan_file: str | None = None,
    run_plan_json: str | None = None,
) -> ResolvedRunPlan:
    if run_plan_file:
        return load_run_plan_file(Path(run_plan_file).resolve())
    if run_plan_json:
        try:
            raw = json.loads(run_plan_json)
        except json.JSONDecodeError as exc:
            raise ValidationError("invalid JSON passed to --run-plan-json") from exc
        return load_run_plan_data(raw)
    raise ValidationError("provide --run-plan-file or --run-plan-json")


def require_platform(plan: ResolvedRunPlan, platform: str) -> None:
    if plan.deployment.platform != platform:
        raise ValidationError(
            f"unsupported deployment platform: {plan.deployment.platform}; only {platform} is implemented"
        )


def render_execution_manifest(
    plan: ResolvedRunPlan,
    *,
    execution_name: str = DEFAULT_EXECUTION_NAME,
    setup_mode: str = "auto",
    teardown: bool = True,
    skip_kueue_reservation: bool = False,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    return _TEKTON.render_run(
        plan,
        execution_name=execution_name,
        setup_mode=setup_mode,
        teardown=teardown,
        skip_kueue_reservation=skip_kueue_reservation,
        benchflow_image=benchflow_image,
    )


def render_matrix_execution_manifest(
    plans: list[ResolvedRunPlan],
    *,
    execution_name: str = DEFAULT_MATRIX_EXECUTION_NAME,
    child_execution_name: str = DEFAULT_EXECUTION_NAME,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")
    return _TEKTON.render_matrix(
        plans,
        execution_name=execution_name,
        child_execution_name=child_execution_name,
        benchflow_image=benchflow_image,
    )


def _execution_summaries_for_backend(namespace: str) -> list[dict[str, Any]]:
    try:
        items = _TEKTON.list(
            namespace, label_selector="app.kubernetes.io/name=benchflow"
        )
    except CommandError:
        return []
    return [_TEKTON.summarize(item).to_dict() for item in items]


def list_benchflow_executions(
    namespace: str,
    *,
    include_completed: bool = True,
) -> list[dict[str, Any]]:
    summaries = _execution_summaries_for_backend(namespace)
    summaries.sort(key=lambda item: item.get("start_time", "") or "", reverse=True)
    if not include_completed:
        summaries = [item for item in summaries if not item.get("finished")]
    return summaries


def _ensure_execution_exists(namespace: str, name: str) -> None:
    if _TEKTON.get(namespace, name) is None:
        raise CommandError(
            f"no BenchFlow execution named {name!r} found in {namespace}"
        )


def get_execution(namespace: str, name: str) -> dict[str, Any]:
    payload = _TEKTON.get(namespace, name)
    if payload is None:
        raise CommandError(f"PipelineRun {name} in namespace {namespace} was not found")
    return payload


def summarize_execution(namespace: str, name: str) -> dict[str, Any]:
    payload = _TEKTON.get(namespace, name)
    if payload is None:
        raise CommandError(f"PipelineRun {name} in namespace {namespace} was not found")
    return _TEKTON.summarize(payload).to_dict()


def cancel_execution(namespace: str, name: str) -> None:
    _ensure_execution_exists(namespace, name)
    _TEKTON.cancel(namespace, name)


def follow_execution(
    namespace: str,
    name: str,
    *,
    poll_interval: int = 5,
) -> bool:
    _ensure_execution_exists(namespace, name)
    return _TEKTON.follow(namespace, name, poll_interval=poll_interval)


def list_execution_steps(namespace: str, name: str) -> list[str]:
    _ensure_execution_exists(namespace, name)
    return _TEKTON.list_steps(namespace, name)


def stream_execution_logs(
    namespace: str,
    name: str,
    *,
    step_name: str | None = None,
    all_logs: bool = False,
    all_containers: bool = False,
) -> None:
    _ensure_execution_exists(namespace, name)
    _TEKTON.logs(
        namespace,
        name,
        step_name=step_name,
        all_logs=all_logs,
        all_containers=all_containers,
    )


def submit_execution_manifest(manifest: dict[str, Any], namespace: str) -> str:
    labels = {
        str(key): str(value)
        for key, value in (manifest.get("metadata", {}) or {}).get("labels", {}).items()
    }
    reservation_name = ""
    if reservation_required_for_labels(labels):
        cluster_name = queue_name_from_labels(labels)
        requested_gpus = requested_gpus_from_labels(labels)
        metadata = manifest.get("metadata", {}) or {}
        execution_prefix = str(
            metadata.get("name") or metadata.get("generateName") or "benchflow"
        ).rstrip("-")
        execution_timeout = str(
            (manifest.get("spec", {}) or {}).get("timeouts", {}).get("pipeline") or "3h"
        )
        reservation_name = create_reservation_workload(
            namespace=namespace,
            cluster_name=cluster_name,
            execution_prefix=execution_prefix,
            requested_gpu_count=requested_gpus,
            execution_timeout=execution_timeout,
        )
        step(
            f"Waiting for queue admission in {cluster_name} for a {requested_gpus}-GPU execution"
        )
        try:
            wait_for_reservation(namespace=namespace, workload_name=reservation_name)
        except BaseException:
            delete_reservation_workload(namespace, reservation_name)
            raise

    try:
        submitted = create_manifest(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True), namespace
        )
        name = submitted.get("metadata", {}).get("name")
        if not name:
            raise ValidationError("execution submission returned no name")
        execution_name = str(name)
        if reservation_name:
            try:
                link_reservation_to_execution(
                    namespace=namespace,
                    workload_name=reservation_name,
                    execution_name=execution_name,
                )
            except BaseException:
                delete_reservation_workload(namespace, reservation_name)
                raise
        return execution_name
    except BaseException:
        if reservation_name:
            delete_reservation_workload(namespace, reservation_name)
        raise


def run_matrix_supervisor(
    plans: list[ResolvedRunPlan],
    *,
    child_execution_name: str,
    benchflow_image: str | None = None,
) -> list[str]:
    if not plans:
        raise ValidationError("matrix execution requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix execution requires all child RunPlans to target the same namespace"
        )

    step(
        f"Running matrix supervisor for {plans[0].metadata.name} "
        f"with {len(plans)} profile combination(s)"
    )
    failures: list[str] = []
    total = len(plans)
    setup_state: dict[str, Any] = {}
    setup_hoisted = False
    setup_state_dir: tempfile.TemporaryDirectory[str] | None = None
    setup_state_path: Path | None = None

    matrix_platform = (
        plans[0].deployment.platform
        if len({plan.deployment.platform for plan in plans}) == 1
        else ""
    )
    if (
        matrix_platform in {"llm-d", "rhoai"}
        and all(plan.stages.deploy for plan in plans)
        and all(plan.stages.cleanup for plan in plans)
    ):
        step(f"Setting up {matrix_platform} platform once for the whole matrix")
        setup_state_dir = tempfile.TemporaryDirectory(prefix="benchflow-matrix-setup-")
        setup_state_path = Path(setup_state_dir.name) / "setup-state.json"
        setup_hoisted = True
        setup_state = setup_platform(
            plans[0],
            context=ExecutionContext(state_path=setup_state_path),
        )

    try:
        for index, plan in enumerate(plans, start=1):
            descriptor = (
                f"deployment={plan.profiles.deployment}, "
                f"benchmark={plan.profiles.benchmark}, "
                f"metrics={plan.profiles.metrics}"
            )
            step(f"[{index}/{total}] Submitting child execution")
            detail(descriptor)
            manifest = render_execution_manifest(
                plan,
                execution_name=child_execution_name,
                setup_mode="skip" if setup_hoisted else "auto",
                teardown=False if setup_hoisted else True,
                skip_kueue_reservation=True,
                benchflow_image=benchflow_image,
            )
            name = submit_execution_manifest(manifest, plan.deployment.namespace)
            detail(f"Created execution {name} in namespace {plan.deployment.namespace}")
            succeeded = follow_execution(plan.deployment.namespace, name)
            if succeeded:
                success(f"[{index}/{total}] {name} succeeded")
                continue
            warning(f"[{index}/{total}] {name} failed")
            failures.append(name)
    finally:
        if setup_hoisted:
            step(f"Tearing down hoisted {plans[0].deployment.platform} platform setup")
            if setup_state_path is not None:
                setup_state = load_setup_state(setup_state_path)
            teardown_platform(
                plans[0],
                setup_state,
                context=ExecutionContext(state_path=setup_state_path),
            )
        if setup_state_dir is not None:
            setup_state_dir.cleanup()

    if failures:
        raise ValidationError(
            f"{len(failures)} matrix child run(s) failed: {', '.join(failures)}"
        )

    success(f"Matrix supervisor completed {total} child execution(s)")
    return failures
