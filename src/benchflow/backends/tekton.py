from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from ..cluster import CommandError, require_command, run_command, run_json_command
from ..models import ResolvedRunPlan, ValidationError
from ..ui import detail, step, success, warning
from .base import ExecutionSummary


def render_pipelinerun(
    plan: ResolvedRunPlan,
    *,
    pipeline_name: str,
    setup_mode: str,
    teardown: bool,
) -> dict[str, Any]:
    run_plan_json = json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True)
    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "generateName": f"{plan.metadata.name}-",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": plan.metadata.name,
                "benchflow.io/platform": plan.deployment.platform,
                "benchflow.io/mode": plan.deployment.mode,
                "benchflow.io/execution-backend": "tekton",
            },
        },
        "spec": {
            "pipelineRef": {"name": pipeline_name},
            "taskRunTemplate": {"serviceAccountName": plan.service_account},
            "ttlSecondsAfterFinished": plan.ttl_seconds_after_finished,
            "params": [
                {"name": "RUN_PLAN", "value": run_plan_json},
                {
                    "name": "MODELS_STORAGE_PVC",
                    "value": plan.deployment.model_storage.pvc_name,
                },
                {"name": "SETUP_MODE", "value": setup_mode},
                {"name": "TEARDOWN", "value": str(teardown).lower()},
            ],
            "workspaces": [
                {
                    "name": "results",
                    "persistentVolumeClaim": {"claimName": "benchmark-results"},
                },
                {"name": "source", "emptyDir": {}},
            ],
        },
    }


def render_matrix_pipelinerun(
    plans: list[ResolvedRunPlan],
    *,
    pipeline_name: str,
    child_pipeline_name: str,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    service_accounts = {plan.service_account for plan in plans}
    ttl_values = {plan.ttl_seconds_after_finished for plan in plans}
    backends = {plan.execution.backend for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix submission requires all runs to target one namespace"
        )
    if len(service_accounts) != 1:
        raise ValidationError(
            "matrix submission requires all runs to use the same service account"
        )
    if len(ttl_values) != 1:
        raise ValidationError("matrix submission requires a consistent TTL")
    if backends != {"tekton"}:
        raise ValidationError(
            "Tekton matrix submission requires all child RunPlans to use the tekton backend"
        )

    first_plan = plans[0]
    run_plans_json = json.dumps(
        [plan.to_dict() for plan in plans], separators=(",", ":"), sort_keys=True
    )

    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "generateName": f"{first_plan.metadata.name}-matrix-",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": first_plan.metadata.name,
                "benchflow.io/platform": "matrix",
                "benchflow.io/mode": "matrix",
                "benchflow.io/execution-backend": "tekton",
            },
        },
        "spec": {
            "pipelineRef": {"name": pipeline_name},
            "taskRunTemplate": {
                "serviceAccountName": next(iter(service_accounts)),
            },
            "ttlSecondsAfterFinished": next(iter(ttl_values)),
            "params": [
                {"name": "RUN_PLANS", "value": run_plans_json},
                {"name": "CHILD_PIPELINE_NAME", "value": child_pipeline_name},
            ],
        },
    }


def _get_pipelinerun(namespace: str, name: str) -> dict[str, Any] | None:
    require_command("oc")
    result = run_command(
        ["oc", "get", "pipelinerun", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _pipelinerun_state(pipelinerun: dict[str, Any]) -> tuple[str, bool, bool, str]:
    conditions = pipelinerun.get("status", {}).get("conditions", [])
    if not conditions:
        return ("Pending", False, False, "")

    condition = conditions[0]
    status = condition.get("status", "Unknown")
    reason = condition.get("reason", "Unknown")
    message = condition.get("message", "")

    if status == "True":
        return (reason or "Succeeded", True, True, message)
    if status == "False":
        return (reason or "Failed", True, False, message)
    return (reason or "Running", False, False, message)


def _list_pipelineruns(
    namespace: str, *, label_selector: str = ""
) -> list[dict[str, Any]]:
    require_command("oc")
    argv = ["oc", "get", "pipelinerun", "-n", namespace]
    if label_selector:
        argv.extend(["-l", label_selector])
    argv.extend(["-o", "json"])
    payload = run_json_command(argv)
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


class TektonBackend:
    name = "tekton"

    def render_run(
        self,
        plan: ResolvedRunPlan,
        *,
        execution_name: str,
        setup_mode: str,
        teardown: bool,
    ) -> dict[str, Any]:
        return render_pipelinerun(
            plan,
            pipeline_name=execution_name,
            setup_mode=setup_mode,
            teardown=teardown,
        )

    def render_matrix(
        self,
        plans: list[ResolvedRunPlan],
        *,
        execution_name: str,
        child_execution_name: str,
    ) -> dict[str, Any]:
        return render_matrix_pipelinerun(
            plans,
            pipeline_name=execution_name,
            child_pipeline_name=child_execution_name,
        )

    def get(self, namespace: str, name: str) -> dict[str, Any] | None:
        return _get_pipelinerun(namespace, name)

    def list(self, namespace: str, *, label_selector: str = "") -> list[dict[str, Any]]:
        return _list_pipelineruns(namespace, label_selector=label_selector)

    def summarize(self, resource: dict[str, Any]) -> ExecutionSummary:
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {}) or {}
        status, finished, succeeded, message = _pipelinerun_state(resource)
        status_payload = resource.get("status", {}) or {}
        return ExecutionSummary(
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
            experiment=str(labels.get("benchflow.io/experiment", "")),
            platform=str(labels.get("benchflow.io/platform", "")),
            mode=str(labels.get("benchflow.io/mode", "")),
            backend="tekton",
            status=status,
            finished=finished,
            succeeded=succeeded,
            start_time=str(
                status_payload.get("startTime") or metadata.get("creationTimestamp", "")
            ),
            completion_time=str(status_payload.get("completionTime", "")),
            message=str(message),
        )

    def cancel(self, namespace: str, name: str) -> None:
        require_command("oc")
        if shutil.which("tkn") is not None:
            result = subprocess.run(
                ["tkn", "pipelinerun", "cancel", "-n", namespace, name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
            warning(
                "tkn pipelinerun cancel failed; falling back to oc patch: "
                + (result.stderr.strip() or result.stdout.strip() or "unknown error")
            )
        run_command(
            [
                "oc",
                "patch",
                "pipelinerun",
                name,
                "-n",
                namespace,
                "--type",
                "merge",
                "-p",
                '{"spec":{"status":"Cancelled"}}',
            ]
        )

    def follow(self, namespace: str, name: str, *, poll_interval: int = 5) -> bool:
        require_command("oc")

        if shutil.which("tkn") is not None:
            step(f"Following PipelineRun {name} in namespace {namespace}")
            subprocess.run(
                ["tkn", "pipelinerun", "logs", "-f", "-n", namespace, name],
                check=False,
            )
            payload = _get_pipelinerun(namespace, name)
            if payload is None:
                raise CommandError(
                    f"PipelineRun {name} in namespace {namespace} was not found"
                )
            state, finished, succeeded, message = _pipelinerun_state(payload)
            if succeeded:
                success(f"{name}: {state}")
            else:
                warning(f"{name}: {state}")
            if message:
                detail(message)
            return finished and succeeded

        last_state: tuple[str, bool, bool, str] | None = None
        step(f"Watching PipelineRun {name} in namespace {namespace}")
        while True:
            payload = _get_pipelinerun(namespace, name)
            if payload is None:
                raise CommandError(
                    f"PipelineRun {name} in namespace {namespace} was not found"
                )
            state = _pipelinerun_state(payload)
            if state != last_state:
                label, _, _, message = state
                detail(f"{name}: {label}")
                if message:
                    detail(message)
                last_state = state
            label, finished, succeeded, _ = state
            if finished:
                if succeeded:
                    success(f"{name}: {label}")
                else:
                    warning(f"{name}: {label}")
                return succeeded
            time.sleep(poll_interval)
