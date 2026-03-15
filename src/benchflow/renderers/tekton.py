from __future__ import annotations

import json
from typing import Any

from ..models import ResolvedRunPlan, ValidationError


def render_pipelinerun(
    plan: ResolvedRunPlan,
    pipeline_name: str = "benchflow-e2e",
    *,
    setup_mode: str = "auto",
    teardown: bool = True,
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
    pipeline_name: str = "benchflow-matrix",
    child_pipeline_name: str = "benchflow-e2e",
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    service_accounts = {plan.service_account for plan in plans}
    ttl_values = {plan.ttl_seconds_after_finished for plan in plans}
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
