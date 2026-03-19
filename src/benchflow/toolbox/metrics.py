from __future__ import annotations

import json
import os
from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..metrics import collect_metrics
from ..remote_jobs import (
    remote_job_artifacts_dir,
    remote_run_plan_json,
    run_remote_job,
)


def collect_plan_metrics(
    plan: ResolvedRunPlan,
    *,
    benchmark_start_time: str,
    benchmark_end_time: str,
    context: ExecutionContext,
    mlflow_run_id: str = "",
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("metrics collection requires an artifacts directory")
    if plan.target_cluster.enabled():
        direct_upload = bool(
            mlflow_run_id and os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        )
        remote = run_remote_job(
            plan,
            job_kind="metrics",
            args_builder=lambda job_name: [
                "metrics",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                "--benchmark-start-time",
                benchmark_start_time,
                "--benchmark-end-time",
                benchmark_end_time,
                "--artifacts-dir",
                remote_job_artifacts_dir(job_name),
                *(
                    [
                        "--mlflow-run-id",
                        mlflow_run_id,
                        "--artifact-path-prefix",
                        "target/metrics",
                        "--cleanup-after-upload",
                        "--upload-direct-to-mlflow",
                    ]
                    if direct_upload
                    else []
                ),
            ],
            mount_results_pvc=True,
        )
        metrics_dir = context.artifacts_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "remote-target-metrics.json").write_text(
            json.dumps(
                {
                    "remote_job_name": remote.job_name,
                    "remote_path": f"{remote_job_artifacts_dir(remote.job_name)}/metrics",
                    "uploaded_to_mlflow": direct_upload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return metrics_dir
    return collect_metrics(
        plan,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        artifacts_dir=context.artifacts_dir,
    )
