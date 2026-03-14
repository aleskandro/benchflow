from __future__ import annotations

from itertools import product

from .loaders import ProfileCatalog
from .models import (
    Experiment,
    ExperimentSpec,
    Metadata,
    MlflowSpec,
    ModelSpec,
    ResolvedRunPlan,
    StageSpec,
    ValidationError,
    normalize_profile_refs,
)
from .plans import resolve_run_plan


def profile_matrix_axes(
    experiment: Experiment,
) -> tuple[list[str], list[str], list[str]]:
    return (
        normalize_profile_refs(
            experiment.spec.deployment_profile, "spec.deployment_profile"
        ),
        normalize_profile_refs(
            experiment.spec.benchmark_profile, "spec.benchmark_profile"
        ),
        normalize_profile_refs(experiment.spec.metrics_profile, "spec.metrics_profile"),
    )


def is_matrix_experiment(experiment: Experiment) -> bool:
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    return any(
        len(values) > 1
        for values in (deployment_profiles, benchmark_profiles, metrics_profiles)
    )


def experiment_matrix_size(experiment: Experiment) -> int:
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    return len(deployment_profiles) * len(benchmark_profiles) * len(metrics_profiles)


def expand_experiment_matrix(experiment: Experiment) -> list[Experiment]:
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    expanded: list[Experiment] = []

    for deployment_profile, benchmark_profile, metrics_profile in product(
        deployment_profiles, benchmark_profiles, metrics_profiles
    ):
        expanded.append(
            Experiment(
                api_version=experiment.api_version,
                kind=experiment.kind,
                metadata=Metadata(
                    name=experiment.metadata.name,
                    labels=dict(experiment.metadata.labels),
                ),
                spec=ExperimentSpec(
                    model=ModelSpec(
                        name=experiment.spec.model.name,
                        revision=experiment.spec.model.revision,
                    ),
                    deployment_profile=[deployment_profile],
                    benchmark_profile=[benchmark_profile],
                    metrics_profile=[metrics_profile],
                    namespace=experiment.spec.namespace,
                    service_account=experiment.spec.service_account,
                    ttl_seconds_after_finished=experiment.spec.ttl_seconds_after_finished,
                    stages=StageSpec(
                        download=experiment.spec.stages.download,
                        deploy=experiment.spec.stages.deploy,
                        benchmark=experiment.spec.stages.benchmark,
                        collect=experiment.spec.stages.collect,
                        cleanup=experiment.spec.stages.cleanup,
                    ),
                    mlflow=MlflowSpec(
                        experiment=experiment.spec.mlflow.experiment,
                        tags=dict(experiment.spec.mlflow.tags),
                    ),
                ),
            )
        )

    return expanded


def resolve_experiment_matrix(
    experiment: Experiment, catalog: ProfileCatalog
) -> list[ResolvedRunPlan]:
    return [
        resolve_run_plan(item, catalog) for item in expand_experiment_matrix(experiment)
    ]


def require_single_experiment_plan(experiment: Experiment) -> Experiment:
    matrix_size = experiment_matrix_size(experiment)
    if matrix_size != 1:
        raise ValidationError(
            f"experiment expands to {matrix_size} profile combinations; "
            "this command only supports a single combination"
        )
    return expand_experiment_matrix(experiment)[0]
