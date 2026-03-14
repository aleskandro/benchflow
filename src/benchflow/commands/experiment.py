from __future__ import annotations

import argparse
from pathlib import Path

from ..cluster import (
    CommandError,
    cancel_pipelinerun,
    create_manifest,
    follow_pipelinerun,
    get_current_namespace,
    get_pipelinerun,
    list_benchflow_pipelineruns,
    summarize_pipelinerun,
)
from ..models import StageSpec
from ..renderers.deployment import write_deployment_assets
from ..renderers.tekton import render_pipelinerun
from .shared import (
    add_experiment_input_arguments,
    add_parser,
    dump,
    dump_yaml,
    format_experiment_list,
    load_plan,
)
from ..ui import detail, step, success


EXPERIMENT_SUBCOMMANDS = (
    "list",
    "cancel",
    "validate",
    "resolve",
    "render-pipelinerun",
    "render-deployment",
    "run",
    "cleanup",
)

EXPERIMENT_OPTIONS = (
    "--repo-root",
    "--profiles-dir",
    "--namespace",
    "--name",
    "--label",
    "--model",
    "--model-revision",
    "--deployment-profile",
    "--benchmark-profile",
    "--metrics-profile",
    "--service-account",
    "--ttl-seconds-after-finished",
    "--mlflow-experiment",
    "--mlflow-tag",
    "--download",
    "--no-download",
    "--deploy",
    "--no-deploy",
    "--benchmark",
    "--no-benchmark",
    "--collect",
    "--no-collect",
    "--cleanup",
    "--no-cleanup",
)

EXPERIMENT_COMMAND_OPTIONS = {
    "list": ("--namespace", "--all", "--format"),
    "cancel": ("--namespace", "--all"),
    "validate": EXPERIMENT_OPTIONS,
    "resolve": (*EXPERIMENT_OPTIONS, "--format"),
    "render-pipelinerun": (*EXPERIMENT_OPTIONS, "--pipeline-name"),
    "render-deployment": (*EXPERIMENT_OPTIONS, "--output-dir"),
    "run": (*EXPERIMENT_OPTIONS, "--pipeline-name", "--output", "--follow"),
    "cleanup": (*EXPERIMENT_OPTIONS, "--pipeline-name", "--output", "--no-follow"),
}


def _namespace_from_args(args: argparse.Namespace) -> str:
    return args.namespace or get_current_namespace()


def cmd_validate(args: argparse.Namespace) -> int:
    load_plan(args)
    print("valid")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    namespace = _namespace_from_args(args)
    entries = list_benchflow_pipelineruns(
        namespace, include_completed=args.include_completed
    )
    if args.format == "json":
        print(dump(entries, "json"))
        return 0
    if args.format == "yaml":
        print(dump(entries, "yaml"))
        return 0
    if not entries:
        print("No BenchFlow experiments found.")
        return 0
    print(format_experiment_list(entries))
    return 0


def _resolve_cancel_targets(
    namespace: str, identifier: str, *, cancel_all: bool
) -> list[dict[str, object]]:
    try:
        exact = summarize_pipelinerun(get_pipelinerun(namespace, identifier))
    except CommandError:
        exact = None
    else:
        if not exact.get("experiment"):
            raise CommandError(
                f"PipelineRun {identifier} exists in {namespace} but is not a BenchFlow experiment"
            )
        if exact.get("finished"):
            raise CommandError(f"PipelineRun {identifier} is already finished")
        return [exact]

    entries = list_benchflow_pipelineruns(namespace, include_completed=False)
    matches = [entry for entry in entries if entry.get("experiment") == identifier]
    if not matches:
        raise CommandError(
            f"no active BenchFlow experiment found for {identifier!r} in namespace {namespace}"
        )
    if len(matches) > 1 and not cancel_all:
        names = ", ".join(str(entry["name"]) for entry in matches)
        raise CommandError(
            f"multiple active PipelineRuns match experiment {identifier!r}: {names}; "
            "use the PipelineRun name or pass --all"
        )
    return matches if cancel_all else [matches[0]]


def cmd_cancel(args: argparse.Namespace) -> int:
    namespace = _namespace_from_args(args)
    targets = _resolve_cancel_targets(
        namespace, args.identifier, cancel_all=args.all_matches
    )
    step(
        f"Cancelling {len(targets)} BenchFlow PipelineRun"
        f"{'' if len(targets) == 1 else 's'} in namespace {namespace}"
    )
    for target in targets:
        name = str(target["name"])
        detail(f"{name} (experiment={target['experiment']}, status={target['status']})")
        cancel_pipelinerun(namespace, name)
        success(f"Cancellation requested for {name}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    print(dump(plan.to_dict(), args.format))
    return 0


def cmd_render_pipelinerun(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    manifest = render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    print(dump_yaml(manifest))
    return 0


def cmd_render_deployment(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    output_dir = Path(args.output_dir).resolve()
    written = write_deployment_assets(plan, output_dir)
    for path in written:
        print(path)
    return 0


def _render_manifest_yaml(
    args: argparse.Namespace, *, cleanup_only: bool = False
) -> tuple[object, str, str]:
    plan = load_plan(args)
    if cleanup_only:
        plan.stages = StageSpec(
            download=False,
            deploy=False,
            benchmark=False,
            collect=False,
            cleanup=True,
        )

    manifest = render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    manifest_yaml = dump_yaml(manifest)
    namespace = plan.deployment.namespace
    return plan, manifest_yaml, namespace


def _submit_manifest(manifest_yaml: str, namespace: str) -> str:
    submitted = create_manifest(manifest_yaml, namespace)
    name = submitted.get("metadata", {}).get("name")
    if not name:
        raise CommandError("oc create returned no PipelineRun name")
    return str(name)


def cmd_run(args: argparse.Namespace) -> int:
    _, manifest_yaml, namespace = _render_manifest_yaml(args)

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(namespace, name) else 1
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    _, manifest_yaml, namespace = _render_manifest_yaml(args, cleanup_only=True)

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(namespace, name) else 1
    return 0


def register_experiment_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    hidden: bool = False,
) -> None:
    list_runs = add_parser(
        subparsers,
        "list",
        help_text="List BenchFlow experiments in the cluster",
        description="List BenchFlow PipelineRuns in a namespace. Active runs are shown by default.",
        hidden=hidden,
    )
    list_runs.add_argument(
        "--namespace",
        help="Namespace to inspect. Defaults to the current oc project.",
    )
    list_runs.add_argument(
        "--all",
        dest="include_completed",
        action="store_true",
        help="Include finished PipelineRuns as well as active ones.",
    )
    list_runs.add_argument(
        "--format",
        choices=("table", "yaml", "json"),
        default="table",
        help="Output format.",
    )
    list_runs.set_defaults(func=cmd_list, include_completed=False)

    cancel = add_parser(
        subparsers,
        "cancel",
        help_text="Cancel a running BenchFlow experiment",
        description=(
            "Cancel a running BenchFlow PipelineRun by PipelineRun name or by "
            "experiment name label."
        ),
        hidden=hidden,
    )
    cancel.add_argument(
        "identifier",
        help="PipelineRun name or experiment name to cancel.",
    )
    cancel.add_argument(
        "--namespace",
        help="Namespace that contains the PipelineRun. Defaults to the current oc project.",
    )
    cancel.add_argument(
        "--all",
        dest="all_matches",
        action="store_true",
        help="Cancel all active PipelineRuns that match the experiment name.",
    )
    cancel.set_defaults(func=cmd_cancel, all_matches=False)

    validate = add_parser(
        subparsers,
        "validate",
        help_text="Validate an experiment definition",
        description="Validate an experiment file or CLI-defined experiment.",
        hidden=hidden,
    )
    add_experiment_input_arguments(validate)
    validate.set_defaults(func=cmd_validate)

    resolve = add_parser(
        subparsers,
        "resolve",
        help_text="Resolve profiles into a complete RunPlan",
        description="Resolve an experiment into the fully expanded RunPlan used by BenchFlow.",
        hidden=hidden,
    )
    add_experiment_input_arguments(resolve)
    resolve.add_argument("--format", choices=("yaml", "json"), default="yaml")
    resolve.set_defaults(func=cmd_resolve)

    render_pr = add_parser(
        subparsers,
        "render-pipelinerun",
        help_text="Render the Tekton PipelineRun manifest",
        description="Render the Tekton PipelineRun that would be submitted for an experiment.",
        hidden=hidden,
    )
    add_experiment_input_arguments(render_pr)
    render_pr.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference in the rendered PipelineRun.",
    )
    render_pr.set_defaults(func=cmd_render_pipelinerun)

    render_deployment = add_parser(
        subparsers,
        "render-deployment",
        help_text="Render deployment manifests to disk",
        description="Render deployment assets for an experiment without submitting a run.",
        hidden=hidden,
    )
    add_experiment_input_arguments(render_deployment)
    render_deployment.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the rendered deployment assets should be written.",
    )
    render_deployment.set_defaults(func=cmd_render_deployment)

    run = add_parser(
        subparsers,
        "run",
        help_text="Submit an experiment as a PipelineRun",
        description="Submit an experiment to the cluster and optionally follow it.",
        hidden=hidden,
    )
    add_experiment_input_arguments(run)
    run.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference when rendering the PipelineRun.",
    )
    run.add_argument(
        "--output",
        help="Write the rendered PipelineRun manifest to this file before submitting.",
    )
    run.add_argument(
        "--follow",
        action="store_true",
        help="Follow the PipelineRun after submission.",
    )
    run.set_defaults(func=cmd_run, follow=False)

    cleanup = add_parser(
        subparsers,
        "cleanup",
        help_text="Submit a cleanup-only PipelineRun",
        description="Submit a cleanup-only run for an experiment.",
        hidden=hidden,
    )
    add_experiment_input_arguments(cleanup)
    cleanup.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference when rendering the cleanup PipelineRun.",
    )
    cleanup.add_argument(
        "--output",
        help="Write the rendered cleanup PipelineRun manifest to this file before submitting.",
    )
    cleanup.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="Submit the cleanup PipelineRun without following it.",
    )
    cleanup.set_defaults(func=cmd_cleanup, follow=True)
