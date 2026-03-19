from __future__ import annotations

import argparse
from pathlib import Path

import click

from ..contracts import ResolvedRunPlan, StageSpec, ValidationError
from ..orchestration import (
    follow_execution,
    load_run_plan_from_sources,
    render_execution_manifest,
    submit_execution_manifest,
)
from .shared import dump_yaml, invoke_handler


def _load_run_plan(args: argparse.Namespace) -> ResolvedRunPlan:
    if not args.run_plan and not args.run_plan_json:
        raise ValidationError("provide RUN_PLAN or --run-plan-json")
    return load_run_plan_from_sources(
        run_plan_file=str(args.run_plan) if args.run_plan else None,
        run_plan_json=args.run_plan_json,
    )


def cmd_validate(args: argparse.Namespace) -> int:
    _load_run_plan(args)
    print("valid")
    return 0


def cmd_render_pipelinerun(args: argparse.Namespace) -> int:
    plan = _load_run_plan(args)
    manifest = render_execution_manifest(
        plan,
        execution_name=args.pipeline_name,
        benchflow_image=args.benchflow_image,
    )
    print(dump_yaml(manifest))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan = _load_run_plan(args)
    manifest_yaml = dump_yaml(
        render_execution_manifest(
            plan,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    )

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = submit_execution_manifest(
        render_execution_manifest(
            plan,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        ),
        plan.deployment.namespace,
    )
    print(name)

    if args.follow:
        return (
            0
            if follow_execution(
                plan.deployment.namespace, name, backend=plan.execution.backend
            )
            else 1
        )
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    plan = _load_run_plan(args)
    plan.stages = StageSpec(
        download=False,
        deploy=False,
        benchmark=False,
        collect=False,
        cleanup=True,
    )
    manifest_yaml = dump_yaml(
        render_execution_manifest(
            plan,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    )

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = submit_execution_manifest(
        render_execution_manifest(
            plan,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        ),
        plan.deployment.namespace,
    )
    print(name)

    if args.follow:
        return (
            0
            if follow_execution(
                plan.deployment.namespace, name, backend=plan.execution.backend
            )
            else 1
        )
    return 0


def run_plan_input_options(func):
    decorators = [
        click.argument(
            "run_plan",
            required=False,
            type=click.Path(dir_okay=False, path_type=Path),
        ),
        click.option(
            "--run-plan-json",
            help="Inline RunPlan JSON payload.",
        ),
    ]
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


@click.group(
    "run-plan",
    help="Validate, render, or submit already resolved RunPlan documents.",
    short_help="Work directly with resolved RunPlans",
)
def run_plan_group() -> None:
    pass


@run_plan_group.command(
    "validate",
    help="Validate a resolved RunPlan file or JSON payload.",
    short_help="Validate a RunPlan",
)
@run_plan_input_options
def run_plan_validate(**kwargs: object) -> int:
    return invoke_handler(cmd_validate, **kwargs)


@run_plan_group.command(
    "render-pipelinerun",
    help="Render the PipelineRun manifest for a resolved RunPlan.",
    short_help="Render a PipelineRun from a RunPlan",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference in the rendered PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
def run_plan_render_pipelinerun(**kwargs: object) -> int:
    return invoke_handler(cmd_render_pipelinerun, **kwargs)


@run_plan_group.command(
    "run",
    help="Submit a resolved RunPlan to the cluster and optionally follow it.",
    short_help="Submit a RunPlan as an execution",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the rendered PipelineRun manifest to this file before submitting.",
)
@click.option(
    "--follow",
    is_flag=True,
    help="Follow the execution after submission.",
)
def run_plan_run(**kwargs: object) -> int:
    return invoke_handler(cmd_run, **kwargs)


@run_plan_group.command(
    "cleanup",
    help="Submit a cleanup-only execution from a resolved RunPlan.",
    short_help="Submit a cleanup execution from a RunPlan",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the cleanup PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the rendered cleanup PipelineRun manifest to this file before submitting.",
)
@click.option(
    "--follow/--no-follow",
    default=True,
    show_default=True,
    help="Follow the cleanup execution after submission.",
)
def run_plan_cleanup(**kwargs: object) -> int:
    return invoke_handler(cmd_cleanup, **kwargs)
