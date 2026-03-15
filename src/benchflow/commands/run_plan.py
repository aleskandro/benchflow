from __future__ import annotations

import argparse
from pathlib import Path

import click

from ..cluster import CommandError, create_manifest, follow_pipelinerun
from ..execution import load_run_plan_from_sources
from ..models import ResolvedRunPlan, StageSpec, ValidationError
from ..renderers.tekton import render_pipelinerun
from .shared import dump_yaml, invoke_handler


def _load_run_plan(args: argparse.Namespace) -> ResolvedRunPlan:
    if not args.run_plan and not args.run_plan_json:
        raise ValidationError("provide RUN_PLAN or --run-plan-json")
    return load_run_plan_from_sources(
        run_plan_file=str(args.run_plan) if args.run_plan else None,
        run_plan_json=args.run_plan_json,
    )


def _submit_manifest(manifest_yaml: str, namespace: str) -> str:
    submitted = create_manifest(manifest_yaml, namespace)
    name = submitted.get("metadata", {}).get("name")
    if not name:
        raise CommandError("oc create returned no PipelineRun name")
    return str(name)


def cmd_validate(args: argparse.Namespace) -> int:
    _load_run_plan(args)
    print("valid")
    return 0


def cmd_render_pipelinerun(args: argparse.Namespace) -> int:
    plan = _load_run_plan(args)
    manifest = render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    print(dump_yaml(manifest))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan = _load_run_plan(args)
    manifest_yaml = dump_yaml(
        render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    )

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, plan.deployment.namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(plan.deployment.namespace, name) else 1
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
        render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    )

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, plan.deployment.namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(plan.deployment.namespace, name) else 1
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
    help="Render the Tekton PipelineRun manifest for a resolved RunPlan.",
    short_help="Render a PipelineRun from a RunPlan",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference in the rendered PipelineRun.",
)
def run_plan_render_pipelinerun(**kwargs: object) -> int:
    return invoke_handler(cmd_render_pipelinerun, **kwargs)


@run_plan_group.command(
    "run",
    help="Submit a resolved RunPlan to the cluster and optionally follow it.",
    short_help="Submit a RunPlan as a PipelineRun",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the PipelineRun.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the rendered PipelineRun manifest to this file before submitting.",
)
@click.option(
    "--follow",
    is_flag=True,
    help="Follow the PipelineRun after submission.",
)
def run_plan_run(**kwargs: object) -> int:
    return invoke_handler(cmd_run, **kwargs)


@run_plan_group.command(
    "cleanup",
    help="Submit a cleanup-only PipelineRun from a resolved RunPlan.",
    short_help="Submit a cleanup PipelineRun from a RunPlan",
)
@run_plan_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the cleanup PipelineRun.",
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
    help="Follow the cleanup PipelineRun after submission.",
)
def run_plan_cleanup(**kwargs: object) -> int:
    return invoke_handler(cmd_cleanup, **kwargs)
