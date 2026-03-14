from __future__ import annotations

import argparse

import click

from ..loaders import list_profile_entries
from .shared import (
    dump,
    format_profile_list,
    invoke_handler,
    load_profile_document,
    profile_source_options,
    profiles_dir_from,
)


def cmd_profiles_list(args: argparse.Namespace) -> int:
    profiles_dir = profiles_dir_from(args)
    entries = [
        entry
        for entry in list_profile_entries(profiles_dir)
        if args.kind in (None, entry.kind)
    ]
    payload = [entry.to_dict() for entry in entries]

    if args.format == "table":
        output = format_profile_list(payload)
        if output:
            print(output)
        return 0

    print(dump(payload, args.format))
    return 0


def cmd_profiles_show(args: argparse.Namespace) -> int:
    profiles_dir = profiles_dir_from(args)
    document = load_profile_document(profiles_dir, args.name, args.kind)
    print(dump(document, args.format))
    return 0


@click.group(
    "profiles",
    help="List and inspect deployment, benchmark, and metrics profiles.",
    short_help="Inspect available profiles",
)
def profiles_group() -> None:
    pass


@profiles_group.command(
    "list",
    help="List the profiles available in the current repository.",
    short_help="List available profiles",
)
@profile_source_options
@click.option(
    "--kind",
    type=click.Choice(("deployment", "benchmark", "metrics")),
    help="Restrict the list to one profile kind.",
)
@click.option(
    "--format",
    "format",
    type=click.Choice(("table", "yaml", "json")),
    default="table",
    show_default=True,
    help="Output format.",
)
def profiles_list(**kwargs: object) -> int:
    return invoke_handler(cmd_profiles_list, **kwargs)


@profiles_group.command(
    "show",
    help="Print a single profile document in YAML or JSON.",
    short_help="Show a single profile",
)
@profile_source_options
@click.argument("name")
@click.option(
    "--kind",
    type=click.Choice(("deployment", "benchmark", "metrics")),
    help="Profile kind when the name is ambiguous.",
)
@click.option(
    "--format",
    "format",
    type=click.Choice(("yaml", "json")),
    default="yaml",
    show_default=True,
    help="Output format.",
)
def profiles_show(**kwargs: object) -> int:
    return invoke_handler(cmd_profiles_show, **kwargs)
