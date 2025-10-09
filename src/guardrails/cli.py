"""Entry point for the ``guardrails`` command-line tool.

This module implements the main CLI for validating and inspecting guardrails
configurations. It provides a straightforward interface for quality control
and manual validation of guardrail bundles.

Example:
    $ guardrails validate config.json --media-type=text/plain
"""

import argparse
import sys
from pathlib import Path

from .runtime import instantiate_guardrails, load_pipeline_bundles


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the Guardrails CLI.

    Sets up the root parser and adds the "validate" subcommand, including all
    required and optional arguments.

    Returns:
        argparse.ArgumentParser: The configured parser instance.
    """
    parser = argparse.ArgumentParser(prog="guardrails", description="Guardrails CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate guardrails configuration",
    )
    validate_parser.add_argument(
        "config_file",
        type=str,
        help="Path to the guardrails JSON configuration file",
    )
    validate_parser.add_argument(
        "-m",
        "--media-type",
        dest="media_type",
        type=str,
        default=None,
        help="Optional media type to filter guardrails (e.g. 'text/plain')",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the Guardrails CLI.

    Parses command-line arguments, loads and validates guardrail configurations,
    and prints summary output. Supports optional filtering by media type.

    Args:
        argv (list[str] | None): Optional list of arguments for testing or
            programmatic use. If not provided, defaults to sys.argv.

    Returns:
        None. Exits with status 0 on success, 1 on validation error, 2 on usage error.

    Example:
        # Validate a configuration file and show results:
        main(["validate", "config.json", "--media-type=text/plain"])
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        try:
            pipeline = load_pipeline_bundles(Path(args.config_file))

            # Collect all guardrails from all stages
            all_guardrails = []
            for stage in pipeline.stages():
                stage_guardrails = instantiate_guardrails(stage)
                all_guardrails.extend(stage_guardrails)

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        total = len(all_guardrails)
        if args.media_type:
            applicable = [g for g in all_guardrails if g.definition.media_type == args.media_type]
            count_applicable = len(applicable)
            print(
                f"Config valid: {total} guardrails loaded, {count_applicable} matching media-type '{args.media_type}'",
            )
        else:
            print(f"Config valid: {total} guardrails loaded")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
