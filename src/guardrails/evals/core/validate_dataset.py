"""Dataset validation utility for guardrail evaluation.

This module provides functions and a CLI for validating evaluation datasets in JSONL format.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from .types import Sample

logger = logging.getLogger(__name__)


def validate_dataset(dataset_path: Path) -> tuple[bool, list[str]]:
    """Validate the entire dataset file.

    Args:
        dataset_path: Path to the dataset JSONL file

    Returns:
        Tuple containing:
            - Boolean indicating if validation was successful
            - List of error messages

    Raises:
        FileNotFoundError: If the dataset file does not exist
        OSError: If there are any file I/O errors
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    has_errors = False
    error_messages = []

    try:
        with dataset_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    Sample.model_validate_json(line)
                except ValidationError as e:
                    has_errors = True
                    error_messages.append(f"Line {line_num}: Invalid JSON")
                    error_messages.append(f"  - {str(e)}")
                except Exception as e:
                    has_errors = True
                    error_messages.append(f"Line {line_num}: Invalid Sample format")
                    error_messages.append(f"  - {str(e)}")

    except OSError as e:
        logger.error("Failed to read dataset file: %s", str(e))
        raise

    if not has_errors:
        logger.info("Dataset validation successful")
        return True, ["Validation successful!"]
    else:
        error_messages.insert(0, "Dataset validation failed!")
        logger.error("Dataset validation failed: %s", "\n".join(error_messages))
        return False, error_messages


def main() -> None:
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(description="Validate evaluation dataset format")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the evaluation dataset JSONL file",
    )

    args = parser.parse_args()

    try:
        success, messages = validate_dataset(args.dataset_path)
        for message in messages:
            print(message)
        exit(0 if success else 1)
    except (FileNotFoundError, OSError) as e:
        logger.error("Validation failed: %s", str(e))
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
