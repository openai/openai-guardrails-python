"""JSONL dataset loader for guardrail evaluation.

This module provides a loader for reading and validating evaluation datasets in JSONL format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .types import DatasetLoader, Sample
from .validate_dataset import validate_dataset

logger = logging.getLogger(__name__)


class JsonlDatasetLoader(DatasetLoader):
    """Loads and validates datasets from JSONL files."""

    def load(self, path: Path) -> list[Sample]:
        """Load and validate dataset from a JSONL file.

        Args:
            path: Path to the JSONL file

        Returns:
            List of validated samples

        Raises:
            FileNotFoundError: If the dataset file does not exist
            ValueError: If the dataset validation fails
            json.JSONDecodeError: If any line in the file is not valid JSON
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Validate dataset first
        try:
            validate_dataset(path)
        except ValueError as e:
            logger.error("Dataset validation failed: %s", e)
            raise ValueError(f"Dataset validation failed: {e}") from e

        samples: list[Sample] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = Sample.model_validate_json(line)
                        samples.append(sample)
                    except Exception as e:
                        logger.error("Invalid JSON in dataset at line %d: %s", line_num, e)
                        raise ValueError(f"Invalid JSON in dataset at line {line_num}: {e}") from e

            logger.info("Loaded %d samples from %s", len(samples), path)
            return samples

        except OSError as e:
            logger.error("Error reading dataset file: %s", e)
            raise OSError(f"Error reading dataset file: {e}") from e
