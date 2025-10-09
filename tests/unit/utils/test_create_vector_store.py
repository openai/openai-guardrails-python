"""Tests for create_vector_store helper."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from guardrails.utils.create_vector_store import SUPPORTED_FILE_TYPES, create_vector_store_from_path


class _FakeAsyncOpenAI:
    def __init__(self) -> None:
        self._vector_store_id = "vs_123"
        self._file_counter = 0
        self._file_status: list[str] = []

        async def create_vector_store(name: str) -> SimpleNamespace:
            _ = name
            return SimpleNamespace(id=self._vector_store_id)

        async def add_file(vector_store_id: str, file_id: str) -> None:
            self._file_status.append("processing")

        async def list_files(vector_store_id: str) -> SimpleNamespace:
            if self._file_status:
                self._file_status = ["completed" for _ in self._file_status]
            return SimpleNamespace(data=[SimpleNamespace(status=s) for s in self._file_status])

        async def create_file(file, purpose: str) -> SimpleNamespace:  # noqa: ANN001
            _ = (file, purpose)
            self._file_counter += 1
            return SimpleNamespace(id=f"file_{self._file_counter}")

        self.vector_stores = SimpleNamespace(
            create=create_vector_store,
            files=SimpleNamespace(create=add_file, list=list_files),
        )
        self.files = SimpleNamespace(create=create_file)


@pytest.mark.asyncio
async def test_create_vector_store_from_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Supported files inside directory should be uploaded and vector store id returned."""
    sample_file = tmp_path / "doc.txt"
    sample_file.write_text("data")

    client = _FakeAsyncOpenAI()

    vector_store_id = await asyncio.wait_for(create_vector_store_from_path(tmp_path, client), timeout=1)

    assert vector_store_id == "vs_123"  # noqa: S101


@pytest.mark.asyncio
async def test_create_vector_store_no_supported_files(tmp_path: Path) -> None:
    """Directory without supported files should raise ValueError."""
    (tmp_path / "ignored.bin").write_text("data")
    client = _FakeAsyncOpenAI()

    with pytest.raises(ValueError):
        await create_vector_store_from_path(tmp_path, client)


def test_supported_file_types_contains_common_extensions() -> None:
    """Ensure supported extensions include basic formats."""
    assert ".txt" in SUPPORTED_FILE_TYPES  # noqa: S101
