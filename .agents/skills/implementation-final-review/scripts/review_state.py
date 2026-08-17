#!/usr/bin/env python3
"""Print deterministic content and repository fingerprints for a review state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", os.fspath(repo), *args), stderr=subprocess.PIPE)


def _git_diff(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", os.fspath(repo), *args),
        capture_output=True,
    )
    if completed.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hidden_index_paths(repo: Path) -> tuple[tuple[str, str], ...]:
    raw_entries = _git(repo, "ls-files", "-v", "-z")
    hidden_paths: list[tuple[str, str]] = []
    for entry in raw_entries.split(b"\0"):
        if len(entry) < 3 or entry[1:2] != b" ":
            continue
        tag = entry[:1]
        relative_path = os.fsdecode(entry[2:])
        if tag.islower():
            hidden_paths.append(("assume-unchanged", relative_path))
        elif tag == b"S":
            candidate = repo / relative_path
            if candidate.exists() or candidate.is_symlink():
                hidden_paths.append(("materialized skip-worktree", relative_path))
    return tuple(sorted(hidden_paths))


def _require_visible_index_state(repo: Path, context: str = "repository") -> None:
    hidden_paths = _hidden_index_paths(repo)
    if hidden_paths:
        details = ", ".join(f"{kind}={path}" for kind, path in hidden_paths)
        raise ValueError(
            f"The {context} contains unsupported hidden index paths: {details}"
        )


def _index_gitlinks(repo: Path) -> dict[str, str]:
    raw_entries = _git(repo, "ls-files", "--stage", "-z")
    gitlinks: dict[str, str] = {}
    for raw_entry in raw_entries.split(b"\0"):
        metadata, separator, raw_path = raw_entry.partition(b"\t")
        fields = metadata.split()
        if separator and len(fields) == 3 and fields[0] == b"160000" and fields[2] == b"0":
            gitlinks[os.fsdecode(raw_path)] = fields[1].decode()
    return gitlinks


def _is_repository_root(path: Path) -> bool:
    try:
        top_level = _git(path, "rev-parse", "--show-toplevel")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return Path(os.fsdecode(top_level.rstrip(b"\n"))).resolve() == path.resolve()


def _require_clean_submodule(repo: Path, display_path: str, expected_head: str) -> None:
    _require_visible_index_state(repo, f"submodule {display_path}")
    actual_head = _git(repo, "rev-parse", "HEAD^{commit}").decode().strip()
    if actual_head != expected_head:
        raise ValueError(
            f"Submodule HEAD does not match the parent index: {display_path}"
        )
    for nested_relative_path, nested_head in _index_gitlinks(repo).items():
        nested_path = repo / nested_relative_path
        _require_clean_gitlink(
            nested_path,
            f"{display_path}/{nested_relative_path}",
            nested_head,
        )
    if _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError(f"Dirty submodule worktrees are unsupported: {display_path}")


def _require_clean_gitlink(path: Path, display_path: str, expected_head: str) -> None:
    if _is_repository_root(path):
        _require_clean_submodule(path, display_path, expected_head)
    elif path.is_dir() and any(path.iterdir()):
        raise ValueError(
            f"Materialized gitlink is not an initialized submodule: {display_path}"
        )


def _require_clean_submodules(repo: Path) -> None:
    for relative_path, expected_head in _index_gitlinks(repo).items():
        _require_clean_gitlink(repo / relative_path, relative_path, expected_head)


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".review-state-diff-",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(data)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _canonical_pathspecs(pathspecs: tuple[str, ...]) -> tuple[str, ...]:
    canonical: list[str] = []
    seen: set[str] = set()
    for pathspec in pathspecs:
        if not pathspec:
            raise ValueError("Pathspecs must not be empty.")
        if "\0" in pathspec:
            raise ValueError("Pathspecs must not contain NUL bytes.")
        if pathspec not in seen:
            canonical.append(pathspec)
            seen.add(pathspec)
    return tuple(canonical)


def _base_has_literal_path(repo: Path, base: str, pathspec: str) -> bool:
    raw_path = os.fsencode(pathspec)
    entries = _git(
        repo,
        "ls-tree",
        "-z",
        base,
        "--",
        f":(literal){pathspec}",
    )
    for entry in entries.split(b"\0"):
        metadata, separator, entry_path = entry.partition(b"\t")
        fields = metadata.split()
        if (
            separator
            and entry_path == raw_path
            and len(fields) >= 2
            and fields[1] != b"tree"
        ):
            return True
    return False


def _load_pathspec_file(path: Path) -> tuple[str, ...]:
    try:
        values = [line for line in path.read_text().splitlines() if line]
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read pathspec file {path}: {error}") from error
    return _canonical_pathspecs(tuple(values))


def _workspace_entry(repo: Path, relative_path: str) -> dict[str, object]:
    path = repo / relative_path
    if path.is_symlink():
        content = b"symlink\0" + os.fsencode(os.readlink(path))
        return {
            "path": relative_path,
            "kind": "symlink",
            "sha256": _digest(content),
        }
    if path.is_file():
        content = b"file\0" + path.read_bytes()
        return {
            "path": relative_path,
            "kind": "file",
            "executable": bool(path.stat().st_mode & 0o111),
            "sha256": _digest(content),
        }
    indexed_head = _index_gitlinks(repo).get(relative_path)
    if path.is_dir():
        if indexed_head is not None:
            return {
                "path": relative_path,
                "kind": "gitlink",
                "head": indexed_head,
            }
        if _is_repository_root(path):
            raise ValueError(
                f"Untracked nested Git repositories are unsupported: {relative_path}"
            )
        return {"path": relative_path, "kind": "directory"}
    if indexed_head is not None:
        return {
            "path": relative_path,
            "kind": "gitlink",
            "head": indexed_head,
        }
    return {"path": relative_path, "kind": "missing"}


def _workspace_entries(
    repo: Path, base: str, pathspecs: tuple[str, ...]
) -> list[dict[str, object]]:
    git_pathspecs = _git_pathspecs(repo, base, pathspecs)
    tracked_paths = _git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "--ignore-submodules=none",
        "-z",
        base,
        "--",
        *git_pathspecs,
    )
    untracked_paths = _untracked_paths(repo, base, pathspecs)
    paths = {
        os.fsdecode(raw_path)
        for raw_path in (*tracked_paths.split(b"\0"), *untracked_paths)
        if raw_path
    }
    return [_workspace_entry(repo, relative_path) for relative_path in sorted(paths)]


def _untracked_paths(repo: Path, base: str, pathspecs: tuple[str, ...]) -> tuple[bytes, ...]:
    literal_pathspecs = _literal_pathspecs(repo, base, pathspecs)
    raw_paths = _git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *_git_pathspecs(repo, base, pathspecs, literal_pathspecs),
    )
    paths = {raw_path for raw_path in raw_paths.split(b"\0") if raw_path}
    for pathspec in literal_pathspecs:
        raw_path = os.fsencode(pathspec)
        tracked_paths = _git(repo, "ls-files", "-z", "--", f":(literal){pathspec}")
        if raw_path not in tracked_paths.split(b"\0"):
            paths.add(raw_path)
    return tuple(sorted(paths))


def _literal_pathspecs(
    repo: Path, base: str, pathspecs: tuple[str, ...]
) -> frozenset[str]:
    literal_pathspecs: set[str] = set()
    for pathspec in pathspecs:
        relative_path = PurePosixPath(pathspec)
        if (
            relative_path.is_absolute()
            or pathspec != relative_path.as_posix()
            or any(part in {".", ".."} for part in relative_path.parts)
        ):
            continue
        candidate = repo.joinpath(*relative_path.parts)
        raw_path = os.fsencode(pathspec)
        tracked_paths = _git(repo, "ls-files", "-z", "--", f":(literal){pathspec}")
        if (
            candidate.is_file()
            or candidate.is_symlink()
            or raw_path in tracked_paths.split(b"\0")
            or _base_has_literal_path(repo, base, pathspec)
        ):
            literal_pathspecs.add(pathspec)
    return frozenset(literal_pathspecs)


def _git_pathspecs(
    repo: Path,
    base: str,
    pathspecs: tuple[str, ...],
    literal_pathspecs: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if literal_pathspecs is None:
        literal_pathspecs = _literal_pathspecs(repo, base, pathspecs)
    return tuple(
        f":(literal){pathspec}" if pathspec in literal_pathspecs else pathspec
        for pathspec in pathspecs
    )


def _complete_diff(repo: Path, base: str, pathspecs: tuple[str, ...]) -> bytes:
    chunks = [
        _git(
            repo,
            "diff",
            "--binary",
            "--full-index",
            "--ignore-submodules=none",
            base,
            "--",
            *_git_pathspecs(repo, base, pathspecs),
        )
    ]
    for raw_path in _untracked_paths(repo, base, pathspecs):
        chunks.append(
            _git_diff(
                repo,
                "diff",
                "--no-index",
                "--binary",
                "--full-index",
                "--",
                "/dev/null",
                os.fsdecode(raw_path),
            )
        )
    return b"".join(chunks)


def _content_fingerprint(base: str, workspace: list[dict[str, object]]) -> str:
    canonical = json.dumps(
        {"base": base, "workspace": workspace},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest(canonical.encode())


def _repository_fingerprint(
    *,
    content_fingerprint: str,
    head: str,
    status_sha256: str,
    tracked_diff_sha256: str,
    complete_diff_sha256: str,
    unfiltered_status_sha256: str,
    unfiltered_content_fingerprint: str,
) -> str:
    canonical = json.dumps(
        {
            "content_fingerprint": content_fingerprint,
            "head": head,
            "status_sha256": status_sha256,
            "tracked_diff_sha256": tracked_diff_sha256,
            "complete_diff_sha256": complete_diff_sha256,
            "unfiltered_status_sha256": unfiltered_status_sha256,
            "unfiltered_content_fingerprint": unfiltered_content_fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest(canonical.encode())


def review_state(
    repo: Path,
    base: str,
    pathspecs: tuple[str, ...] = (),
    components: dict[str, tuple[str, ...]] | None = None,
    complete_diff_output: Path | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    if complete_diff_output is not None:
        complete_diff_output = complete_diff_output.expanduser().resolve()
        if complete_diff_output.is_relative_to(repo):
            raise ValueError("Complete diff output must be outside the repository.")
    _require_visible_index_state(repo)
    _require_clean_submodules(repo)
    pathspecs = _canonical_pathspecs(pathspecs)
    if components and not pathspecs:
        pathspecs = _canonical_pathspecs(
            tuple(
                pathspec
                for component_pathspecs in components.values()
                for pathspec in component_pathspecs
            )
        )
    resolved_base = _git(repo, "rev-parse", f"{base}^{{commit}}").decode().strip()
    head = _git(repo, "rev-parse", "HEAD^{commit}").decode().strip()
    try:
        _git(repo, "merge-base", "--is-ancestor", resolved_base, head)
    except subprocess.CalledProcessError as error:
        raise ValueError("Base must be an ancestor of HEAD.") from error
    tracked_diff = _git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--ignore-submodules=none",
        resolved_base,
        "--",
        *_git_pathspecs(repo, resolved_base, pathspecs),
    )
    complete_diff = _complete_diff(repo, resolved_base, pathspecs)
    status = _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        "--",
        *_git_pathspecs(repo, resolved_base, pathspecs),
    )
    workspace = _workspace_entries(repo, resolved_base, pathspecs)
    unfiltered_status = _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    unfiltered_workspace = _workspace_entries(repo, resolved_base, ())
    unfiltered_by_path = {str(entry["path"]): entry for entry in unfiltered_workspace}
    for entry in workspace:
        unfiltered_by_path.setdefault(str(entry["path"]), entry)
    unfiltered_workspace = [unfiltered_by_path[path] for path in sorted(unfiltered_by_path)]

    content_fingerprint = _content_fingerprint(resolved_base, workspace)
    component_states: dict[str, dict[str, object]] = {}
    component_owners: dict[str, list[str]] = {}
    for name, component_pathspecs in sorted((components or {}).items()):
        canonical_component_pathspecs = _canonical_pathspecs(component_pathspecs)
        if not canonical_component_pathspecs:
            raise ValueError(f"Component manifest is empty: {name}")
        component_workspace = _workspace_entries(repo, resolved_base, canonical_component_pathspecs)
        for entry in component_workspace:
            component_owners.setdefault(str(entry["path"]), []).append(name)
        component_states[name] = {
            "content_fingerprint": _content_fingerprint(resolved_base, component_workspace),
            "pathspecs": list(canonical_component_pathspecs),
            "workspace": component_workspace,
        }
    if component_states:
        combined_paths = {str(entry["path"]) for entry in workspace}
        component_paths = set(component_owners)
        missing_paths = sorted(combined_paths - component_paths)
        extra_paths = sorted(component_paths - combined_paths)
        overlapping_paths = {
            path: owners for path, owners in component_owners.items() if len(owners) > 1
        }
        if missing_paths or extra_paths or overlapping_paths:
            raise ValueError(
                "Component manifests must partition the combined review content exactly: "
                f"missing={missing_paths}, extra={extra_paths}, "
                f"overlapping={overlapping_paths}"
            )

    repository_state = {
        "content_fingerprint": content_fingerprint,
        "head": head,
        "status_sha256": _digest(status),
        "tracked_diff_sha256": _digest(tracked_diff),
        "complete_diff_sha256": _digest(complete_diff),
    }
    repository_fingerprint = _repository_fingerprint(
        **repository_state,
        unfiltered_status_sha256=_digest(unfiltered_status),
        unfiltered_content_fingerprint=_content_fingerprint(resolved_base, unfiltered_workspace),
    )
    if complete_diff_output is not None:
        _write_bytes_atomically(complete_diff_output, complete_diff)
    return {
        "fingerprint": content_fingerprint,
        "content_fingerprint": content_fingerprint,
        "repository_fingerprint": repository_fingerprint,
        "base": resolved_base,
        "pathspecs": list(pathspecs),
        "workspace": workspace,
        "complete_diff_paths": [str(entry["path"]) for entry in workspace],
        "components": component_states,
        "unfiltered": {
            "status_sha256": _digest(unfiltered_status),
            "workspace": unfiltered_workspace,
        },
        **repository_state,
    }


def _parse_component_files(values: list[str]) -> dict[str, tuple[str, ...]]:
    components: dict[str, tuple[str, ...]] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not raw_path:
            raise ValueError(
                "Component pathspec files must use lowercase NAME=FILE with a nonempty file."
            )
        if name in components:
            raise ValueError(f"Duplicate component name: {name}")
        components[name] = _load_pathspec_file(Path(raw_path))
    return components


def _component(value: str) -> tuple[str, str]:
    name, separator, pathspec = value.partition("=")
    if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not pathspec:
        raise argparse.ArgumentTypeError("component must use lowercase NAME=PATHSPEC")
    if "\0" in pathspec:
        raise argparse.ArgumentTypeError("component pathspec must not contain NUL bytes")
    return name, pathspec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Resolved merge-base commit or revision.")
    parser.add_argument(
        "--pathspec",
        action="append",
        default=[],
        help="Task-owned Git pathspec. Repeat to scope the review; omit to include all changes.",
    )
    parser.add_argument(
        "--pathspec-file",
        action="append",
        default=[],
        type=Path,
        help="File containing canonical task-owned pathspecs, one per line.",
    )
    parser.add_argument(
        "--component-pathspec-file",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="Named component manifest. Repeat for runtime, tests-examples, or metadata.",
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        type=_component,
        metavar="NAME=PATHSPEC",
        help="Named component pathspec. Repeat a name to group paths into one fingerprint.",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository worktree path.")
    parser.add_argument(
        "--complete-diff-output",
        type=Path,
        help="Write the complete binary diff, including task-owned untracked files, to this path.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output.")
    args = parser.parse_args()
    try:
        loaded_pathspec_files = [_load_pathspec_file(path) for path in args.pathspec_file]
        if any(not pathspecs for pathspecs in loaded_pathspec_files):
            raise ValueError("A supplied pathspec file must contain at least one pathspec.")
        file_pathspecs = tuple(
            pathspec for pathspecs in loaded_pathspec_files for pathspec in pathspecs
        )
        pathspecs = _canonical_pathspecs((*args.pathspec, *file_pathspecs))
        component_files = _parse_component_files(args.component_pathspec_file)
        component_values: dict[str, list[str]] = {
            name: list(component_pathspecs) for name, component_pathspecs in component_files.items()
        }
        for name, pathspec in args.component:
            component_values.setdefault(name, []).append(pathspec)
        components = {
            name: _canonical_pathspecs(tuple(component_pathspecs))
            for name, component_pathspecs in component_values.items()
        }
        state = review_state(
            args.repo,
            args.base,
            pathspecs,
            components,
            complete_diff_output=args.complete_diff_output,
        )
    except ValueError as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        parser.error(f"Git command failed with exit status {error.returncode}.")
    except (OSError, UnicodeError) as error:
        parser.error(f"Cannot inspect repository state: {error}")
    print(
        json.dumps(
            state,
            ensure_ascii=True,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
