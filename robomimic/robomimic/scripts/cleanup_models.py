#!/usr/bin/env python3
"""Keep the CLI-selected checkpoint for each Robomimic training run."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SUCCESS_PATTERN = re.compile(r"success_([0-9]+(?:\.[0-9]+)?)")
EPOCH_PATTERN = re.compile(r"epoch_(\d+)")


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str
    retained_path: Path | None


def checkpoint_rank(path: Path) -> tuple[bool, float, int, float]:
    """Match ``experiment_cli.discover_checkpoints`` selection order."""
    success_match = SUCCESS_PATTERN.search(path.name)
    epoch_match = EPOCH_PATTERN.search(path.name)
    success = float(success_match.group(1)) if success_match else None
    epoch = int(epoch_match.group(1)) if epoch_match else None
    return (
        success is not None,
        success if success is not None else float("-inf"),
        epoch if epoch is not None else -1,
        path.stat().st_mtime,
    )


def iter_candidates(root: Path) -> list[Candidate]:
    """Return superseded archives plus all per-epoch resume checkpoints."""
    archive_paths = sorted(root.rglob("model_epoch_*.pth"))
    model_directories = sorted({path.parent for path in archive_paths})
    candidates: list[Candidate] = []

    for model_directory in model_directories:
        run_archives = sorted(model_directory.glob("model_epoch_*.pth"))
        retained_path = max(run_archives, key=checkpoint_rank)
        candidates.extend(
            Candidate(
                path=path,
                reason="superseded_archive",
                retained_path=retained_path,
            )
            for path in run_archives
            if path != retained_path
        )

    candidates.extend(
        Candidate(path=path, reason="per_epoch_resume_checkpoint", retained_path=None)
        for path in sorted(root.rglob("last.pth"))
    )
    return sorted(candidates, key=lambda candidate: candidate.path)


def write_manifest(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("path\tbytes\treason\tretained_checkpoint\n")
        for candidate in candidates:
            retained = str(candidate.retained_path) if candidate.retained_path else ""
            file.write(
                f"{candidate.path}\t{candidate.path.stat().st_size}\t"
                f"{candidate.reason}\t{retained}\n"
            )


def delete_paths(candidates: list[Candidate], dry_run: bool, assume_yes: bool) -> None:
    if not candidates:
        print("No obsolete checkpoints found.")
        return

    total_bytes = sum(candidate.path.stat().st_size for candidate in candidates)
    print(f"Found {len(candidates)} checkpoint(s), reclaiming {total_bytes} bytes.")
    if dry_run:
        for candidate in candidates:
            print(f"[dry-run] {candidate.reason}: {candidate.path}")
        return
    if not assume_yes:
        answer = input("Delete these checkpoints? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("No checkpoints deleted.")
            return

    for candidate in candidates:
        candidate.path.unlink()
    print(f"Deleted {len(candidates)} checkpoint(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the checkpoint selected by the Robomimic CLI for each run, "
            "and delete every last.pth resume checkpoint."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent.parent / "trained_models",
        help="trained_models directory to scan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be deleted without removing files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="delete all matching files without prompting",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write a TSV audit trail before deletion",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    candidates = iter_candidates(root)
    if args.manifest is not None:
        write_manifest(candidates, args.manifest.expanduser())
    delete_paths(candidates, dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
