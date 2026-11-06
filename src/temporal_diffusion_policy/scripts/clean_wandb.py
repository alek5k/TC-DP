"""Delete W&B .pth run files that have matching local checkpoints.

Local files are only inspected, never modified. By default, scan the repository's
``data`` and ``data_new`` trees. Supply the W&B account or team with --entity
or WANDB_ENTITY. Use --dry-run to preview remote deletions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath


def collect_checkpoints(directories: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """Index by checkpoint directory and filename, including moved data_new files."""
    checkpoints: dict[tuple[str, str], list[Path]] = {}

    def scan_error(error: OSError) -> None:
        raise error

    for directory in directories:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        # Follow a symlink at the root (data), but not nested directory symlinks.
        for parent, _, filenames in os.walk(directory, onerror=scan_error):
            for filename in filenames:
                path = Path(parent) / filename
                if filename.endswith(".pth") and path.is_file():
                    checkpoints.setdefault((path.parent.name, filename), []).append(path)
    return checkpoints


def find_local_copy(remote, config: dict, checkpoints: dict) -> Path | None:
    """Require a directory/filename match and equal size; never match basename alone."""
    remote_path = PurePosixPath(remote.name)
    if remote_path.suffix != ".pth":
        return None
    if len(remote_path.parts) > 1:
        directory = remote_path.parent.name
    else:
        # Some W&B save configurations strip all directories from the filename.
        checkpoint_dir = config.get("checkpoint_dir")
        if not checkpoint_dir:
            return None
        directory = PurePosixPath(str(checkpoint_dir)).name
    for local in checkpoints.get((directory, remote_path.name), []):
        if local.is_file() and local.stat().st_size == remote.size:
            return local
    return None


def clean_run(run, checkpoints: dict, dry_run: bool) -> tuple[int, int]:
    # Materialize before deleting so pagination cannot skip files during mutation.
    files = [file for file in run.files() if file.name.endswith(".pth")]
    deleted = errors = 0
    for remote in files:
        try:
            local = find_local_copy(remote, run.config, checkpoints)
            if local is None:
                print(f"Skipping (no matching local copy): {run.id}/{remote.name}")
                continue
            action = "Would delete from W&B" if dry_run else "Deleting from W&B"
            print(f"{action}: {run.id}/{remote.name} (local: {local})")
            if not dry_run:
                remote.delete()
            deleted += 1
        except Exception as error:
            errors += 1
            print(f"Error processing {run.id}/{remote.name}: {error}")
    return deleted, errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories", nargs="*", type=Path,
        default=[repo_root / "data", repo_root / "data_new"],
        help="Local checkpoint trees (default: repository data and data_new).",
    )
    parser.add_argument(
        "--entity", default=os.environ.get("WANDB_ENTITY"),
        required=not os.environ.get("WANDB_ENTITY"),
        help="W&B account or team (or set WANDB_ENTITY).",
    )
    parser.add_argument("--project", default="TemporalDiffusionPolicy")
    parser.add_argument("--dry-run", action="store_true", help="Preview remote deletions.")
    args = parser.parse_args()
    try:
        checkpoints = collect_checkpoints(args.directories)
    except OSError as error:
        parser.exit(1, f"Unable to scan local checkpoints: {error}\n")
    if not checkpoints:
        print("No local .pth files found; nothing to delete from W&B.")
        return 0

    import wandb

    api = wandb.Api(timeout=60)
    project = f"{args.entity}/{args.project}"
    print(f"Scanning W&B project {project}; local checkpoint files will be preserved.")
    deleted = errors = 0
    for run in api.runs(project):
        try:
            run_deleted, run_errors = clean_run(run, checkpoints, args.dry_run)
            deleted += run_deleted
            errors += run_errors
        except Exception as error:
            errors += 1
            print(f"Error listing W&B run {run.id}: {error}")
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} {deleted} W&B .pth file(s); {errors} error(s). Local files preserved.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
