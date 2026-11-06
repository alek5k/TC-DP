#!/usr/bin/env python3
"""Prune non-best validation checkpoints and remove stale incomplete runs.

Deletion rules:
1. In every checkpoint directory, retain the lowest-validation-loss
   ``epoch_*_valloss_*.pth`` file and offer to delete the rest.
2. ``final.pth`` and ``epoch_*_no_val.pth`` are never removed by validation
   pruning.
3. A whole directory is eligible for removal only when its name ends with a
   checkpoint datestamp (``<Environment>_YYYYMMDD_HHMMSS``), its date is
   older than today, and it has no ``final.pth``.
"""

from __future__ import annotations

from datetime import datetime, date
import argparse
from pathlib import Path
import re
import shutil


CHECKPOINTS_DIR = Path.cwd() / "data" / "checkpoints"
DATESTAMP_SUFFIX_RE = re.compile(r"(?:^|_)(\d{8})_?\d{6}(?:_seed\d+)?$")
VALIDATION_CHECKPOINT_RE = re.compile(
    r"^epoch_(?P<epoch>\d+)_valloss_(?P<loss>\d+(?:\.\d+)?)\.pth$"
)


def parse_day_from_folder_name(folder_name: str) -> date | None:
    """Read the date from legacy or environment-prefixed checkpoint names."""
    match = DATESTAMP_SUFFIX_RE.search(folder_name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def confirm_delete(path: Path, reason: str | None = None) -> bool:
    suffix = f" ({reason})" if reason else ""
    response = input(f"Delete '{path.name}'{suffix}? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def validation_checkpoints(path: Path) -> list[tuple[float, int, Path]]:
    """Return validation checkpoints as ``(loss, epoch, path)`` tuples."""
    checkpoints = []
    for checkpoint_path in path.glob("epoch_*_valloss_*.pth"):
        match = VALIDATION_CHECKPOINT_RE.match(checkpoint_path.name)
        if match is None:
            continue
        checkpoints.append((
            float(match.group("loss")),
            int(match.group("epoch")),
            checkpoint_path,
        ))
    return checkpoints


def prune_higher_validation_loss_checkpoints(path: Path) -> tuple[int, int]:
    """Offer to remove all but the best validation checkpoint in ``path``."""
    checkpoints = validation_checkpoints(path)
    if len(checkpoints) < 2:
        return 0, 0

    # A later epoch breaks equal-loss ties so only one best checkpoint remains.
    best_loss, best_epoch, best_path = min(
        checkpoints, key=lambda item: (item[0], -item[1])
    )
    stale_checkpoints = [
        (loss, epoch, checkpoint_path)
        for loss, epoch, checkpoint_path in checkpoints
        if checkpoint_path != best_path
    ]
    print(
        f"BEST  {best_path.name}: validation loss {best_loss:.4f} "
        f"at epoch {best_epoch}"
    )

    removed = 0
    kept = 0
    for loss, epoch, checkpoint_path in sorted(stale_checkpoints):
        relation = "tie" if loss == best_loss else "higher"
        reason = f"retaining epoch {best_epoch}, val_loss={best_loss:.4f} (better)"
        if confirm_delete(checkpoint_path, reason):
            checkpoint_path.unlink()
            removed += 1
            print(f"DELETE {checkpoint_path.name}: {relation} validation loss")
        else:
            kept += 1
            print(f"KEEP  {checkpoint_path.name}: user declined")
    return removed, kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=CHECKPOINTS_DIR,
        help=f"Checkpoint directory to clean (default: {CHECKPOINTS_DIR})",
    )
    args = parser.parse_args()
    checkpoints_dir = args.checkpoint_root

    if not checkpoints_dir.exists():
        print(f"Nothing to do: {checkpoints_dir} does not exist.")
        return 0

    if not checkpoints_dir.is_dir():
        print(f"Nothing to do: {checkpoints_dir} is not a directory.")
        return 0

    today = date.today()
    removed_directories = 0
    removed_checkpoints = 0
    kept = 0

    print("Scanning checkpoint directories...\n")

    for path in sorted(checkpoints_dir.iterdir()):
        if not path.is_dir():
            continue

        print(f"Directory: {path.name}")

        removed, retained = prune_higher_validation_loss_checkpoints(path)
        removed_checkpoints += removed
        kept += retained

        folder_day = parse_day_from_folder_name(path.name)
        if folder_day is None:
            kept += 1
            print(f"KEEP  {path.name}: no checkpoint datestamp suffix")
            print()
            continue

        if folder_day >= today:
            kept += 1
            print(f"KEEP  {path.name}: not older than today")
            print()
            continue

        final_ckpt = path / "final.pth"
        if final_ckpt.exists():
            kept += 1
            print(f"KEEP  {path.name}: final.pth exists")
            print()
            continue

        print(f"ELIGIBLE {path.name}: older than today and missing final.pth")
        if confirm_delete(path):
            shutil.rmtree(path)
            removed_directories += 1
            print(f"DELETE {path.name}: user confirmed")
        else:
            kept += 1
            print(f"KEEP  {path.name}: user declined")
        print()

    print("\nSummary")
    print(f"Removed checkpoint files: {removed_checkpoints}")
    print(f"Removed directories:       {removed_directories}")
    print(f"Kept:    {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
