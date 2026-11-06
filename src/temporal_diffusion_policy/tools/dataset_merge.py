#!/usr/bin/env python3
"""Merge two compatible Zarr datasets by delegating to diffusion_policy_contrib."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_policy_contrib.tools.dataset_merge import merge_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First input .zarr dataset")
    parser.add_argument("second", type=Path, help="Second input .zarr dataset")
    parser.add_argument("output", type=Path, help="Merged output .zarr dataset")
    parser.add_argument(
        "--remove-last-first",
        action="store_true",
        help="Exclude the final episode from the first source.",
    )
    parser.add_argument(
        "--remove-last-second",
        action="store_true",
        help="Exclude the final episode from the second source.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the merge-plan confirmation prompt.",
    )
    args = parser.parse_args()

    for source in (args.first, args.second):
        if not source.is_dir():
            parser.error(f"Input dataset does not exist: {source}")
    if args.output.exists():
        parser.error(f"Output already exists and will not be overwritten: {args.output}")

    merge_datasets(
        sources=[
            {
                "path": str(args.first),
                "remove_last_episode": args.remove_last_first,
            },
            {
                "path": str(args.second),
                "remove_last_episode": args.remove_last_second,
            },
        ],
        output=str(args.output),
        require_confirmation=not args.yes,
    )


if __name__ == "__main__":
    main()
