"""
Compare paired v15 datasets for episode, step-count, and done-signal parity.

By default, checks these dataset directories:

    datasets/can/mh
    datasets/lift/mh
    datasets/square/mh
    datasets/transport/mh

Example usage:

    python compare_v15_datasets.py
    python compare_v15_datasets.py --dataset_dirs datasets/can/mh datasets/lift/mh
"""

import argparse
import os
import sys

import h5py
import numpy as np


DEFAULT_DATASET_DIRS = (
    "datasets/can/mh",
    "datasets/lift/mh",
    "datasets/square/mh",
    "datasets/transport/mh",
)


def sort_demo_keys(demos):
    """
    Sort demo keys like demo_0, demo_1, ..., demo_10 in numeric order when possible.
    """
    def demo_sort_key(name):
        try:
            return (0, int(name[5:]))
        except (ValueError, IndexError):
            return (1, name)

    return sorted(demos, key=demo_sort_key)


def load_dataset_info(dataset_path, require_dones=True):
    """
    Read per-episode metadata needed for parity checks.
    """
    info = {}
    with h5py.File(dataset_path, "r") as f:
        demos = sort_demo_keys(list(f["data"].keys()))
        for demo in demos:
            demo_group = f["data"][demo]
            num_steps = int(demo_group["actions"].shape[0])
            if require_dones and "dones" not in demo_group:
                raise KeyError("missing 'dones' dataset for episode {}".format(demo))
            dones = demo_group["dones"][()] if "dones" in demo_group else None
            if demo_group.attrs.get("num_samples", num_steps) != num_steps:
                raise ValueError(
                    "episode {} has num_samples attr {} but actions length {}".format(
                        demo, demo_group.attrs.get("num_samples"), num_steps
                    )
                )
            info[demo] = {
                "num_steps": num_steps,
                "dones": dones,
            }
    return info


def compare_episode_and_step_counts(dataset_dir, left_name, right_name):
    """
    Compare episode membership and step counts for a pair of datasets.
    """
    left_path = os.path.join(dataset_dir, left_name)
    right_path = os.path.join(dataset_dir, right_name)

    if not os.path.exists(left_path):
        return ["missing file: {}".format(left_path)]
    if not os.path.exists(right_path):
        return ["missing file: {}".format(right_path)]

    left_info = load_dataset_info(left_path, require_dones=False)
    right_info = load_dataset_info(right_path, require_dones=False)

    mismatches = []

    left_demos = set(left_info.keys())
    right_demos = set(right_info.keys())
    if left_demos != right_demos:
        only_left = sort_demo_keys(list(left_demos - right_demos))
        only_right = sort_demo_keys(list(right_demos - left_demos))
        if only_left:
            mismatches.append("episodes only in {}: {}".format(left_name, ", ".join(only_left)))
        if only_right:
            mismatches.append("episodes only in {}: {}".format(right_name, ", ".join(only_right)))

    shared_demos = sort_demo_keys(list(left_demos & right_demos))
    for demo in shared_demos:
        left_steps = left_info[demo]["num_steps"]
        right_steps = right_info[demo]["num_steps"]
        if left_steps != right_steps:
            mismatches.append(
                "episode {} step count mismatch: {}={} {}={}".format(
                    demo, left_name, left_steps, right_name, right_steps
                )
            )
            continue

    return mismatches


def compare_dones(dataset_dir, left_name, right_name):
    """
    Compare done arrays for a pair of datasets.
    """
    left_path = os.path.join(dataset_dir, left_name)
    right_path = os.path.join(dataset_dir, right_name)

    if not os.path.exists(left_path):
        return ["missing file: {}".format(left_path)]
    if not os.path.exists(right_path):
        return ["missing file: {}".format(right_path)]

    left_info = load_dataset_info(left_path, require_dones=True)
    right_info = load_dataset_info(right_path, require_dones=True)

    mismatches = []

    left_demos = set(left_info.keys())
    right_demos = set(right_info.keys())
    if left_demos != right_demos:
        only_left = sort_demo_keys(list(left_demos - right_demos))
        only_right = sort_demo_keys(list(right_demos - left_demos))
        if only_left:
            mismatches.append("episodes only in {}: {}".format(left_name, ", ".join(only_left)))
        if only_right:
            mismatches.append("episodes only in {}: {}".format(right_name, ", ".join(only_right)))

    shared_demos = sort_demo_keys(list(left_demos & right_demos))
    for demo in shared_demos:
        left_dones = left_info[demo]["dones"]
        right_dones = right_info[demo]["dones"]
        if left_dones.shape != right_dones.shape:
            mismatches.append(
                "episode {} done shape mismatch: {}={} {}={}".format(
                    demo, left_name, left_dones.shape, right_name, right_dones.shape
                )
            )
            continue

        if not np.array_equal(left_dones, right_dones):
            diff_indices = np.where(left_dones != right_dones)[0]
            mismatches.append(
                "episode {} done signal mismatch at {} step(s); first differing index={}".format(
                    demo, diff_indices.shape[0], int(diff_indices[0])
                )
            )

    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=list(DEFAULT_DATASET_DIRS),
        help="dataset directories containing image_v15.hdf5 and low_dim_v15.hdf5",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_v15.hdf5",
        help="image dataset filename inside each dataset directory",
    )
    parser.add_argument(
        "--demo_name",
        type=str,
        default="demo_v15.hdf5",
        help="demo dataset filename inside each dataset directory",
    )
    parser.add_argument(
        "--low_dim_name",
        type=str,
        default="low_dim_v15.hdf5",
        help="low-dim dataset filename inside each dataset directory",
    )
    args = parser.parse_args()

    any_failures = False

    for dataset_dir in args.dataset_dirs:
        image_path = os.path.join(dataset_dir, args.image_name)
        demo_path = os.path.join(dataset_dir, args.demo_name)
        low_dim_path = os.path.join(dataset_dir, args.low_dim_name)
        comparisons = (
            ("demo vs image episodes / steps", args.demo_name, args.image_name, demo_path, image_path, compare_episode_and_step_counts),
            ("demo vs low-dim episodes / steps", args.demo_name, args.low_dim_name, demo_path, low_dim_path, compare_episode_and_step_counts),
            ("image vs low-dim dones", args.image_name, args.low_dim_name, image_path, low_dim_path, compare_dones),
        )

        print("")
        print("==== Comparing {} ====".format(dataset_dir))
        for label, left_name, right_name, left_path, right_path, compare_fn in comparisons:
            print("{}:".format(label))
            print("  left dataset: {}".format(left_path))
            print("  right dataset: {}".format(right_path))
            try:
                mismatches = compare_fn(
                    dataset_dir=dataset_dir,
                    left_name=left_name,
                    right_name=right_name,
                )
            except Exception as exc:
                mismatches = ["error while reading datasets: {}".format(exc)]

            if mismatches:
                any_failures = True
                print("  FAIL")
                for mismatch in mismatches:
                    print("    - {}".format(mismatch))
            else:
                info = load_dataset_info(left_path, require_dones=False)
                total_steps = sum(info[demo]["num_steps"] for demo in info)
                print("  PASS")
                print("    episodes: {}".format(len(info)))
                print("    total steps: {}".format(total_steps))

    print("")
    if any_failures:
        print("Dataset comparison found mismatches.")
        sys.exit(1)

    print("All dataset pairs matched.")


if __name__ == "__main__":
    main()
