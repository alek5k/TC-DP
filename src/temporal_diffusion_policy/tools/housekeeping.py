import os
import re
import json
from pathlib import Path
from typing import Literal
import torch

PROJECT_ROOT = Path.cwd()

def cleanup_checkpoint_files(checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints", dry_run=True):
    """
    Delete all .pth files in checkpoints directory and subdirectories,
    except for files named 'final.pth'.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        dry_run: If True, only print what would be deleted without actually deleting (default: True)
    
    Returns:
        tuple: (num_deleted, total_size_mb) - number of files deleted and total size in MB
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0, 0.0
    
    files_to_delete = []
    total_size = 0
    
    # Find all .pth files recursively
    for pth_file in checkpoints_path.rglob("*.pth"):
        # Skip if the file is named 'final.pth'
        if pth_file.name == "final.pth":
            print(f"Keeping: {pth_file}")
            continue
        
        file_size = pth_file.stat().st_size
        files_to_delete.append((pth_file, file_size))
        total_size += file_size
    
    # Convert total size to MB
    total_size_mb = total_size / (1024 * 1024)
    
    if not files_to_delete:
        print("No checkpoint files to delete.")
        return 0, 0.0
    
    print(f"\nFound {len(files_to_delete)} files to delete (total: {total_size_mb:.2f} MB)")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no files will be deleted) ---")
        for file_path, size in files_to_delete:
            print(f"Would delete: {file_path} ({size / (1024 * 1024):.2f} MB)")
        print("\nTo actually delete these files, run with dry_run=False")
        return 0, 0.0
    else:
        # Actually delete the files
        num_deleted = 0
        for file_path, size in files_to_delete:
            try:
                file_path.unlink()
                print(f"Deleted: {file_path} ({size / (1024 * 1024):.2f} MB)")
                num_deleted += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        print(f"\nDeleted {num_deleted} files, freed {total_size_mb:.2f} MB")
        return num_deleted, total_size_mb


def rename_checkpoint_parameter(
    checkpoint_subdirs=None,
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    subdict_key="train_cfg",
    old_param="in_use_saturating_time_at_rest_encoding_alpha",
    new_param="in_use_idleness_encoding_alpha",
    dry_run=True
):
    """
    Rename a parameter in checkpoint files, processing final.pth in each checkpoint subdirectory.
    
    This function iterates through checkpoint subdirectories and modifies final.pth files,
    renaming the specified parameter within a subdictionary of the checkpoint.
    
    Args:
        checkpoint_subdirs: List of subdirectory names to process (e.g., ["20260122_071500"]). 
                           If None, processes all subdirectories in checkpoints_dir.
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        subdict_key: The key of the subdictionary containing the parameter (default: "train_cfg")
        old_param: The old parameter name to find (default: "in_use_saturating_time_at_rest_encoding_alpha")
        new_param: The new parameter name to use (default: "in_use_idleness_encoding_alpha")
        dry_run: If True, only report what would be changed without modifying files (default: True)
    
    Returns:
        int: number of checkpoint files modified
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0
    
    # Get subdirectories to process
    if checkpoint_subdirs is None:
        # Get all subdirectories
        subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    else:
        # Use specified subdirectories
        subdirs = [checkpoints_path / subdir for subdir in checkpoint_subdirs]
        # Filter out non-existent directories
        subdirs = [d for d in subdirs if d.exists() and d.is_dir()]
    
    if not subdirs:
        print("No checkpoint subdirectories found to process.")
        return 0
    
    print(f"Found {len(subdirs)} checkpoint subdirectories to check.")
    print(f"Looking in subdictionary: '{subdict_key}'")
    print(f"Looking for parameter: '{old_param}'")
    print(f"Will rename to: '{new_param}'\n")
    
    # sort by name for consistent output
    subdirs.sort(key=lambda x: x.name)

    modified_files = []
    
    for subdir in subdirs:
        final_pth = subdir / "final.pth"
        
        if not final_pth.exists():
            continue
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
            
            # Check if this is a dict-like checkpoint
            if not isinstance(checkpoint, dict):
                print(f"Skipping {subdir.name}/final.pth: not a dictionary")
                continue
            
            # Check if the subdictionary exists
            if subdict_key not in checkpoint:
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' not found")
                continue
            
            # Check if the subdictionary is a dict
            if not isinstance(checkpoint[subdict_key], dict):
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' is not a dictionary")
                continue
            
            # Check if the parameter exists in the subdictionary
            if old_param in checkpoint[subdict_key]:
                # Rename the parameter
                checkpoint[subdict_key][new_param] = checkpoint[subdict_key].pop(old_param)
                modified_files.append((final_pth, checkpoint))
                print(f"✓ Found parameter '{old_param}' in: {subdir.name}/final.pth")
            else:
                print(f"✗ Parameter '{old_param}' not found in: {subdir.name}/final.pth")

        except Exception as e:
            print(f"Error processing {subdir.name}/final.pth: {e}")
            continue
    
    if not modified_files:
        print(f"\nNo files contain the parameter '{old_param}' to rename.")
        return 0
    
    print(f"\nFound parameter '{old_param}' in {len(modified_files)} files.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no files will be modified) ---")
        for file_path, _ in modified_files:
            print(f"Would modify: {file_path.parent.name}/final.pth")
        print("\nTo actually modify these files, run with dry_run=False")
        return 0
    else:
        # Save the modified checkpoints
        num_modified = 0
        for file_path, modified_checkpoint in modified_files:
            try:
                torch.save(modified_checkpoint, file_path)
                print(f"Modified: {file_path.parent.name}/final.pth")
                num_modified += 1
            except Exception as e:
                print(f"Error saving {file_path.parent.name}/final.pth: {e}")
        
        print(f"\nSuccessfully modified {num_modified} checkpoint files.")
        return num_modified


def remove_checkpoint_parameter(
    checkpoint_subdirs=None,
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    subdict_key="train_cfg",
    param_to_remove="old_parameter_name",
    dry_run=True
):
    """
    Remove a parameter from checkpoint files, processing final.pth in each checkpoint subdirectory.
    
    This function iterates through checkpoint subdirectories and modifies final.pth files,
    removing the specified parameter from a subdictionary of the checkpoint.
    
    Args:
        checkpoint_subdirs: List of subdirectory names to process (e.g., ["20260122_071500"]). 
                           If None, processes all subdirectories in checkpoints_dir.
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        subdict_key: The key of the subdictionary containing the parameter (default: "train_cfg")
        param_to_remove: The parameter name to remove
        dry_run: If True, only report what would be changed without modifying files (default: True)
    
    Returns:
        int: number of checkpoint files modified
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0
    
    # Get subdirectories to process
    if checkpoint_subdirs is None:
        # Get all subdirectories
        subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    else:
        # Use specified subdirectories
        subdirs = [checkpoints_path / subdir for subdir in checkpoint_subdirs]
        # Filter out non-existent directories
        subdirs = [d for d in subdirs if d.exists() and d.is_dir()]
    
    if not subdirs:
        print("No checkpoint subdirectories found to process.")
        return 0
    
    print(f"Found {len(subdirs)} checkpoint subdirectories to check.")
    print(f"Looking in subdictionary: '{subdict_key}'")
    print(f"Parameter to remove: '{param_to_remove}'\n")
    
    # sort by name for consistent output
    subdirs.sort(key=lambda x: x.name)

    modified_files = []
    
    for subdir in subdirs:
        final_pth = subdir / "final.pth"
        
        if not final_pth.exists():
            continue
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
            
            # Check if this is a dict-like checkpoint
            if not isinstance(checkpoint, dict):
                print(f"Skipping {subdir.name}/final.pth: not a dictionary")
                continue
            
            # Check if the subdictionary exists
            if subdict_key not in checkpoint:
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' not found")
                continue
            
            # Check if the subdictionary is a dict
            if not isinstance(checkpoint[subdict_key], dict):
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' is not a dictionary")
                continue
            
            # Check if the parameter exists in the subdictionary
            if param_to_remove in checkpoint[subdict_key]:
                # Remove the parameter
                old_value = checkpoint[subdict_key].pop(param_to_remove)
                modified_files.append((final_pth, checkpoint))
                print(f"✓ Found parameter '{param_to_remove}' [{old_value}] in: {subdir.name}/final.pth")
            else:
                print(f"✗ Parameter '{param_to_remove}' not found in: {subdir.name}/final.pth")

        except Exception as e:
            print(f"Error processing {subdir.name}/final.pth: {e}")
            continue
    
    if not modified_files:
        print(f"\nNo files contain the parameter '{param_to_remove}' to remove.")
        return 0
    
    print(f"\nFound parameter '{param_to_remove}' in {len(modified_files)} files.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no files will be modified) ---")
        for file_path, _ in modified_files:
            print(f"Would modify: {file_path.parent.name}/final.pth")
        print("\nTo actually modify these files, run with dry_run=False")
        return 0
    else:
        # Save the modified checkpoints
        num_modified = 0
        for file_path, modified_checkpoint in modified_files:
            try:
                torch.save(modified_checkpoint, file_path)
                print(f"Modified: {file_path.parent.name}/final.pth")
                num_modified += 1
            except Exception as e:
                print(f"Error saving {file_path.parent.name}/final.pth: {e}")
        
        print(f"\nSuccessfully modified {num_modified} checkpoint files.")
        return num_modified


def add_checkpoint_parameter(
    checkpoint_subdirs=None,
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    subdict_key="train_cfg",
    param_to_add="new_parameter_name",
    value=None,
    dry_run=True
):
    """
    Add a parameter to checkpoint files if it doesn't exist, processing final.pth in each checkpoint subdirectory.
    
    This function iterates through checkpoint subdirectories and modifies final.pth files,
    adding the specified parameter to a subdictionary if it doesn't already exist.
    
    Args:
        checkpoint_subdirs: List of subdirectory names to process (e.g., ["20260122_071500"]). 
                           If None, processes all subdirectories in checkpoints_dir.
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        subdict_key: The key of the subdictionary to add the parameter to (default: "train_cfg")
        param_to_add: The parameter name to add
        value: The value to set for the parameter
        dry_run: If True, only report what would be changed without modifying files (default: True)
    
    Returns:
        int: number of checkpoint files modified
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0
    
    # Get subdirectories to process
    if checkpoint_subdirs is None:
        # Get all subdirectories
        subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    else:
        # Use specified subdirectories
        subdirs = [checkpoints_path / subdir for subdir in checkpoint_subdirs]
        # Filter out non-existent directories
        subdirs = [d for d in subdirs if d.exists() and d.is_dir()]
    
    if not subdirs:
        print("No checkpoint subdirectories found to process.")
        return 0
    
    print(f"Found {len(subdirs)} checkpoint subdirectories to check.")
    print(f"Looking in subdictionary: '{subdict_key}'")
    print(f"Parameter to add: '{param_to_add}'")
    print(f"Value: {value}\n")
    
    # sort by name for consistent output
    subdirs.sort(key=lambda x: x.name)

    modified_files = []
    
    for subdir in subdirs:
        final_pth = subdir / "final.pth"
        
        if not final_pth.exists():
            continue
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
            
            # Check if this is a dict-like checkpoint
            if not isinstance(checkpoint, dict):
                print(f"Skipping {subdir.name}/final.pth: not a dictionary")
                continue
            
            # Check if the subdictionary exists
            if subdict_key not in checkpoint:
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' not found")
                continue
            
            # Check if the subdictionary is a dict
            if not isinstance(checkpoint[subdict_key], dict):
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' is not a dictionary")
                continue
            
            # Check if the parameter already exists
            if param_to_add in checkpoint[subdict_key]:
                existing_value = checkpoint[subdict_key][param_to_add]
                print(f"○ Parameter '{param_to_add}' already exists in: {subdir.name}/final.pth (value: {existing_value})")
            else:
                # Add the parameter
                checkpoint[subdict_key][param_to_add] = value
                modified_files.append((final_pth, checkpoint))
                print(f"✓ Will add parameter '{param_to_add}' [{value}] to: {subdir.name}/final.pth")

        except Exception as e:
            print(f"Error processing {subdir.name}/final.pth: {e}")
            continue
    
    if not modified_files:
        print(f"\nNo files need the parameter '{param_to_add}' added (already exists in all or none found).")
        return 0
    
    print(f"\nWill add parameter '{param_to_add}' to {len(modified_files)} files.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no files will be modified) ---")
        for file_path, _ in modified_files:
            print(f"Would modify: {file_path.parent.name}/final.pth")
        print("\nTo actually modify these files, run with dry_run=False")
        return 0
    else:
        # Save the modified checkpoints
        num_modified = 0
        for file_path, modified_checkpoint in modified_files:
            try:
                torch.save(modified_checkpoint, file_path)
                print(f"Modified: {file_path.parent.name}/final.pth")
                num_modified += 1
            except Exception as e:
                print(f"Error saving {file_path.parent.name}/final.pth: {e}")
        
        print(f"\nSuccessfully modified {num_modified} checkpoint files.")
        return num_modified


def compare_checkpoint_keys(
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    subdict_key="train_cfg",
    do_checkpoint_specific_analysis=False
):
    """
    Compare keys in a subdictionary across all checkpoint files to find differences.
    
    This function loads final.pth from each checkpoint subdirectory and compares
    the keys in the specified subdictionary to identify which keys are present in
    some checkpoints but not others.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        subdict_key: The key of the subdictionary to compare (default: "train_cfg")
        do_checkpoint_specific_analysis: Whether to perform checkpoint-specific analysis (default: False)
    Returns:
        dict: A dictionary with analysis results including:
            - 'all_keys': set of all keys found across all checkpoints
            - 'common_keys': set of keys present in all checkpoints
            - 'key_presence': dict mapping each key to list of checkpoints containing it
            - 'checkpoint_keys': dict mapping each checkpoint to its set of keys
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return None
    
    # Get all subdirectories
    subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: x.name)
    
    if not subdirs:
        print("No checkpoint subdirectories found.")
        return None
    
    print(f"Analyzing {len(subdirs)} checkpoint subdirectories...")
    print(f"Comparing keys in subdictionary: '{subdict_key}'\n")
    
    checkpoint_keys = {}
    checkpoint_values = {}  # Store the actual values for each checkpoint
    all_keys = set()
    
    for subdir in subdirs:
        final_pth = subdir / "final.pth"
        
        if not final_pth.exists():
            print(f"Skipping {subdir.name}: no final.pth found")
            continue
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
            
            # Check if this is a dict-like checkpoint
            if not isinstance(checkpoint, dict):
                print(f"Skipping {subdir.name}: not a dictionary")
                continue
            
            # Check if the subdictionary exists
            if subdict_key not in checkpoint:
                print(f"Skipping {subdir.name}: '{subdict_key}' not found")
                continue
            
            # Check if the subdictionary is a dict
            if not isinstance(checkpoint[subdict_key], dict):
                print(f"Skipping {subdir.name}: '{subdict_key}' is not a dictionary")
                continue
            
            # Get the keys and values
            keys = set(checkpoint[subdict_key].keys())
            checkpoint_keys[subdir.name] = keys
            checkpoint_values[subdir.name] = checkpoint[subdict_key]
            all_keys.update(keys)
            
        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
            continue
    
    if not checkpoint_keys:
        print(f"\nNo valid checkpoints with '{subdict_key}' subdictionary found.")
        return None
    
    print(f"\nSuccessfully loaded {len(checkpoint_keys)} checkpoints.\n")
    
    # Find common keys (present in all checkpoints)
    common_keys = set.intersection(*checkpoint_keys.values()) if checkpoint_keys else set()
    
    # Build key presence map
    key_presence = {}
    for key in sorted(all_keys):
        checkpoints_with_key = [ckpt for ckpt, keys in checkpoint_keys.items() if key in keys]
        key_presence[key] = checkpoints_with_key
    
    # Print results
    print("="*80)
    print(f"SUMMARY:")
    print(f"  Total unique keys found: {len(all_keys)}")
    print(f"  Keys present in all checkpoints: {len(common_keys)}")
    print(f"  Keys present in some checkpoints: {len(all_keys - common_keys)}")
    print("="*80)
    
    # Print common keys
    if common_keys:
        print(f"\nKeys present in ALL {len(checkpoint_keys)} checkpoints:")
        for key in sorted(common_keys):
            # Get all values for this key across checkpoints
            values = set()
            for ckpt_name in checkpoint_keys.keys():
                val = checkpoint_values[ckpt_name].get(key)
                values.add(str(val))
            
            # If all values are the same, show once; otherwise show "VARIES"
            if len(values) == 1:
                print(f"  ✓ {key} [{values.pop()}]")
            else:
                print(f"  ✓ {key} [VARIES: {', '.join(sorted(values))}]")
    
    # Print keys that differ
    differing_keys = all_keys - common_keys
    if differing_keys:
        print(f"\nKeys present in SOME (not all) checkpoints:")
        for key in sorted(differing_keys):
            checkpoints_with_key = key_presence[key]
            count = len(checkpoints_with_key)
            
            # Get all values for this key
            values = set()
            for ckpt_name in checkpoints_with_key:
                val = checkpoint_values[ckpt_name].get(key)
                values.add(str(val))
            
            if len(values) == 1:
                print(f"\n  • {key} [{values.pop()}]")
            else:
                print(f"\n  • {key} [VARIES: {', '.join(sorted(values))}]")
            print(f"    Present in {count}/{len(checkpoint_keys)} checkpoints:")
            for ckpt in checkpoints_with_key:
                val = checkpoint_values[ckpt].get(key)
                print(f"      - {ckpt} [{val}]")
    
    if do_checkpoint_specific_analysis:
        # Find checkpoints that have unique key sets
        print(f"\n{'='*80}")
        print("CHECKPOINT-SPECIFIC ANALYSIS:")
        print("="*80)
        
        for ckpt_name in sorted(checkpoint_keys.keys()):
            keys = checkpoint_keys[ckpt_name]
            unique_keys = keys - common_keys
            missing_keys = common_keys - keys
            
            if unique_keys or missing_keys:
                print(f"\n{ckpt_name}:")
                if unique_keys:
                    print(f"  Extra keys (not in all):")
                    for key in sorted(unique_keys):
                        val = checkpoint_values[ckpt_name].get(key)
                        print(f"    + {key} [{val}]")
                if missing_keys:
                    print(f"  Missing keys (that others have):")
                    for key in sorted(missing_keys):
                        print(f"    - {key}")
    
    return {
        'all_keys': all_keys,
        'common_keys': common_keys,
        'key_presence': key_presence,
        'checkpoint_keys': checkpoint_keys
    }


def check_inference_checkpoint_paths(inference_dir=f"{PROJECT_ROOT}/data/inference"):
    """
    Check if checkpoint paths referenced in inference zarr files exist.
    
    This function loops through each .zarr directory in data/inference/,
    reads the .zattrs file, extracts the 'checkpoint_path', and checks
    whether that checkpoint path exists on the filesystem.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
    
    Returns:
        dict: Results with 'valid' and 'invalid' lists of tuples (zarr_name, checkpoint_path, exists)
    """
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return None
    
    # Find all .zarr directories
    zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    zarr_dirs.sort()
    
    if not zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return None
    
    print(f"Found {len(zarr_dirs)} .zarr directories to check.\n")
    
    valid_checkpoints = []
    invalid_checkpoints = []
    
    for zarr_dir in zarr_dirs:
        zattrs_file = zarr_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"⚠ {zarr_dir.name}: .zattrs file not found")
            continue
        
        try:
            # Read the .zattrs JSON file
            with open(zattrs_file, 'r') as f:
                attrs = json.load(f)
            
            # Extract checkpoint_path
            if 'checkpoint_path' not in attrs:
                print(f"⚠ {zarr_dir.name}: no 'checkpoint_path' field in .zattrs")
                continue
            
            checkpoint_path = attrs['checkpoint_path']
            checkpoint_path_obj = Path(checkpoint_path)
            
            # Check if the checkpoint path exists
            exists = checkpoint_path_obj.exists()
            
            if exists:
                valid_checkpoints.append((zarr_dir.name, checkpoint_path, exists))
                print(f"✓ {zarr_dir.name}")
                print(f"  Checkpoint: {checkpoint_path} [EXISTS]")
            else:
                invalid_checkpoints.append((zarr_dir.name, checkpoint_path, exists))
                print(f"✗ {zarr_dir.name}")
                print(f"  Checkpoint: {checkpoint_path} [NOT FOUND]")
                
        except json.JSONDecodeError as e:
            print(f"⚠ {zarr_dir.name}: error parsing .zattrs JSON: {e}")
        except Exception as e:
            print(f"⚠ {zarr_dir.name}: error reading .zattrs: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Total .zarr directories checked: {len(zarr_dirs)}")
    print(f"  Valid checkpoint paths: {len(valid_checkpoints)}")
    print(f"  Invalid checkpoint paths: {len(invalid_checkpoints)}")
    print("="*80)
    
    if invalid_checkpoints:
        print("\nInference files with missing checkpoints:")
        for zarr_name, ckpt_path, _ in invalid_checkpoints:
            print(f"  - {zarr_name}")
            print(f"    Missing: {ckpt_path}")
    
    return {
        'valid': valid_checkpoints,
        'invalid': invalid_checkpoints
    }


def find_unreferenced_checkpoints(
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    inference_dir=f"{PROJECT_ROOT}/data/inference"
):
    """
    Find checkpoint directories that are NOT referenced by any inference dataset.
    
    This function scans all checkpoint directories and checks if each checkpoint
    is referenced by at least one inference dataset's .zattrs file. Checkpoints
    without any references are considered orphaned.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        inference_dir: Path to the inference directory (default: "data/inference")
    
    Returns:
        dict: Results with 'referenced' and 'unreferenced' lists of checkpoint paths
    
    Example:
        find_unreferenced_checkpoints()
    """
    checkpoints_path = Path(checkpoints_dir)
    inference_path = Path(inference_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return None
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return None
    
    # Get all checkpoint subdirectories with final.pth
    checkpoint_dirs = []
    for subdir in checkpoints_path.iterdir():
        if subdir.is_dir() and (subdir / "final.pth").exists():
            checkpoint_dirs.append(subdir)
    checkpoint_dirs.sort(key=lambda x: x.name)
    
    if not checkpoint_dirs:
        print("No checkpoint directories with final.pth found.")
        return None
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories with final.pth")
    
    # Get all inference datasets and their checkpoint references
    zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    
    if not zarr_dirs:
        print(f"Found 0 inference datasets in {inference_dir}")
        print(f"\nAll {len(checkpoint_dirs)} checkpoints are unreferenced!")
        return {
            'referenced': [],
            'unreferenced': [(str(ckpt), []) for ckpt in checkpoint_dirs]
        }
    
    print(f"Found {len(zarr_dirs)} inference datasets\n")
    
    # Build a map of checkpoint paths to datasets that reference them
    checkpoint_references = {}  # checkpoint_path -> list of dataset names
    
    for zarr_dir in zarr_dirs:
        zattrs_file = zarr_dir / ".zattrs"
        
        if not zattrs_file.exists():
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                attrs = json.load(f)
            
            if 'checkpoint_path' in attrs:
                checkpoint_path = Path(attrs['checkpoint_path']).resolve()
                
                if checkpoint_path not in checkpoint_references:
                    checkpoint_references[checkpoint_path] = []
                checkpoint_references[checkpoint_path].append(zarr_dir.name)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠ Warning: Could not read {zarr_dir.name}/.zattrs: {e}")
            continue
    
    # Check each checkpoint directory
    referenced = []
    unreferenced = []
    
    for ckpt_dir in checkpoint_dirs:
        ckpt_path = ckpt_dir.resolve()
        
        if ckpt_path in checkpoint_references:
            datasets = checkpoint_references[ckpt_path]
            referenced.append((str(ckpt_dir), datasets))
            print(f"✓ {ckpt_dir.name}")
            print(f"  Referenced by {len(datasets)} dataset(s):")
            for ds in datasets:
                print(f"    - {ds}")
        else:
            unreferenced.append((str(ckpt_dir), []))
            print(f"✗ {ckpt_dir.name}")
            print(f"  NOT referenced by any dataset")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Total checkpoints: {len(checkpoint_dirs)}")
    print(f"  Referenced checkpoints: {len(referenced)}")
    print(f"  Unreferenced checkpoints: {len(unreferenced)}")
    print("="*80)
    
    if unreferenced:
        print(f"\nUnreferenced checkpoints ({len(unreferenced)}):")
        for ckpt_path, _ in unreferenced:
            ckpt_name = Path(ckpt_path).name
            print(f"  - {ckpt_name}")
    
    return {
        'referenced': referenced,
        'unreferenced': unreferenced
    }


def rename_inference_datasets(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    rename_patterns=None,
    dry_run=True
):
    """
    Rename inference dataset directories and update their internal .zattrs files.
    
    This function applies string replacement patterns to inference dataset folder names
    and updates the 'inference_dataset_path' field in each dataset's .zattrs file to match.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        rename_patterns: Dict of string replacement patterns {old_str: new_str}.
                        If None, uses default patterns:
                        {'sydney1': 's1', 'sydney2': 's2'}
        dry_run: If True, only report what would be renamed without actually renaming (default: True)
    
    Returns:
        int: number of datasets renamed
    
    Example:
        rename_inference_datasets(
            rename_patterns={'sydney1': 's1', 'sydney2': 's2'},
            dry_run=False
        )
    """
    if rename_patterns is None:
        rename_patterns={
            'sydney1': 'b08d56', 
            'sydney2': 'bc54d6',
            '+VT': '+ID',
            '+SP': '+SaP',
            '+GP': '+SiP'
        }
    
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    zarr_dirs.sort()
    
    if not zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    print(f"Found {len(zarr_dirs)} .zarr directories to check.")
    print(f"Rename patterns: {rename_patterns}\n")
    
    renames = []
    
    for zarr_dir in zarr_dirs:
        old_name = zarr_dir.name
        new_name = old_name
        
        # Apply all rename patterns
        for old_str, new_str in rename_patterns.items():
            new_name = new_name.replace(old_str, new_str)
        
        # If the name changed, add to renames list
        if new_name != old_name:
            new_path = zarr_dir.parent / new_name
            
            # Check .zattrs file
            zattrs_file = zarr_dir / ".zattrs"
            if not zattrs_file.exists():
                print(f"Warning: {old_name} has no .zattrs file, skipping")
                continue
            
            try:
                import json
                with open(zattrs_file, 'r') as f:
                    zattrs = json.load(f)
                
                # Check if inference_dataset_path exists
                if 'inference_dataset_path' not in zattrs:
                    print(f"Warning: {old_name} has no 'inference_dataset_path' in .zattrs, skipping")
                    continue
                
                old_dataset_path = zattrs['inference_dataset_path']
                
                # Reconstruct the correct path using the new directory name
                new_dataset_path = f"{inference_dir}/{new_name}"
                
                renames.append({
                    'old_dir': zarr_dir,
                    'new_dir': new_path,
                    'old_name': old_name,
                    'new_name': new_name,
                    'zattrs_file': zattrs_file,
                    'zattrs': zattrs,
                    'old_dataset_path': old_dataset_path,
                    'new_dataset_path': new_dataset_path
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse .zattrs for {old_name}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing {old_name}: {e}")
                continue
    
    if not renames:
        print("\nNo datasets need to be renamed.")
        return 0
    
    print(f"\nFound {len(renames)} datasets to rename:")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        for item in renames:
            print(f"\nWould rename:")
            print(f"  Directory:  {item['old_name']}")
            print(f"           → {item['new_name']}")
            print(f"  .zattrs['inference_dataset_path']:")
            print(f"    {item['old_dataset_path']}")
            print(f"  → {item['new_dataset_path']}")
        print("\nTo actually rename these datasets, run with dry_run=False")
        return 0
    else:
        # Actually perform the renames
        num_renamed = 0
        for item in renames:
            try:
                # Update .zattrs file
                item['zattrs']['inference_dataset_path'] = item['new_dataset_path']
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                # Rename directory
                item['old_dir'].rename(item['new_dir'])
                
                print(f"✓ Renamed: {item['old_name']} → {item['new_name']}")
                num_renamed += 1
                
            except Exception as e:
                print(f"✗ Error renaming {item['old_name']}: {e}")
        
        print(f"\nSuccessfully renamed {num_renamed} datasets.")
        return num_renamed


def add_inference_zattrs_parameter(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    param_to_add=None,
    value=None,
    dry_run=True
):
    """
    Add a parameter to .zattrs files in inference datasets if it doesn't already exist.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to update (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        param_to_add: The parameter name to add to .zattrs
        value: The value to set for the parameter
        dry_run: If True, only report what would be changed without actually updating (default: True)
    
    Returns:
        int: number of datasets modified
    
    Example:
        add_inference_zattrs_parameter(
            pattern='WaitAtGoal',
            param_to_add='new_field',
            value='some_value',
            dry_run=False
        )
    """
    if param_to_add is None:
        print("Error: Must provide param_to_add")
        return 0
    
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Searching for {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Searching for datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print("No filter specified, will process all datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to process.")
        return 0
    
    print(f"\nFound {len(datasets_to_process)} dataset(s) to check.")
    print(f"Parameter to add: '{param_to_add}' = {value}\n")
    
    datasets_to_modify = []
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Warning: {dataset_dir.name} has no .zattrs file, skipping")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Check if parameter already exists
            if param_to_add in zattrs:
                existing_value = zattrs[param_to_add]
                print(f"○ {dataset_dir.name}: parameter already exists (value: {existing_value})")
            else:
                # Add the parameter
                zattrs[param_to_add] = value
                datasets_to_modify.append({
                    'name': dataset_dir.name,
                    'zattrs_file': zattrs_file,
                    'zattrs': zattrs
                })
                print(f"✓ {dataset_dir.name}: will add parameter")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse .zattrs for {dataset_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {dataset_dir.name}: {e}")
            continue
    
    if not datasets_to_modify:
        print(f"\nNo datasets need modification (all already have '{param_to_add}' or were skipped).")
        return 0
    
    print(f"\nFound {len(datasets_to_modify)} dataset(s) to modify.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        print("To actually apply these changes, run with dry_run=False")
        return 0
    else:
        # Actually update the files
        num_modified = 0
        for item in datasets_to_modify:
            try:
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                print(f"✓ Modified: {item['name']}")
                num_modified += 1
                
            except Exception as e:
                print(f"✗ Error updating {item['name']}: {e}")
        
        print(f"\nSuccessfully modified {num_modified} datasets.")
        return num_modified


def remove_inference_zattrs_parameter(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    param_to_remove=None,
    use_wildcard=False,
    dry_run=True
):
    """
    Remove a parameter from .zattrs files in inference datasets.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to update (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        param_to_remove: The parameter name to remove from .zattrs. Can use wildcard matching
                        if use_wildcard=True (e.g., 'checkpoint_train_cfg.*' removes all
                        parameters starting with 'checkpoint_train_cfg.')
        use_wildcard: If True, param_to_remove is treated as a wildcard pattern supporting:
                     - '*' at the end for prefix matching (e.g., 'checkpoint_*')
                     - '*' at the beginning for suffix matching (e.g., '*_alpha')
                     - '*' in the middle for contains matching (e.g., 'checkpoint_*_alpha')
        dry_run: If True, only report what would be changed without actually updating (default: True)
    
    Returns:
        int: number of datasets modified
    
    Example:
        # Remove exact parameter name
        remove_inference_zattrs_parameter(
            pattern='revalidation',
            param_to_remove='old_field',
            dry_run=False
        )
        
        # Remove all parameters starting with a prefix
        remove_inference_zattrs_parameter(
            param_to_remove='checkpoint_train_cfg.*',
            use_wildcard=True,
            dry_run=False
        )
    """
    if param_to_remove is None:
        print("Error: Must provide param_to_remove")
        return 0
    
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Searching for {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Searching for datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print("No filter specified, will process all datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to process.")
        return 0
    
    print(f"\nFound {len(datasets_to_process)} dataset(s) to check.")
    if use_wildcard:
        print(f"Parameter pattern to remove (wildcard): '{param_to_remove}'\n")
    else:
        print(f"Parameter to remove: '{param_to_remove}'\n")
    
    datasets_to_modify = []
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Warning: {dataset_dir.name} has no .zattrs file, skipping")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Find parameters to remove
            params_to_remove = []
            
            if use_wildcard:
                # Handle wildcard matching
                import fnmatch
                for key in zattrs.keys():
                    if fnmatch.fnmatch(key, param_to_remove):
                        params_to_remove.append(key)
            else:
                # Exact match
                if param_to_remove in zattrs:
                    params_to_remove.append(param_to_remove)
            
            if params_to_remove:
                removed_params = {}
                for param in params_to_remove:
                    removed_params[param] = zattrs.pop(param)
                
                datasets_to_modify.append({
                    'name': dataset_dir.name,
                    'zattrs_file': zattrs_file,
                    'zattrs': zattrs,
                    'removed_params': removed_params
                })
                
                if len(params_to_remove) == 1:
                    print(f"✓ {dataset_dir.name}: found parameter '{params_to_remove[0]}' (value: {removed_params[params_to_remove[0]]})")
                else:
                    print(f"✓ {dataset_dir.name}: found {len(params_to_remove)} parameters:")
                    for param in params_to_remove:
                        print(f"    - {param} (value: {removed_params[param]})")
            else:
                if use_wildcard:
                    print(f"✗ {dataset_dir.name}: no parameters matching pattern")
                else:
                    print(f"✗ {dataset_dir.name}: parameter not found")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse .zattrs for {dataset_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {dataset_dir.name}: {e}")
            continue
    
    if not datasets_to_modify:
        if use_wildcard:
            print(f"\nNo datasets contain parameters matching pattern '{param_to_remove}'.")
        else:
            print(f"\nNo datasets contain the parameter '{param_to_remove}'.")
        return 0
    
    if use_wildcard:
        total_params = sum(len(item['removed_params']) for item in datasets_to_modify)
        print(f"\nFound {total_params} total parameter(s) matching pattern in {len(datasets_to_modify)} dataset(s).")
    else:
        print(f"\nFound parameter '{param_to_remove}' in {len(datasets_to_modify)} dataset(s).")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        if use_wildcard:
            print("To actually remove these parameters, run with dry_run=False")
        else:
            print("To actually remove this parameter, run with dry_run=False")
        return 0
    else:
        # Actually update the files
        num_modified = 0
        for item in datasets_to_modify:
            try:
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                print(f"✓ Modified: {item['name']}")
                num_modified += 1
                
            except Exception as e:
                print(f"✗ Error updating {item['name']}: {e}")
        
        print(f"\nSuccessfully modified {num_modified} datasets.")
        return num_modified




def update_inference_zattrs_parameter(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    old_param=None,
    new_param=None,
    new_value=None,
    dry_run=True
):
    """
    Rename a parameter in .zattrs files in inference datasets and optionally update its value.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to update (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        old_param: The old parameter name to rename
        new_param: The new parameter name
        new_value: The new value to set. If None, keeps the old value (default: None)
        dry_run: If True, only report what would be changed without actually updating (default: True)
    
    Returns:
        int: number of datasets modified
    
    Example:
        # Rename parameter and keep its value
        rename_inference_zattrs_parameter(
            pattern='sydney1',
            old_param='saturating_progress_alpha',
            new_param='saturating_progress_encoding_omega',
            dry_run=False
        )
        
        # Rename parameter and update its value
        rename_inference_zattrs_parameter(
            old_param='old_field',
            new_param='new_field',
            new_value=42,
            dry_run=False
        )
    """
    if old_param is None or new_param is None:
        print("Error: Must provide both old_param and new_param")
        return 0
    
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Searching for {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Searching for datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print("No filter specified, will process all datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to process.")
        return 0
    
    print(f"\nFound {len(datasets_to_process)} dataset(s) to check.")
    print(f"Rename: '{old_param}' → '{new_param}'")
    if new_value is not None:
        print(f"New value: {new_value}")
    else:
        print(f"Keep existing value")
    print()
    
    datasets_to_modify = []
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Warning: {dataset_dir.name} has no .zattrs file, skipping")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Check if old parameter exists
            if old_param not in zattrs:
                print(f"✗ {dataset_dir.name}: parameter '{old_param}' not found")
            elif new_param in zattrs and new_param != old_param:
                print(f"⚠ {dataset_dir.name}: parameter '{new_param}' already exists, skipping")
            else:
                old_value = zattrs.pop(old_param)
                value_to_set = new_value if new_value is not None else old_value
                zattrs[new_param] = value_to_set
                
                datasets_to_modify.append({
                    'name': dataset_dir.name,
                    'zattrs_file': zattrs_file,
                    'zattrs': zattrs,
                    'old_value': old_value,
                    'new_value': value_to_set
                })
                
                if new_value is not None and old_value != new_value:
                    print(f"✓ {dataset_dir.name}: rename and update value ({old_value} → {value_to_set})")
                else:
                    print(f"✓ {dataset_dir.name}: rename, keep value ({old_value})")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse .zattrs for {dataset_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {dataset_dir.name}: {e}")
            continue
    
    if not datasets_to_modify:
        print(f"\nNo datasets need modification.")
        return 0
    
    print(f"\nFound {len(datasets_to_modify)} dataset(s) to modify.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        print("To actually apply these changes, run with dry_run=False")
        return 0
    else:
        # Actually update the files
        num_modified = 0
        for item in datasets_to_modify:
            try:
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                print(f"✓ Modified: {item['name']}")
                num_modified += 1
                
            except Exception as e:
                print(f"✗ Error updating {item['name']}: {e}")
        
        print(f"\nSuccessfully modified {num_modified} datasets.")
        return num_modified


def missing_config_files(checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints", dry_run=True, operation: Literal['generate', 'delete'] = 'generate'):
    """
    Generate or delete config.txt files in checkpoint directories.
    
    This function checks each checkpoint subdirectory and either:
    - 'generate': Creates config.txt files from final.pth for directories that don't have one
    - 'delete': Removes existing config_from_pytorch_ckpt.txt files
    
    Args:
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        dry_run: If True, only report what would be created/deleted without actually doing it (default: True)
        operation: Specify whether to 'generate' missing config files or 'delete' existing ones (default: 'generate')
    Returns:
        int: number of config files created or deleted
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0
    
    # Get all subdirectories
    subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: x.name)
    
    if not subdirs:
        print("No checkpoint subdirectories found.")
        return 0
    
    if operation == 'delete':
        print(f"Checking {len(subdirs)} checkpoint subdirectories for config files to delete...\n")
        
        files_to_delete = []
        
        for subdir in subdirs:
            config_file = subdir / "final.pth.readonly_dump.txt"
            
            if config_file.exists():
                files_to_delete.append((subdir, config_file))
                print(f"✓ {subdir.name}: found {config_file.name}")
        
        if not files_to_delete:
            print(f"\nNo config files found to delete.")
            return 0
        
        print(f"\nFound {len(files_to_delete)} config files to delete")
        
        if dry_run:
            print("\n--- DRY RUN MODE (no files will be deleted) ---")
            for subdir, config_file in files_to_delete:
                print(f"Would delete: {subdir.name}/{config_file.name}")
            print("\nTo actually delete these files, run with dry_run=False")
            return 0
        else:
            # Delete the config files
            num_deleted = 0
            for subdir, config_file in files_to_delete:
                try:
                    config_file.unlink()
                    print(f"Deleted: {subdir.name}/{config_file.name}")
                    num_deleted += 1
                except Exception as e:
                    print(f"Error deleting {subdir.name}/{config_file.name}: {e}")
            
            print(f"\nSuccessfully deleted {num_deleted} config files.")
            return num_deleted
    
    else:  # operation == 'generate'
        print(f"Checking {len(subdirs)} checkpoint subdirectories for missing config.txt files...\n")
        
        missing_configs = []
        
        for subdir in subdirs:
            config_file = subdir / "final.pth.readonly_dump.txt"
            final_pth = subdir / "final.pth"
        
            # Check if final.pth exists
            if not final_pth.exists():
                print(f"⚠ {subdir.name}: no final.pth found, skipping")
                continue
            
            try:
                # Load the checkpoint
                checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
                
                # Check if this is a dict-like checkpoint
                if not isinstance(checkpoint, dict):
                    print(f"⚠ {subdir.name}: checkpoint is not a dictionary, skipping")
                    continue
                
                # Check if train_cfg exists
                if 'train_cfg' not in checkpoint:
                    print(f"⚠ {subdir.name}: no 'train_cfg' in checkpoint, skipping")
                    continue
                
                train_cfg = checkpoint['train_cfg']
                
                if not isinstance(train_cfg, dict):
                    print(f"⚠ {subdir.name}: 'train_cfg' is not a dictionary, skipping")
                    continue
                
                missing_configs.append((subdir, config_file, train_cfg))
                print(f"✓ {subdir.name}: will create {config_file.name}")
                
            except Exception as e:
                print(f"⚠ {subdir.name}: error loading checkpoint: {e}")
                continue
        
        if not missing_configs:
            print(f"\nNo missing config.txt files found (all checkpoints have configs or are invalid).")
            return 0
        
        print(f"\nFound {len(missing_configs)} checkpoint directories missing config.txt")
        
        if dry_run:
            print("\n--- DRY RUN MODE (no files will be created) ---")
            for subdir, config_file, _ in missing_configs:
                print(f"Would create: {subdir.name}/{config_file.name}")
            print("\nTo actually create these files, run with dry_run=False")
            return 0
        else:
            # Create the config files
            num_created = 0
            for subdir, config_file, train_cfg in missing_configs:
                try:
                    with open(config_file, 'w') as f:
                        # Write each key-value pair
                        for key in sorted(train_cfg.keys()):
                            value = train_cfg[key]
                            f.write(f"{key}: {value}\n")
                    
                    print(f"Created: {subdir.name}/{config_file.name}")
                    num_created += 1
                except Exception as e:
                    print(f"Error creating {subdir.name}/{config_file.name}: {e}")
            
            print(f"\nSuccessfully created {num_created} config files.")
            return num_created


def sort_inference_zattrs(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    dry_run=True
):
    """
    Sort all parameters in .zattrs files alphabetically and rewrite them.
    
    This ensures consistent formatting across all .zattrs files by sorting
    the keys alphabetically.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to sort (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        dry_run: If True, only report what would be changed without actually updating (default: True)
    
    Returns:
        int: number of datasets sorted and rewritten
    
    Example:
        # Sort all datasets
        sort_inference_zattrs(dry_run=False)
        
        # Sort only specific datasets
        sort_inference_zattrs(
            pattern='WaitAtGoal',
            dry_run=False
        )
    """
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Searching for {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Searching for datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print("No filter specified, will process all datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to process.")
        return 0
    
    print(f"\nFound {len(datasets_to_process)} dataset(s) to sort.\n")
    
    datasets_to_sort = []
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Warning: {dataset_dir.name} has no .zattrs file, skipping")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Check if already sorted
            keys = list(zattrs.keys())
            sorted_keys = sorted(keys)
            
            if keys == sorted_keys:
                print(f"○ {dataset_dir.name}: already sorted")
            else:
                # Create sorted version
                sorted_zattrs = {k: zattrs[k] for k in sorted_keys}
                datasets_to_sort.append({
                    'name': dataset_dir.name,
                    'zattrs_file': zattrs_file,
                    'zattrs': sorted_zattrs,
                    'original_order': keys,
                    'sorted_order': sorted_keys
                })
                print(f"✓ {dataset_dir.name}: will sort")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse .zattrs for {dataset_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {dataset_dir.name}: {e}")
            continue
    
    if not datasets_to_sort:
        print(f"\nAll datasets are already sorted alphabetically.")
        return 0
    
    print(f"\nFound {len(datasets_to_sort)} dataset(s) to sort and rewrite.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        print("To actually sort and rewrite these files, run with dry_run=False")
        return 0
    else:
        # Actually update the files
        num_sorted = 0
        for item in datasets_to_sort:
            try:
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                print(f"✓ Sorted: {item['name']}")
                num_sorted += 1
                
            except Exception as e:
                print(f"✗ Error updating {item['name']}: {e}")
        
        print(f"\nSuccessfully sorted and rewrote {num_sorted} .zattrs files.")
        return num_sorted


def compare_inference_zattrs(
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    do_dataset_specific_analysis=False
):
    """
    Compare keys in .zattrs files across all inference datasets to find differences.
    
    This function reads .zattrs from each inference dataset and compares the keys
    to identify which keys are present in some datasets but not others, and shows
    value differences.
    
    Args:
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to compare (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        do_dataset_specific_analysis: Whether to perform dataset-specific analysis (default: False)
    
    Returns:
        dict: A dictionary with analysis results including:
            - 'all_keys': set of all keys found across all datasets
            - 'common_keys': set of keys present in all datasets
            - 'key_presence': dict mapping each key to list of datasets containing it
            - 'dataset_keys': dict mapping each dataset to its set of keys
    
    Example:
        # Compare all datasets
        compare_inference_zattrs()
        
        # Compare only WaitAtGoal datasets
        compare_inference_zattrs(pattern='WaitAtGoal')
    """
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return None
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return None
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Comparing {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Comparing datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print(f"Comparing all {len(all_zarr_dirs)} datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to compare.")
        return None
    
    print(f"Analyzing {len(datasets_to_process)} inference datasets...\n")
    
    dataset_keys = {}
    dataset_values = {}  # Store the actual values for each dataset
    all_keys = set()
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Skipping {dataset_dir.name}: no .zattrs file found")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Get the keys and values
            keys = set(zattrs.keys())
            dataset_keys[dataset_dir.name] = keys
            dataset_values[dataset_dir.name] = zattrs
            all_keys.update(keys)
            
        except json.JSONDecodeError as e:
            print(f"Skipping {dataset_dir.name}: error parsing .zattrs JSON: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_dir.name}: {e}")
            continue
    
    if not dataset_keys:
        print("\nNo valid datasets with .zattrs files found.")
        return None
    
    print(f"\nSuccessfully loaded {len(dataset_keys)} datasets.\n")
    
    # Find common keys (present in all datasets)
    common_keys = set.intersection(*dataset_keys.values()) if dataset_keys else set()
    
    # Build key presence map
    key_presence = {}
    for key in sorted(all_keys):
        datasets_with_key = [ds for ds, keys in dataset_keys.items() if key in keys]
        key_presence[key] = datasets_with_key
    
    # Print results
    print("="*80)
    print(f"SUMMARY:")
    print(f"  Total unique keys found: {len(all_keys)}")
    print(f"  Keys present in all datasets: {len(common_keys)}")
    print(f"  Keys present in some datasets: {len(all_keys - common_keys)}")
    print("="*80)
    
    # Print common keys
    if common_keys:
        print(f"\nKeys present in ALL {len(dataset_keys)} datasets:")
        for key in sorted(common_keys):
            # Get all values for this key across datasets
            values = set()
            for ds_name in dataset_keys.keys():
                val = dataset_values[ds_name].get(key)
                # Convert to string for comparison, handle different types
                values.add(str(val))
            
            # If all values are the same, show once; otherwise show "VARIES"
            if len(values) == 1:
                print(f"  ✓ {key} [{values.pop()}]")
            else:
                # Limit the display if there are too many unique values
                if len(values) <= 5:
                    print(f"  ✓ {key} [VARIES: {', '.join(sorted(values))}]")
                else:
                    print(f"  ✓ {key} [VARIES: {len(values)} unique values]")
    
    # Print keys that differ
    differing_keys = all_keys - common_keys
    if differing_keys:
        print(f"\nKeys present in SOME (not all) datasets:")
        for key in sorted(differing_keys):
            datasets_with_key = key_presence[key]
            count = len(datasets_with_key)
            
            # Get all values for this key
            values = set()
            for ds_name in datasets_with_key:
                val = dataset_values[ds_name].get(key)
                values.add(str(val))
            
            if len(values) == 1:
                print(f"\n  • {key} [{values.pop()}]")
            else:
                if len(values) <= 5:
                    print(f"\n  • {key} [VARIES: {', '.join(sorted(values))}]")
                else:
                    print(f"\n  • {key} [VARIES: {len(values)} unique values]")
            print(f"    Present in {count}/{len(dataset_keys)} datasets")
            
            # Show first few examples if not too many
            if count <= 10:
                for ds in datasets_with_key:
                    val = dataset_values[ds].get(key)
                    print(f"      - {ds} [{val}]")
            else:
                print(f"      (showing first 10 of {count})")
                for ds in datasets_with_key[:10]:
                    val = dataset_values[ds].get(key)
                    print(f"      - {ds} [{val}]")
    
    if do_dataset_specific_analysis:
        # Find datasets that have unique key sets
        print(f"\n{'='*80}")
        print("DATASET-SPECIFIC ANALYSIS:")
        print("="*80)
        
        for ds_name in sorted(dataset_keys.keys()):
            keys = dataset_keys[ds_name]
            unique_keys = keys - common_keys
            missing_keys = common_keys - keys
            
            if unique_keys or missing_keys:
                print(f"\n{ds_name}:")
                if unique_keys:
                    print(f"  Extra keys (not in all):")
                    for key in sorted(unique_keys):
                        val = dataset_values[ds_name].get(key)
                        print(f"    + {key} [{val}]")
                if missing_keys:
                    print(f"  Missing keys (that others have):")
                    for key in sorted(missing_keys):
                        print(f"    - {key}")
    
    return {
        'all_keys': all_keys,
        'common_keys': common_keys,
        'key_presence': key_presence,
        'dataset_keys': dataset_keys
    }


def conditional_update_checkpoint_parameter(
    conditions,
    updates,
    checkpoint_subdirs=None,
    checkpoints_dir=f"{PROJECT_ROOT}/data/checkpoints",
    subdict_key="train_cfg",
    dry_run=True
):
    """
    Conditionally update parameters in checkpoint files based on specified conditions.
    
    This function checks if all specified conditions are met in each checkpoint's
    subdictionary, and if so, applies the specified updates.
    
    Args:
        conditions: Dict of parameter conditions to check {param_name: expected_value}.
                   Use None to check if a parameter is missing/null.
        updates: Dict of parameter updates to apply if conditions are met {param_name: new_value}.
        checkpoint_subdirs: List of subdirectory names to process (e.g., ["20260122_071500"]). 
                           If None, processes all subdirectories in checkpoints_dir.
        checkpoints_dir: Path to the checkpoints directory (default: "data/checkpoints")
        subdict_key: The key of the subdictionary to check/modify (default: "train_cfg")
        dry_run: If True, only report what would be changed without modifying files (default: True)
    
    Returns:
        int: number of checkpoint files modified
    
    Example:
        conditions = {
            'saturating_progress_alpha': 0.047,
            'use_idleness_encoding': True,
            'idleness_encoding_alpha': None  # checks if missing or None
        }
        updates = {
            'idleness_encoding_alpha': 0.05
        }
        conditional_update_checkpoint_parameter(conditions, updates, dry_run=False)
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return 0
    
    # Get subdirectories to process
    if checkpoint_subdirs is None:
        # Get all subdirectories
        subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
    else:
        # Use specified subdirectories
        subdirs = [checkpoints_path / subdir for subdir in checkpoint_subdirs]
        # Filter out non-existent directories
        subdirs = [d for d in subdirs if d.exists() and d.is_dir()]
    
    if not subdirs:
        print("No checkpoint subdirectories found to process.")
        return 0
    
    print(f"Found {len(subdirs)} checkpoint subdirectories to check.")
    print(f"Looking in subdictionary: '{subdict_key}'")
    print(f"\nConditions to check:")
    for param, value in conditions.items():
        if value is None:
            print(f"  - {param} is None/missing")
        else:
            print(f"  - {param} == {value}")
    print(f"\nUpdates to apply if conditions met:")
    for param, value in updates.items():
        print(f"  - {param} = {value}")
    print()
    
    # sort by name for consistent output
    subdirs.sort(key=lambda x: x.name)

    modified_files = []
    
    for subdir in subdirs:
        final_pth = subdir / "final.pth"
        
        if not final_pth.exists():
            continue
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(final_pth, map_location='cpu', weights_only=False)
            
            # Check if this is a dict-like checkpoint
            if not isinstance(checkpoint, dict):
                print(f"Skipping {subdir.name}/final.pth: not a dictionary")
                continue
            
            # Check if the subdictionary exists
            if subdict_key not in checkpoint:
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' not found")
                continue
            
            # Check if the subdictionary is a dict
            if not isinstance(checkpoint[subdict_key], dict):
                print(f"Skipping {subdir.name}/final.pth: '{subdict_key}' is not a dictionary")
                continue
            
            # Check all conditions
            all_conditions_met = True
            condition_details = []
            
            for param_name, expected_value in conditions.items():
                actual_value = checkpoint[subdict_key].get(param_name)
                
                if expected_value is None:
                    # Check if parameter is missing or None
                    condition_met = actual_value is None
                    condition_details.append(f"{param_name} is {'None/missing' if condition_met else repr(actual_value)}")
                else:
                    # Check if parameter matches expected value
                    condition_met = actual_value == expected_value
                    condition_details.append(f"{param_name} == {expected_value} (actual: {actual_value})")
                
                if not condition_met:
                    all_conditions_met = False
            
            if all_conditions_met:
                # Apply updates
                for param_name, new_value in updates.items():
                    old_value = checkpoint[subdict_key].get(param_name)
                    checkpoint[subdict_key][param_name] = new_value
                
                modified_files.append((final_pth, checkpoint))
                print(f"✓ {subdir.name}/final.pth")
                print(f"  Conditions met: {', '.join(condition_details)}")
                print(f"  Will apply updates: {updates}")
            else:
                print(f"○ {subdir.name}/final.pth")
                print(f"  Conditions NOT met: {', '.join(condition_details)}")

        except Exception as e:
            print(f"Error processing {subdir.name}/final.pth: {e}")
            continue
    
    if not modified_files:
        print(f"\nNo files met the specified conditions.")
        return 0
    
    print(f"\nFound {len(modified_files)} files meeting conditions.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no files will be modified) ---")
        for file_path, _ in modified_files:
            print(f"Would modify: {file_path.parent.name}/final.pth")
        print("\nTo actually modify these files, run with dry_run=False")
        return 0
    else:
        # Save the modified checkpoints
        num_modified = 0
        for file_path, modified_checkpoint in modified_files:
            try:
                torch.save(modified_checkpoint, file_path)
                print(f"Modified: {file_path.parent.name}/final.pth")
                num_modified += 1
            except Exception as e:
                print(f"Error saving {file_path.parent.name}/final.pth: {e}")
        
        print(f"\nSuccessfully modified {num_modified} checkpoint files.")
        return num_modified


def conditional_update_inference_zattrs(
    conditions,
    updates,
    inference_dir=f"{PROJECT_ROOT}/data/inference",
    dataset_names=None,
    pattern=None,
    dry_run=True
):
    """
    Conditionally update parameters in .zattrs files based on specified conditions.
    
    This function checks if all specified conditions are met in each dataset's
    .zattrs file, and if so, applies the specified updates.
    
    Args:
        conditions: Dict of parameter conditions to check {param_name: expected_value}.
                   Use None to check if a parameter is missing/null.
        updates: Dict of parameter updates to apply if conditions are met {param_name: new_value}.
        inference_dir: Path to the inference directory (default: "data/inference")
        dataset_names: List of exact dataset names to process (including .zarr extension).
                      If None, uses pattern matching or all datasets.
        pattern: String pattern to match in dataset names (case-insensitive).
                Only used if dataset_names is None.
        dry_run: If True, only report what would be changed without modifying files (default: True)
    
    Returns:
        int: number of datasets modified
    
    Example:
        # Add missing parameter only where certain conditions are met
        conditions = {
            'use_idleness_encoding': True,
            'idleness_encoding_alpha': None  # checks if missing or None
        }
        updates = {
            'idleness_encoding_alpha': 0.047
        }
        conditional_update_inference_zattrs(conditions, updates, dry_run=False)
        
        # Update parameter based on another parameter's value
        conditions = {
            'host': 'sydney1'
        }
        updates = {
            'host': 'b08d56'
        }
        conditional_update_inference_zattrs(conditions, updates, pattern='WaitAtGoal', dry_run=False)
    """
    inference_path = Path(inference_dir)
    
    if not inference_path.exists():
        print(f"Inference directory '{inference_dir}' does not exist.")
        return 0
    
    # Get all .zarr directories
    all_zarr_dirs = [d for d in inference_path.iterdir() if d.is_dir() and d.name.endswith('.zarr')]
    all_zarr_dirs.sort()
    
    if not all_zarr_dirs:
        print("No .zarr directories found in inference directory.")
        return 0
    
    # Filter datasets
    datasets_to_process = []
    
    if dataset_names is not None:
        for name in dataset_names:
            matching = [d for d in all_zarr_dirs if d.name == name]
            datasets_to_process.extend(matching)
        print(f"Searching for {len(dataset_names)} specific dataset(s)...")
    elif pattern is not None:
        pattern_lower = pattern.lower()
        datasets_to_process = [d for d in all_zarr_dirs if pattern_lower in d.name.lower()]
        print(f"Searching for datasets matching pattern '{pattern}'...")
    else:
        datasets_to_process = all_zarr_dirs
        print("No filter specified, will process all datasets...")
    
    if not datasets_to_process:
        print("\nNo datasets found to process.")
        return 0
    
    print(f"\nFound {len(datasets_to_process)} dataset(s) to check.")
    print(f"\nConditions to check:")
    for param, value in conditions.items():
        if value is None:
            print(f"  - {param} is None/missing")
        else:
            print(f"  - {param} == {value}")
    print(f"\nUpdates to apply if conditions met:")
    for param, value in updates.items():
        print(f"  - {param} = {value}")
    print()
    
    datasets_to_modify = []
    
    for dataset_dir in datasets_to_process:
        zattrs_file = dataset_dir / ".zattrs"
        
        if not zattrs_file.exists():
            print(f"Warning: {dataset_dir.name} has no .zattrs file, skipping")
            continue
        
        try:
            with open(zattrs_file, 'r') as f:
                zattrs = json.load(f)
            
            # Check all conditions
            all_conditions_met = True
            condition_details = []
            
            for param_name, expected_value in conditions.items():
                actual_value = zattrs.get(param_name)
                
                if expected_value is None:
                    # Check if parameter is missing or None
                    condition_met = actual_value is None
                    condition_details.append(f"{param_name} is {'None/missing' if condition_met else repr(actual_value)}")
                else:
                    # Check if parameter matches expected value
                    condition_met = actual_value == expected_value
                    condition_details.append(f"{param_name} == {expected_value} (actual: {actual_value})")
                
                if not condition_met:
                    all_conditions_met = False
            
            if all_conditions_met:
                # Apply updates
                for param_name, new_value in updates.items():
                    old_value = zattrs.get(param_name)
                    zattrs[param_name] = new_value
                
                datasets_to_modify.append({
                    'name': dataset_dir.name,
                    'zattrs_file': zattrs_file,
                    'zattrs': zattrs,
                    'condition_details': condition_details,
                    'updates': updates
                })
                print(f"✓ {dataset_dir.name}")
                print(f"  Conditions met: {', '.join(condition_details)}")
            else:
                print(f"○ {dataset_dir.name}")
                print(f"  Conditions NOT met: {', '.join(condition_details)}")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse .zattrs for {dataset_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {dataset_dir.name}: {e}")
            continue
    
    if not datasets_to_modify:
        print(f"\nNo datasets met the specified conditions.")
        return 0
    
    print(f"\nFound {len(datasets_to_modify)} dataset(s) meeting conditions.")
    
    if dry_run:
        print("\n--- DRY RUN MODE (no changes will be made) ---")
        for item in datasets_to_modify:
            print(f"Would modify: {item['name']}")
        print("\nTo actually modify these files, run with dry_run=False")
        return 0
    else:
        # Actually update the files
        num_modified = 0
        for item in datasets_to_modify:
            try:
                with open(item['zattrs_file'], 'w') as f:
                    json.dump(item['zattrs'], f, indent=4)
                
                print(f"✓ Modified: {item['name']}")
                num_modified += 1
                
            except Exception as e:
                print(f"✗ Error updating {item['name']}: {e}")
        
        print(f"\nSuccessfully modified {num_modified} datasets.")
        return num_modified


if __name__ == "__main__":

    # Run with dry_run=True by default to preview what would be deleted

    #########################
    ### CHECKPOINT FILES ####
    #########################
    find_unreferenced_checkpoints()

    if do_key_comparison := False:
        compare_checkpoint_keys()
    
    if do_cleanup_checkpoint_files := False:
        cleanup_checkpoint_files(dry_run=False)

    if convert_old_naming_format_to_new := False:
        rename_mappings = {
        "in_use_saturating_time_at_rest_encoding_alpha": "in_use_idleness_encoding_alpha",
        "in_use_saturating_progress_encoding_alpha": "in_use_saturating_progress_encoding_omega",
        "saturating_time_at_rest_alpha": "idleness_encoding_alpha",
        "saturating_progress_encoding_alpha": "saturating_progress_encoding_omega",
        "saturating_progress_alpha": "saturating_progress_encoding_omega",
        "use_saturating_time_at_rest_from_velocity": "use_idleness_encoding",
        "use_saturating_progress_from_step_count": "use_saturating_progress_encoding",
        "geometric_progress_encoding_dim": "sinusoidal_progress_encoding_dim",
        "use_geometric_sinusoidal_progress_encoding": "use_sinusoidal_progress_encoding",
        }

        for old_param, new_param in rename_mappings.items():
            print("\n" + "="*60)
            rename_checkpoint_parameter(
                old_param=old_param,
                new_param=new_param,
                dry_run=False)

    if delete_old_parameters := False:
        old_params_to_delete = [
            "num_repeat_time_at_rest_input",
            "test_counterfactual_time_at_rest",
            "time_at_rest_embedding_dim",
            "use_normalised_agent_velocity",
            "use_sinusoidal_step_representation",
            "use_time_at_rest",
            "use_time_at_rest_embedding",
        ]

        for param in old_params_to_delete:
            print("\n" + "="*60)
            remove_checkpoint_parameter(
                param_to_remove=param,
                dry_run=True)

    if add_new_parameters_with_defaults := False:
        new_params_with_defaults = {
            "num_dataloader_workers": 8,
            "override_checkpoint_name": "",
        }
        for param, default_value in new_params_with_defaults.items():
            print("\n" + "="*60)
            add_checkpoint_parameter(
                param_to_add=param,
                value=default_value,
                dry_run=False)

    if add_missing_idleness_param := False:
        # These parameters were in use prior to having separate ones for in-use idleness
        print("\n" + "="*60)
        conditional_update_checkpoint_parameter(
            conditions={
                'use_idleness_encoding': True,
                'saturating_progress_encoding_omega': 0.047,
                'idleness_encoding_alpha': None
            },
            updates={
                'idleness_encoding_alpha': 0.047
            },
            dry_run=True)
    
    if add_missing_in_use_params := False:
        print("\n" + "="*60)
        conditional_update_checkpoint_parameter(
            conditions={'use_idleness_encoding': True, 'idleness_encoding_alpha': 0.047},
            updates={'in_use_idleness_encoding_alpha': 0.047},
            dry_run=False)
        conditional_update_checkpoint_parameter(
                conditions={'use_saturating_progress_encoding': True, 'saturating_progress_encoding_omega': 0.047},
                updates={'in_use_saturating_progress_encoding_omega': 0.047},
            dry_run=False)
    
    if update_hostname_in_checkpoints := False:
        print("\n" + "="*60)
        conditional_update_checkpoint_parameter(
            conditions={'host': 'sydney1'},
            updates={'host': 'b08d56'},
            dry_run=False)
        conditional_update_checkpoint_parameter(
            conditions={'host': 'sydney2'},
            updates={'host': 'bc54d6'},
            dry_run=False)


    if generate_or_delete_missing_config_files := True:
        missing_config_files(dry_run=False, operation='generate')



    #########################
    ### INFERENCE DATASETS ##
    #########################
    # Checking inference checkpoint paths
    
    # check_inference_checkpoint_paths()
    # compare_inference_zattrs()
    
    # Rename inference datasets (sydney1 → s1, etc.)
    # print("Renaming inference datasets in DRY RUN mode...\n")
    if rename_datasets := False:
        rename_inference_datasets(
            # e.g. if its moved: inference_dir=f"{PROJECT_ROOT}/data/inference/_archive/archive_liftqa_v2_static_alpha",
            rename_patterns={
                'sydney1': 'b08d56', 
                'sydney2': 'bc54d6',
                '+VT': '+ID',
                '+SP': '+SaP',
                '+GP': '+SiP'
            },
            dry_run=False
        )

    if add_missing_zattrs_parameters := False:
        new_params_with_defaults = {
            "idleness_encoding_alpha": None,
            "saturating_progress_encoding_omega": None,
        }
        for param, default_value in new_params_with_defaults.items():
            print("\n" + "="*60)
            add_inference_zattrs_parameter(
                param_to_add=param,
                value=default_value,
                dry_run=False)

    if rename_inference_zattrs_parameters := False:
        rename_mappings = {
            "in_use_saturating_time_at_rest_encoding_alpha": "in_use_idleness_encoding_alpha",
            "in_use_saturating_progress_encoding_alpha": "in_use_saturating_progress_encoding_omega",
            "saturating_time_at_rest_alpha": "idleness_encoding_alpha",
            "saturating_progress_encoding_alpha": "saturating_progress_encoding_omega",
            "saturating_progress_alpha": "saturating_progress_encoding_omega",
            "use_saturating_time_at_rest_from_velocity": "use_idleness_encoding",
            "use_saturating_progress_from_step_count": "use_saturating_progress_encoding",
            "geometric_progress_encoding_dim": "sinusoidal_progress_encoding_dim",
            "use_geometric_sinusoidal_progress_encoding": "use_sinusoidal_progress_encoding",
        }
        for old_param, new_param in rename_mappings.items():
            print("\n" + "="*60)
            update_inference_zattrs_parameter(
                old_param=old_param,
                new_param=new_param,
                dry_run=False)

    if remove_inference_zattrs_parameters := False:
        old_params_to_remove = [
            "num_repeat_time_at_rest_input",
            "test_counterfactual_time_at_rest",
            "time_at_rest_embedding_dim",
            "use_normalised_agent_velocity",
            "use_sinusoidal_step_representation",
            "use_time_at_rest",
            "use_time_at_rest_embedding",
            "test_temperature_value",
            "use_dynamic_temperature_value",
            "use_dynamic_temperature_value_based_on_velocity",
            "use_normalised_agent_velocity",
            "dynamic_temperature_value_while_resting",
        ]

        remove_inference_zattrs_parameter(
            # inference_dir=f"{PROJECT_ROOT}/data/inference/_archive/archive_liftqa_v2_2_tightcamerabounds",
            use_wildcard=True,
            param_to_remove="checkpoint_train_cfg.*",
            dry_run=False
        )

        for param in old_params_to_remove:
            print("\n" + "="*60)
            remove_inference_zattrs_parameter(
                param_to_remove=param,
                dry_run=False)

    if sort_inference_zattrs_files := False:
        sort_inference_zattrs(dry_run=False)
