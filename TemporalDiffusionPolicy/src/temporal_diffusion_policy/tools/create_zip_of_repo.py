#!/usr/bin/env python3
"""
Script to create a zip file of the repository with flexible pattern-based inclusion/exclusion.
Supports glob patterns for both files and folders.
"""

import os
import zipfile
from pathlib import Path
from fnmatch import fnmatch
from typing import List
import argparse


class ZipCreator:
    def __init__(self, root_dir: str, output_zip: str, base_folder: str = None):
        """
        Initialize the ZipCreator.
        
        Args:
            root_dir: Root directory of the repository
            output_zip: Output zip file path
            base_folder: Base folder name inside the zip (if None, files go to zip root)
        """
        self.root_dir = Path(root_dir).resolve()
        self.output_zip = output_zip
        self.base_folder = base_folder
        self.include_patterns: List[str] = []
        self.exclude_patterns: List[str] = []
        
    def add_include_pattern(self, pattern: str):
        """Add a glob pattern to include (works for both files and folders)."""
        self.include_patterns.append(pattern)
        
    def add_exclude_pattern(self, pattern: str):
        """Add a glob pattern to exclude (works for both files and folders)."""
        # Normalize trailing slashes for consistency
        pattern = pattern.rstrip('/')
        self.exclude_patterns.append(pattern)
    
    def matches_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a pattern.
        
        Args:
            path: Relative path to check
            pattern: Glob pattern
            
        Returns:
            True if path matches pattern
        """
        # Normalize pattern
        pattern = pattern.rstrip('/')
        
        # Direct match
        if fnmatch(path, pattern):
            return True
        
        # Match with wildcard (e.g., pattern/* matches path)
        if fnmatch(path, f"{pattern}/*"):
            return True
        
        # Check if any parent directory matches (for folder exclusions)
        parts = Path(path).parts
        for i in range(len(parts)):
            partial_path = str(Path(*parts[:i+1]))
            if fnmatch(partial_path, pattern):
                return True
        
        return False
    
    def could_contain_matches(self, dir_path: str) -> bool:
        """
        Check if a directory could potentially contain files matching include patterns.
        Used for directory traversal decisions.
        
        Args:
            dir_path: Relative directory path
            
        Returns:
            True if directory should be traversed
        """
        if not self.include_patterns:
            return False
        
        # Check if any include pattern could match files in this directory
        for pattern in self.include_patterns:
            pattern = pattern.rstrip('/')
            # If pattern starts with this directory path, we need to traverse it
            if pattern.startswith(dir_path + '/'):
                return True
            # If directory matches the pattern or is a parent of pattern
            if fnmatch(dir_path, pattern) or fnmatch(dir_path, pattern.split('/')[0] if '/' in pattern else pattern):
                return True
            # Check if directory is part of the pattern path
            pattern_parts = Path(pattern).parts
            dir_parts = Path(dir_path).parts
            if len(dir_parts) <= len(pattern_parts):
                # Check if directory is on the path to pattern
                matches = True
                for i, part in enumerate(dir_parts):
                    if not fnmatch(part, pattern_parts[i]):
                        matches = False
                        break
                if matches:
                    return True
        return False
    
    def should_include(self, relative_path: str, is_dir: bool = False) -> bool:
        """
        Determine if a path should be included based on patterns.
        
        Args:
            relative_path: Path relative to root directory
            is_dir: Whether this is a directory (for traversal) or a file
            
        Returns:
            True if path should be included, False otherwise
        """
        # Check exclude patterns first (they take precedence)
        for pattern in self.exclude_patterns:
            if self.matches_pattern(relative_path, pattern):
                return False
        
        # If no include patterns specified, don't include anything
        if not self.include_patterns:
            return False
        
        # For directories, check if we should traverse into them
        if is_dir:
            return self.could_contain_matches(relative_path)
            
        # For files, check include patterns
        for pattern in self.include_patterns:
            if self.matches_pattern(relative_path, pattern):
                return True
                
        return False
    
    
    def create_zip(self, verbose: bool = True, dry_run: bool = False):
        """
        Create the zip file based on configured patterns.
        
        Args:
            verbose: Print progress information
            dry_run: If True, only show what would be added without creating zip
        """
        files_added = 0
        files_excluded = 0
        
        if dry_run:
            print("DRY RUN MODE - No zip file will be created\n")
            print("Files that would be added:")
            print("-" * 80)
        
        # Open zip file only if not dry run
        zipf = None if dry_run else zipfile.ZipFile(self.output_zip, 'w', zipfile.ZIP_DEFLATED)
        
        try:
            # Walk through the directory tree
            for root, dirs, files in os.walk(self.root_dir):
                root_path = Path(root)
                
                # Calculate relative path
                try:
                    relative_root = root_path.relative_to(self.root_dir)
                except ValueError:
                    continue
                
                # Filter out excluded directories
                dirs_to_remove = []
                for dir_name in dirs:
                    dir_path = root_path / dir_name
                    relative_dir = str(dir_path.relative_to(self.root_dir))
                    
                    if not self.should_include(relative_dir, is_dir=True):
                        dirs_to_remove.append(dir_name)
                        if verbose and not dry_run:
                            print(f"Excluding folder: {relative_dir}")
                
                # Remove excluded directories from walk
                for dir_name in dirs_to_remove:
                    dirs.remove(dir_name)
                
                # Process files
                for file_name in files:
                    file_path = root_path / file_name
                    relative_file = str(file_path.relative_to(self.root_dir))
                    
                    # Skip the output zip file itself
                    if not dry_run and file_path.resolve() == Path(self.output_zip).resolve():
                        continue
                    
                    if self.should_include(relative_file, is_dir=False):
                        if not dry_run:
                            # Add base_folder prefix if specified
                            zip_path = f"{self.base_folder}/{relative_file}" if self.base_folder else relative_file
                            zipf.write(file_path, zip_path)
                        files_added += 1
                        if verbose or dry_run:
                            display_path = f"{self.base_folder}/{relative_file}" if self.base_folder else relative_file
                            print(f"{'Would add' if dry_run else 'Added'}: {display_path}")
                    else:
                        files_excluded += 1
        finally:
            if zipf:
                zipf.close()
        
        if verbose or dry_run:
            print("\n" + "=" * 80)
            if dry_run:
                print("DRY RUN SUMMARY:")
                print(f"Files that would be added: {files_added}")
                print(f"Files that would be excluded: {files_excluded}")
            else:
                print(f"Zip file created: {self.output_zip}")
                print(f"Files added: {files_added}")
                print(f"Files excluded: {files_excluded}")
                print(f"Total size: {os.path.getsize(self.output_zip) / (1024*1024):.2f} MB")


def main():
    """Main function with example configuration."""
    parser = argparse.ArgumentParser(
        description='Create a zip file of the repository with pattern-based filtering'
    )
    parser.add_argument(
        '--output', '-o',
        default='repo_archive.zip',
        help='Output zip file name (default: repo_archive.zip)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        default=False,
        help='Show what would be zipped without creating the zip file'
    )
    
    args = parser.parse_args()
    
    # Get the script's d with base folder
    root_dir = Path.cwd()
    creator = ZipCreator(root_dir, args.output, base_folder="TemporalConditioningDiffusionPolicy")
    
    
    # ============================================================================
    # CONFIGURATION SECTION - Customize these patterns for your needs
    # ============================================================================
    
    # Include patterns (glob-style)
    # Patterns work for both files and folders

    creator.add_include_pattern("docs/*")

    creator.add_include_pattern("src/temporal_diffusion_policy/*")

    creator.add_include_pattern("scripts/*")
    creator.add_include_pattern("tests/*")
    

    creator.add_include_pattern(".gitignore")
    creator.add_include_pattern("pyproject.toml")
    creator.add_include_pattern("setup.py")
    creator.add_include_pattern("README.md")
    creator.add_include_pattern("requirements.txt")
    creator.add_include_pattern("pytest.ini")


    # def add_dataset(creator, dataset_path: str):
    #     creator.add_include_pattern(f"{dataset_path}/meta/*")
    #     creator.add_include_pattern(f"{dataset_path}/.z*")
    #     for attrib in ['full_image', 'agentview_image']:
    #         creator.add_exclude_pattern(f"{dataset_path}/data/{attrib}/*")

    # datasets
    # add_dataset(creator, "data/demonstration/wait_at_goal_dataset_v4.zarr")
    # add_dataset(creator, "data/demonstration/lift_qa_v2.zarr")

    # Exclude patterns (these take precedence over include patterns)
    # Works for files and folders - use patterns to control what you want
    creator.add_exclude_pattern("*.pyc")
    creator.add_exclude_pattern("*__pycache__")

    # from analysis_waittime import (
    #     wait_at_goal_demo_datasets, 
    #     wait_at_goal_baseline_datasets, 
    #     wait_at_goal_main_datasets, 
    #     wait_at_goal_extended_datasets,
    #     liftqa_demo_datasets,
    #     liftqa_baseline_datasets,
    #     liftqa_main_datasets,
    #     liftqa_extended_datasets,
    # )
    # datasets = [
    #     wait_at_goal_demo_datasets,
    #     wait_at_goal_baseline_datasets,
    #     wait_at_goal_main_datasets,
    #     wait_at_goal_extended_datasets,
    #     liftqa_demo_datasets,
    #     liftqa_baseline_datasets,
    #     liftqa_main_datasets,
    #     liftqa_extended_datasets,
    # ]
    # datasets = [item for sublist in datasets for item in sublist]
    # dataset_paths = list(set([d.path for d in datasets]))
    # for dataset_path in dataset_paths:
    #     add_dataset(creator, dataset_path)
    # ============================================================================
    # END CONFIGURATION SECTION
    # ============================================================================
    
    # Create the zip file (or show what would be created)
    creator.create_zip(verbose=not args.quiet, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
