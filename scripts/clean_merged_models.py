#!/usr/bin/env python3
import argparse
import fnmatch
import os
import shutil
from pathlib import Path

# Default directories that should be preserved entirely
DEFAULT_WHITELIST = []


def clean_merge_dir(merge_dir: Path, keep_patterns: list[str], verbose: bool = False, dry_run: bool = False):
    """
    Clean a single merge directory by deleting everything except items
    matching the keep_patterns.
    """
    if verbose or dry_run:
        print(f"  {'Would clean' if dry_run else 'Cleaning'}: {merge_dir}")
    
    # First, identify all items that should be preserved
    items_to_keep = []
    for item in merge_dir.iterdir():
        # Check if the item name matches any of our keep patterns
        if any(fnmatch.fnmatch(item.name, pat) for pat in keep_patterns):
            items_to_keep.append(item)
            if verbose or dry_run:
                item_type = "directory" if item.is_dir() else "file"
                print(f"    {'Would keep' if dry_run else 'Keeping'} {item_type}: {item.name}")
    
    # Now process all items, keeping the identified items
    for item in merge_dir.iterdir():
        # Skip items that we identified to keep
        if item in items_to_keep:
            continue
        
        # Delete everything else
        if item.is_dir():
            if verbose or dry_run:
                print(f"    {'Would remove' if dry_run else 'Removing'} directory: {item.name}")
            if not dry_run:
                shutil.rmtree(item)
        else:
            if verbose or dry_run:
                print(f"    {'Would remove' if dry_run else 'Removing'} file: {item.name}")
            if not dry_run:
                item.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Clean merged_models directories by removing all contents except eval* folders"
    )
    parser.add_argument(
        "--whitelist", 
        "-w", 
        nargs="*", 
        default=DEFAULT_WHITELIST, 
        help="Names of merge directories to preserve entirely"
    )
    parser.add_argument(
        "--keep-patterns", 
        "-k", 
        nargs="*", 
        default=["eval*", "clusters*", "reap_args.yaml"], 
        help="Patterns for directories to keep (default: eval*)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Print detailed information about actions"
    )
    parser.add_argument(
        "--dry-run", 
        "-n", 
        action="store_true",
        help="Smoke test: show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()

    artifacts_dir = Path.cwd() / "artifacts"
    whitelist = set(args.whitelist)
    # Make sure "eval" is explicitly included in keep patterns
    keep_patterns = args.keep_patterns
    if "eval*" in keep_patterns and "eval" not in keep_patterns:
        keep_patterns.append("eval")
    
    if args.verbose:
        print(f"Looking for merged_models directories in: {artifacts_dir}")
        print(f"Keep patterns: {keep_patterns}")
        print(f"Whitelist: {whitelist}")
        print(f"Dry run: {args.dry_run}")

    # Find all merged_models directories and process their nested structure
    for root, dirs, _ in os.walk(artifacts_dir):
        root_path = Path(root)
        subdirs_to_check = ['merged_models', "pruned_models", "non_uniform_merged_models"]
        if root_path.name in subdirs_to_check:
            if args.verbose or args.dry_run:
                print(f"Found merged_models directory: {root_path}")
            
            # Process all nested directories under merged_models
            process_merged_models_dir(root_path, whitelist, keep_patterns, args.verbose, args.dry_run)


def process_merged_models_dir(merged_models_dir: Path, whitelist: set, keep_patterns: list, verbose: bool, dry_run: bool):
    """
    Process a merged_models directory by traversing its nested structure to find and clean model directories.
    """
    # Queue to hold directories to process (using breadth-first search)
    dirs_to_process = [merged_models_dir]
    
    while dirs_to_process:
        current_dir = dirs_to_process.pop(0)
        
        # Check if this directory is a leaf directory that needs cleaning
        # We determine this by checking if it contains files with .safetensors extension
        has_model_files = any(item.suffix == '.safetensors' for item in current_dir.iterdir() if item.is_file())
        
        if has_model_files:
            # This is a leaf directory containing model files that needs cleaning
            if str(current_dir) in whitelist:
                if verbose or dry_run:
                    print(f"  {'Would preserve entire directory' if dry_run else 'Skipping whitelisted'}: {current_dir}")
                continue
            
            # Clean the directory
            clean_merge_dir(current_dir, keep_patterns, verbose, dry_run)
        else:
            # This is an intermediate directory, add its subdirectories to the processing queue
            for item in current_dir.iterdir():
                if item.is_dir():
                    dirs_to_process.append(item)


if __name__ == "__main__":
    main()
