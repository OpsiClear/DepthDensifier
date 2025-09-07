"""
This script runs the densification process in batch mode on multiple scans.

It iterates through all subdirectories in a specified root directory, assuming
each subdirectory contains a single scan with the following structure:
- <scan_name>/images/
- <scan_name>/sparse/0/

The script will create a 'results' directory within each scan folder to store
the output point cloud and the densified COLMAP model.
"""

from pathlib import Path
import tyro
from dataclasses import dataclass, field
import sys
import os
import time

# Add the script's directory to the Python path to allow importing the test script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test import main as densify_main, ScriptConfig, PathsConfig


@dataclass
class BatchConfig:
    """Configuration for the batch processing script."""
    root_dir: Path
    """The root directory containing the individual scan folders."""
    
    output_dir: Path
    """The directory to save the output point clouds and models."""
    
    # Embed the main script's configuration to allow overrides from the CLI.
    # For example, you can run:
    # python scripts/run_batch.py --root-dir data/scans --config.filtering.vote-threshold 3
    config: ScriptConfig = field(default_factory=ScriptConfig)


def main(batch_config: BatchConfig):
    """
    Runs the densification script on all valid scan folders
    found in the specified root directory.
    """
    batch_start_time = time.time()
    root_path = batch_config.root_dir.resolve()
    if not root_path.is_dir():
        print(f"Error: Root directory not found at {root_path}")
        return

    scan_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
    print(f"Found {len(scan_folders)} potential scan folders in {root_path}.")

    time_reports = []

    for scan_folder in scan_folders:
        print(f"\n{'='*80}\nProcessing scan: {scan_folder.name}\n{'='*80}")

        # Define the expected paths for a valid scan folder
        recon_path = scan_folder / "sparse" / "0"
        image_dir = scan_folder / "images"
        
        # Define output paths
        output_dir = batch_config.output_dir / scan_folder.name
        output_model_dir = output_dir / "sparse" / "0"

        # Check if the required input directories exist
        if not recon_path.is_dir() or not image_dir.is_dir():
            print(f"Skipping '{scan_folder.name}': Missing 'sparse/0' or 'images' directory.")
            continue

        # Use the base config provided via the CLI and override the paths for the current scan
        run_config = batch_config.config
        run_config.paths = PathsConfig(
            recon_path=recon_path,
            image_dir=image_dir,
            output_model_dir=output_model_dir,
        )

        scan_start_time = time.time()
        try:
            densify_main(run_config)
            scan_duration = time.time() - scan_start_time
            time_reports.append((scan_folder.name, scan_duration))
            print(f"\nSuccessfully finished processing scan: {scan_folder.name}")
        except Exception as e:
            # Catch exceptions to prevent one failed scan from stopping the entire batch
            time_reports.append((scan_folder.name, "FAILED"))
            print(f"\n!!!!!!!!!!\nAn error occurred while processing '{scan_folder.name}': {e}\n!!!!!!!!!!")
            continue

    batch_duration = time.time() - batch_start_time

    # --- Print Final Time Report ---
    print(f"\n\n{'='*63}")
    print(f"{'Batch Processing Time Report':^63}")
    print(f"{'='*63}")
    print(f"{'Scan Name':<40} | {'Duration (s)':>18}")
    print(f"{'-'*40}-+-{'-'*18}")

    for name, duration in time_reports:
        if isinstance(duration, float):
            print(f"{name:<40} | {duration:>18.2f}")
        else:
            print(f"{name:<40} | {duration:>18}")

    print(f"{'-'*40}-+-{'-'*18}")
    print(f"{'Total Time':<40} | {batch_duration:>18.2f}")
    print(f"{'='*63}\n")


if __name__ == "__main__":
    tyro.cli(main) 