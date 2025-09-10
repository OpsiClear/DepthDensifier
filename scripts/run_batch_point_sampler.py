#!/usr/bin/env python3
"""
Batch script to run depth densification with point sampling on all datasets in the data directory.
"""

import sys
from pathlib import Path
from typing import List
import logging
from dataclasses import dataclass, field
import tyro
from run_pipeline_point_sampler import (
    main as run_pipeline_main, 
    ScriptConfig, 
    PathsConfig, 
    MoGeConfig,
    SamplingConfig,
    ProcessingConfig
)
from depthdensifier.depth_refiner import RefinerConfig
from depthdensifier.floater_filter import FloaterFilterConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing with point sampling."""
    
    data_dir: Path = Path("data/360_v2")
    """Directory containing datasets to process."""
    results_dir: Path = Path("results_point_sampler")  
    """Directory to save results."""
    
    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """Point sampling parameters."""
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    """Processing parameters like downsampling and batch size."""
    
    # Filtering configuration  
    filtering: FloaterFilterConfig = field(default_factory=FloaterFilterConfig)
    """Multi-view consistency filtering parameters."""
    
    # Refiner configuration
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    """Depth refinement parameters."""
    
    # MoGe model configuration
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    """MoGe model configuration."""


def find_datasets(data_dir: Path) -> List[str]:
    """Find all valid datasets in the data directory."""
    datasets = []

    # Skip these files/directories
    skip_items = {'flowers.txt', 'treehill.txt', 'outputs', 'temp'}

    for item in data_dir.iterdir():
        if item.is_dir() and item.name not in skip_items:
            # Check if it has the required structure
            sparse_dir = item / "sparse" / "0"
            images_dir = item / "images"

            if sparse_dir.exists() and images_dir.exists():
                # Check for COLMAP files
                cameras_bin = sparse_dir / "cameras.bin"
                images_bin = sparse_dir / "images.bin"
                points3d_bin = sparse_dir / "points3D.bin"

                if cameras_bin.exists() and images_bin.exists() and points3d_bin.exists():
                    datasets.append(item.name)
                    logger.info(f"Found valid dataset: {item.name}")
                else:
                    logger.warning(f"Dataset {item.name} missing COLMAP files, skipping")
            else:
                logger.warning(f"Dataset {item.name} missing sparse/0 or images directory, skipping")

    return sorted(datasets)


def run_densification_for_dataset(dataset_name: str, batch_config: BatchConfig) -> bool:
    """Run depth densification with point sampling for a single dataset."""
    logger.info(f"Starting densification for dataset: {dataset_name}")

    # Set up paths
    recon_path = batch_config.data_dir / dataset_name / "sparse" / "0"
    image_dir = batch_config.data_dir / dataset_name / "images"
    output_dir = batch_config.results_dir / dataset_name / "sparse" / "0"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create configuration for the pipeline using batch config settings
        config = ScriptConfig(
            paths=PathsConfig(
                recon_path=recon_path,
                image_dir=image_dir,
                output_model_dir=output_dir
            ),
            sampling=batch_config.sampling,
            processing=batch_config.processing,
            filtering=batch_config.filtering,
            refiner=batch_config.refiner,
            moge=batch_config.moge
        )

        logger.info(f"Running pipeline for {dataset_name}")
        logger.info(f"  Reconstruction: {recon_path}")
        logger.info(f"  Images: {image_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Sampling config: strategy={config.sampling.sampling_strategy}, num_points={config.sampling.num_points_per_image}, edge_weight={config.sampling.edge_weight}")
        logger.info(f"  Processing config: downsample_factor={config.processing.pipeline_downsample_factor}, batch_size={config.processing.batch_size}")
        logger.info(f"  Filtering config: vote_threshold={config.filtering.vote_threshold}, depth_threshold={config.filtering.depth_threshold}, grazing_angle={config.filtering.grazing_angle_threshold}")

        # Run the pipeline directly
        run_pipeline_main(config)

        logger.info(f"Successfully completed densification for {dataset_name}")
        return True

    except Exception as e:
        logger.error(f"Exception processing {dataset_name}: {e}")
        return False


def main(config: BatchConfig):
    """Main function to run batch processing with point sampling."""
    if not config.data_dir.exists():
        logger.error(f"Data directory {config.data_dir} does not exist")
        sys.exit(1)

    logger.info("Starting batch depth densification process with point sampling")
    logger.info(f"Data directory: {config.data_dir.absolute()}")
    logger.info(f"Results directory: {config.results_dir.absolute()}")
    logger.info("Configuration:")
    logger.info(f"  Sampling: strategy={config.sampling.sampling_strategy}, num_points={config.sampling.num_points_per_image}, edge_weight={config.sampling.edge_weight}")
    logger.info(f"  Processing: downsample_factor={config.processing.pipeline_downsample_factor}, batch_size={config.processing.batch_size}")
    logger.info(f"  Filtering: vote_threshold={config.filtering.vote_threshold}, depth_threshold={config.filtering.depth_threshold}, grazing_angle={config.filtering.grazing_angle_threshold}")
    logger.info(f"  MoGe model: {config.moge.checkpoint}")

    # Find all datasets
    datasets = find_datasets(config.data_dir)

    if not datasets:
        logger.error("No valid datasets found")
        sys.exit(1)

    logger.info(f"Found {len(datasets)} datasets to process: {', '.join(datasets)}")

    # Process each dataset
    successful = 0
    failed = 0

    for dataset in datasets:
        success = run_densification_for_dataset(dataset, config)

        if success:
            successful += 1
        else:
            failed += 1

        logger.info(f"Progress: {successful + failed}/{len(datasets)} datasets processed")

    # Summary
    logger.info("="*50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(datasets)}")
    logger.info(f"Results saved to: {config.results_dir.absolute()}")
    logger.info(f"Point sampling strategy used: {config.sampling.sampling_strategy}")
    logger.info(f"Points per image: {config.sampling.num_points_per_image}")

    if failed > 0:
        logger.warning(f"{failed} datasets failed processing")
        sys.exit(1)
    else:
        logger.info("All datasets processed successfully!")


if __name__ == "__main__":
    tyro.cli(main)