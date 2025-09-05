#!/usr/bin/env python3
"""
Download test data and models for DepthDensifier.

This script downloads the 360_v2 dataset and sets up the data directory structure.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

import tyro

try:
    from huggingface_hub import snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class DatasetConfig(TypedDict):
    """Configuration for a dataset to download."""

    name: str
    url: str
    filename: str
    extract: bool


class MoGeModelConfig(TypedDict):
    """Configuration for a MoGe model to download."""

    name: str
    repo_id: str
    description: str
    version: str
    metric_scale: bool
    normal_map: bool
    params: str


@dataclass
class DownloadConfig:
    """Configuration for downloading test data and models."""

    data_dir: str = "data"
    """Directory to store data"""

    keep_zip: bool = False
    """Keep zip files after extraction"""

    skip_existing: bool = False
    """Skip download if files already exist"""

    download_datasets: bool = True
    """Download test datasets (360_v2)"""

    download_moge_models: bool = True
    """Download MoGe pretrained models"""

    moge_models: str = "all"
    """Which MoGe models to download: 'all', 'recommended', 'v1', 'v2', or comma-separated model names"""



def create_data_directory(data_dir: Path) -> None:
    """Create the data directory if it doesn't exist.

    :param data_dir: Path to the data directory
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created data directory: {data_dir}")


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL.

    :param url: URL to download from
    :param output_path: Local path to save the file
    """
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded: {output_path.name}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def extract_and_cleanup(
    zip_path: Path, extract_to: Path, keep_zip: bool = False
) -> None:
    """Extract zip file and optionally clean up.

    :param zip_path: Path to the zip file
    :param extract_to: Directory to extract to
    :param keep_zip: Whether to keep the zip file after extraction
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path.name}")

        if not keep_zip:
            zip_path.unlink()
            print(f"Cleaned up: {zip_path.name}")

    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        sys.exit(1)


def check_360v2_dataset_exists(data_dir: Path) -> bool:
    """Check if 360_v2 dataset already exists in data directory.
    
    :param data_dir: Path to the data directory
    :return: True if dataset exists, False otherwise
    """
    # Check in the expected 360_v2 subdirectory
    expected_dir = data_dir / "360_v2"
    if expected_dir.exists() and any(expected_dir.iterdir()):
        return True
    
    # Check if dataset was extracted directly to data directory
    # Look for characteristic folders/files from the 360_v2 dataset
    characteristic_items = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "flowers.txt", "treehill.txt"]
    
    found_items = 0
    for item in characteristic_items:
        item_path = data_dir / item
        if item_path.exists():
            found_items += 1
    
    # If we find at least 5 characteristic items, consider dataset present
    return found_items >= 5


def download_moge_model(
    model_config: MoGeModelConfig, models_dir: Path, skip_existing: bool = False
) -> None:
    """Download a MoGe model from Hugging Face.

    :param model_config: Configuration for the model to download
    :param models_dir: Directory to store models
    :param skip_existing: Skip download if model already exists
    """
    if not HF_AVAILABLE:
        print(f"Warning: Skipping {model_config['name']} (huggingface_hub not available)")
        return

    moge_models_dir = models_dir / "moge"
    moge_models_dir.mkdir(parents=True, exist_ok=True)
    model_path = moge_models_dir / model_config["name"]

    if skip_existing and model_path.exists() and any(model_path.iterdir()):
        print(f"Skipping {model_config['name']} (already exists)")
        return

    try:
        print(f"Downloading {model_config['name']} ({model_config['params']})")
        snapshot_download(
            repo_id=model_config["repo_id"],
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {model_config['name']}")
    except Exception as e:
        print(f"Error downloading {model_config['name']}: {e}")
        if model_path.exists():
            import shutil

            shutil.rmtree(model_path)



def get_moge_models() -> list[MoGeModelConfig]:
    """Get the list of available MoGe models.

    :return: List of MoGe model configurations
    """
    return [
        {
            "name": "moge-vitl",
            "repo_id": "Ruicheng/moge-vitl",
            "description": "MoGe-1 ViT-Large model",
            "version": "v1",
            "metric_scale": False,
            "normal_map": False,
            "params": "314M",
        },
        {
            "name": "moge-2-vitl",
            "repo_id": "Ruicheng/moge-2-vitl",
            "description": "MoGe-2 ViT-Large with metric scale",
            "version": "v2",
            "metric_scale": True,
            "normal_map": False,
            "params": "326M",
        },
        {
            "name": "moge-2-vitl-normal",
            "repo_id": "Ruicheng/moge-2-vitl-normal",
            "description": "MoGe-2 ViT-Large with metric scale and normal maps",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "331M",
        },
        {
            "name": "moge-2-vitb-normal",
            "repo_id": "Ruicheng/moge-2-vitb-normal",
            "description": "MoGe-2 ViT-Base with metric scale and normal maps",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "104M",
        },
        {
            "name": "moge-2-vits-normal",
            "repo_id": "Ruicheng/moge-2-vits-normal",
            "description": "MoGe-2 ViT-Small with metric scale and normal maps",
            "version": "v2",
            "metric_scale": True,
            "normal_map": True,
            "params": "35M",
        },
    ]




def main(config: DownloadConfig) -> None:
    """Main function to download and setup test data.

    :param config: Download configuration
    """
    project_root = Path(__file__).parent
    data_dir = project_root / config.data_dir
    models_dir = project_root / "models"

    print(f"Setting up data in: {data_dir}")
    print(f"Models will be stored in: {models_dir}")

    create_data_directory(data_dir)

    # Download datasets
    if config.download_datasets:
        print("\nDownloading datasets...")

        if check_360v2_dataset_exists(data_dir):
            print("Skipping 360_v2 dataset (already exists)")
        else:
            extracted_dir = data_dir / "360_v2"
            zip_path = data_dir / "360_v2.zip"
            download_file(
                "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip", zip_path
            )
            extract_and_cleanup(zip_path, extracted_dir, config.keep_zip)

    # Download MoGe models
    if config.download_moge_models:
        print("\nDownloading MoGe models...")

        models_dir.mkdir(exist_ok=True)
        available_models = get_moge_models()

        # Select models based on config
        if config.moge_models == "all":
            selected_models = available_models
        elif config.moge_models == "recommended":
            selected_models = [
                m for m in available_models if m["name"] == "moge-2-vitl-normal"
            ]
        elif config.moge_models == "v1":
            selected_models = [m for m in available_models if m["version"] == "v1"]
        elif config.moge_models == "v2":
            selected_models = [m for m in available_models if m["version"] == "v2"]
        else:
            model_names = [name.strip() for name in config.moge_models.split(",")]
            selected_models = [m for m in available_models if m["name"] in model_names]

        if not selected_models:
            print(f"Warning: No models found: {config.moge_models}")
        else:
            for model in selected_models:
                download_moge_model(model, models_dir, config.skip_existing)


    # Create additional directories
    for dir_name in ["outputs", "temp"]:
        (data_dir / dir_name).mkdir(exist_ok=True)

    print("\nSetup complete!")
    print(f"   Data: {data_dir}")
    if config.download_moge_models:
        print(f"   Models: {models_dir}")
        print(
            f"   MoGe models: Load with MoGeModel.from_pretrained('{models_dir}/moge/MODEL_NAME')"
        )


if __name__ == "__main__":
    tyro.cli(main)
