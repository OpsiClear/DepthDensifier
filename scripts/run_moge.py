import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from moge.model.v2 import MoGeModel
from tqdm import tqdm


def save_outputs(output: dict[str, torch.Tensor], output_dir: str | Path, base_name: str) -> None:
    """Save the output of the model to disk.
    
    :param output: Model output dictionary containing depth, mask, and optionally normal
    :param output_dir: Directory to save outputs
    :param base_name: Base filename without extension
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move outputs to cpu and numpy
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()

    # Save depth map
    depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
    # Normalize for visualization
    depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(depth_path, depth_vis)

    # Save mask
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask * 255)

    if "normal" in output:
        normal = output["normal"].cpu().numpy()
        normal_path = os.path.join(output_dir, f"{base_name}_normal.png")
        # Remap from [-1, 1] to [0, 255] for visualization
        normal_vis = (normal * 0.5 + 0.5) * 255
        normal_vis = normal_vis.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(normal_path, cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))


@dataclass
class MoGeConfig:
    """Configuration for MoGe inference."""
    
    input_dir: str
    """Path to the input directory with images"""
    
    output_dir: str
    """Path to the output directory to save results"""
    
    num_tokens: int | None = None
    """The number of base ViT tokens to use for inference. Suggested range: 1200 ~ 2500. Default: 'most'"""
    
    no_fp16: bool = False
    """If True, do not use mixed precision to speed up inference"""


def main(config: MoGeConfig) -> None:
    """Run MoGe inference on a directory of images.
    
    :param config: Configuration for MoGe inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU. This will be very slow.")

    # Load the model from huggingface hub (or load from local).
    print("Loading model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    print("Model loaded.")

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = sorted([f for f in os.listdir(config.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])

    if not image_files:
        print(f"No images found in {config.input_dir}")
        return

    for image_file in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(config.input_dir, image_file)
        
        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Infer 
        with torch.no_grad():
            output = model.infer(
                input_image,
                num_tokens=config.num_tokens,
                use_fp16=not config.no_fp16
            )
        
        base_name = os.path.splitext(image_file)[0]
        save_outputs(output, config.output_dir, base_name)


if __name__ == "__main__":
    tyro.cli(main)