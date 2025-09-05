import cv2
import torch
import os
import argparse
from tqdm import tqdm
import numpy as np
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2


def save_outputs(output, output_dir, base_name):
    """Saves the output of the model to disk."""
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


def main():
    parser = argparse.ArgumentParser(description="Run MoGe inference on a directory of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory with images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save results.")
    parser.add_argument("--num_tokens", type=int, default=None, help="The number of base ViT tokens to use for inference. Suggested range: 1200 ~ 2500. Default: 'most'.")
    parser.add_argument("--no_fp16", action="store_true", help="If True, do not use mixed precision to speed up inference.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU. This will be very slow.")

    # Load the model from huggingface hub (or load from local).
    print("Loading model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    print("Model loaded.")

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions])

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    for image_file in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(args.input_dir, image_file)
        
        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Infer 
        with torch.no_grad():
            output = model.infer(
                input_image,
                num_tokens=args.num_tokens,
                use_fp16=not args.no_fp16
            )
        
        base_name = os.path.splitext(image_file)[0]
        save_outputs(output, args.output_dir, base_name)


if __name__ == "__main__":
    main()