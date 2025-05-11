"""The main inference module."""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from weathergan.turbo.cyclegan_turbo import CycleGAN_Turbo
from weathergan.turbo.my_utils.training_utils import build_transform
from diffusers import StableDiffusionUpscalePipeline
from RealESRGAN import RealESRGAN


def main():
    """Inference pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="path to the input image or directory containing images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="image taken outdoors on a sunny day",
        required=False,
        help="the prompt to be used. It is required when loading a custom model_path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="name of the pretrained model to be used",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path to a local model state dict to be used",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="the directory to save the output",
    )
    parser.add_argument(
        "--sr_algorithm",
        type=str,
        choices=["diffusion", "realesrgan"],
        default="realesrgan",
        help="Super-resolution algorithm to use: 'diffusion' or 'realesrgan'",
    )
    parser.add_argument(
        "--image_prep",
        type=str,
        default="resize_512x512",
        help="the image preparation method",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="a2b",
        help="the direction of translation. None for pretrained models, a2b or b2a for custom paths.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use Float16 precision for faster inference",
    )
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError("Either model_name or model_path should be provided")

    if args.model_path is not None and args.prompt is None:
        raise ValueError("prompt is required when loading a custom model_path.")

    if args.model_name is not None:
        assert (
            args.prompt is None
        ), "prompt is not required when loading a pretrained model."
        assert (
            args.direction is None
        ), "direction is not required when loading a pretrained model."

    # initialize the model
    model = CycleGAN_Turbo(
        pretrained_name=args.model_name, pretrained_path=args.model_path
    )
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the upscaler pipeline based on chosen algorithm
    if args.sr_algorithm == "diffusion":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16,
        )
        pipeline = pipeline.to("cuda")
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_attention_slicing()
        pipeline.enable_model_cpu_offload()
    elif args.sr_algorithm == "realesrgan":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = RealESRGAN(device, scale=4)
        pipeline.load_weights('weights/RealESRGAN_x4.pth', download=True)
    else:
        raise ValueError(f"Invalid SR algorithm: {args.sr_algorithm}")

    # Process single image or directory
    input_path = Path(args.input)
    if input_path.is_file():
        input_image = Image.open(input_path).convert("RGB")
        process_image(input_image, input_path.name, model, T_val, args, pipeline)
    elif input_path.is_dir():
        # Find all image files in the directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in image_extensions]
        
        for image_file in image_files:
            print(f"Processing {image_file.name}...")
            try:
                input_image = Image.open(image_file).convert("RGB")
                process_image(input_image, image_file.name, model, T_val, args, pipeline)
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")


def process_image(image, filename, model, T_val, args, pipeline):
    """Process a single image and save the output."""
    with torch.no_grad():
        input_img = T_val(image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        if args.use_fp16:
            x_t = x_t.half()
        output = model(x_t, direction=args.direction, caption=args.prompt)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_path = os.path.join(args.output_dir, filename)
    output_pil.save(output_path)
    torch.cuda.empty_cache()

    # Upscale the image
    if args.sr_algorithm == "diffusion":
        upscaled_image = pipeline(prompt=args.prompt, image=output_pil).images[0]
    elif args.sr_algorithm == "realesrgan":
        upscaled_image = pipeline.predict(output_pil)
    else:
        raise ValueError(f"Invalid SR algorithm: {args.sr_algorithm}")

    # Rescale back to original size
    upscaled_image = upscaled_image.resize(image.size, Image.LANCZOS)
    print("upscale complete.")

    # Save the output image
    upscaled_image.save(os.path.join(args.output_dir, filename))
    print(f"Complete. Saved output to {os.path.join(args.output_dir, filename)}")


if __name__ == "__main__":
    main()
