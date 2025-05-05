import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.iit import VSAIT
from solver import VSAITSolver
from utils.config import get_cfg
from utils.logger import get_module_logger

logger = get_module_logger(__name__)

def preprocess_image(image_path, img_size=(256, 256)):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms and add batch dimension
    image = transform(image).unsqueeze(0)
    return image

def postprocess_image(tensor):
    """Convert tensor to PIL image"""
    # Denormalize
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.squeeze(0)
    
    # Convert to numpy and create PIL image
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def load_model(cfg, checkpoint_path):
    """Load the VSAITSolver with pretrained weights"""
    # set checkpoint in config so solver._load_model picks it up
    cfg.TASK_MODEL.WEIGHTS = checkpoint_path
    solver = VSAITSolver(cfg)
    solver.eval()
    return solver

def main(args):
    # Parse config files
    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    
    # Load solver (loads model weights)
    solver = load_model(cfg, args.checkpoint)

    # Preprocess input image
    logger.info(f"Processing input image: {args.input_image}")
    input_tensor = preprocess_image(args.input_image)

    # Move to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solver = solver.to(device)
    input_tensor = input_tensor.to(device)
    # verify model is on GPU
    print("On CUDA after .to(device):", next(solver.model.g_net.parameters()).is_cuda)
    print("First conv weight mean:", float(next(solver.model.g_net.parameters()).mean()))

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        output_tensor = solver.model.inference(input_tensor)
    
    # Postprocess output
    output_image = postprocess_image(output_tensor)
    
    # Save output
    output_path = args.output_image or os.path.splitext(args.input_image)[0] + '_translated.png'
    logger.info(f"Saving translated image to: {output_path}")
    output_image.save(output_path)
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Image Inference')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input source image')
    parser.add_argument('--output_image', default='output.png', type=str,
                        help='Path to save translated image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model configuration file')
    
    args = parser.parse_args()
    main(args)
