import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.iit import VSAIT
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
    """Load the VSAIT model with pretrained weights"""
    model = VSAIT(cfg)
    
    # Load checkpoint
    logger.info(f"Loading weights from {checkpoint_path}")
    loaded_model = torch.load(
        checkpoint_path,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    loaded_model = loaded_model.get("state_dict", loaded_model)
    model.load_state_dict(loaded_model, strict=False)
    
    # Set model to evaluation mode
    model.eval()
    return model

def main(args):
    # Parse config files
    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    
    # Load model
    model = load_model(cfg, args.checkpoint)
    
    # Preprocess both source and target reference images
    logger.info(f"Processing source image: {args.source_image}")
    source_tensor = preprocess_image(args.source_image)
    
    logger.info(f"Processing target reference image: {args.target_image}")
    target_tensor = preprocess_image(args.target_image)
    
    # Create batch dictionary
    batch = {
        "source": {
            "image": source_tensor
        },
        "target": {
            "image": target_tensor
        }
    }
    
    # Move to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Move tensors to device
    for domain in batch.values():
        domain["image"] = domain["image"].to(device)
    
    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        output_tensor = model.inference(batch)
    
    # Postprocess output
    output_image = postprocess_image(output_tensor)
    
    # Save output
    output_path = args.output_image if args.output_image else \
                 os.path.splitext(args.source_image)[0] + '_translated.png'
    logger.info(f"Saving translated image to: {output_path}")
    output_image.save(output_path)
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Image Inference')
    parser.add_argument('--source_image', type=str, required=True,
                        help='Path to input source image (e.g., cloudy)')
    parser.add_argument('--target_image', type=str, required=True,
                        help='Path to target reference image (e.g., sunny)')
    parser.add_argument('--output_image', default='output.png', type=str,
                        help='Path to save translated image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model configuration file')
    
    args = parser.parse_args()
    main(args)
