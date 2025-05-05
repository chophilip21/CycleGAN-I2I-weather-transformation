import os
import shutil
from pathlib import Path

def create_sequence_structure(source_dir, target_dir, dataset_name):
    """
    Create the new data structure with sequences from the source directory.
    
    Args:
        source_dir (str): Path to the source directory containing train/val folders
        target_dir (str): Path to the target directory where new structure will be created
        dataset_name (str): Name of the dataset
    """
    # Create the new directory structure
    sequences_dir = Path(target_dir) / dataset_name / 'sequences'
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train and val directories
    for split in ['train', 'val']:
        split_dir = sequences_dir / split / 'frames'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get source images for this split
        source_split_dir = Path(source_dir) / split
        if not source_split_dir.exists():
            print(f"Warning: Source split directory {source_split_dir} not found")
            continue
            
        # Group images by sequence (assuming frame numbers are sequential)
        images = sorted([f for f in source_split_dir.glob('*.jpg')])
        
        if not images:
            print(f"No images found in {source_split_dir}")
            continue
            
        # Create sequences based on frame numbers
        current_sequence = 1
        current_sequence_dir = split_dir / f'sequence{current_sequence}'
        current_sequence_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            # Extract frame number from filename (remove 'frame' prefix and convert to int)
            frame_num = int(img_path.stem.split('_')[-1].replace('frame', ''))
            
            # Create new sequence directory if needed
            if frame_num % 100 == 0:  # Create new sequence every 100 frames
                current_sequence += 1
                current_sequence_dir = split_dir / f'sequence{current_sequence}'
                current_sequence_dir.mkdir(exist_ok=True)
            
            # Copy the image to the sequence directory
            target_path = current_sequence_dir / f'sequence{current_sequence}_frame{frame_num}_info.png'
            shutil.copy(img_path, target_path)
            
            print(f"Moved {img_path} to {target_path}")

def main():
    # Define paths
    source_dir = '/home/philip-ubuntu/Weather_correction/feamgan/data/raw/target'
    target_dir = '/home/philip-ubuntu/Weather_correction/feamgan/data'
    dataset_name = 'sunny'
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} not found")
        return
    
    # Create the new structure
    create_sequence_structure(source_dir, target_dir, dataset_name)
    print("Data structure conversion complete!")

if __name__ == "__main__":
    main()
