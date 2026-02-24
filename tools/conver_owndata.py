import os
import random
import shutil
from pathlib import Path


def get_image_files(directory):
    """Get all image files from a directory (recursively)."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.rglob(f'*{ext}'))
    return image_files


def create_output_directories(output_dirs):
    """Create output directories if they don't exist."""
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def split_data(images, train_ratio=0.7, test_ratio=0.2):
    """Split data into train, test, and query sets."""
    # Shuffle the images randomly
    random.shuffle(images)
    
    # Calculate split sizes
    total = len(images)
    train_size = int(total * train_ratio)
    test_size = int(total * test_ratio)
    query_size = total - train_size - test_size
    
    # Split the images
    train_images = images[:train_size]
    test_images = images[train_size:train_size+test_size]
    query_images = images[train_size+test_size:]
    
    return train_images, test_images, query_images


def copy_images(images, output_dir):
    """Copy images to output directory."""
    for img_path in images:
        # Check if output directory is query
        if output_dir.name == "query":
            # Replace 'c1' with 'c2' in filename
            modified_filename = img_path.name.replace("c1", "c2")
            output_path = output_dir / modified_filename
        else:
            # Use the original filename
            output_path = output_dir / img_path.name
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_path)
        print(f"Copied {img_path} to {output_path}")


def create_own_market1501_datasets():
    """Main function to split and organize data."""
    # Define input path (converted to Linux format)
    INPUT_DIR = Path('/mnt/e/datasets/reid/market1501/248')
    
    # Output directories
    OUTPUT_DIR = Path('/mnt/e/datasets/reid/market1501/market1501')
    OUTPUT_TRAIN = OUTPUT_DIR / 'bounding_box_train'
    OUTPUT_TEST = OUTPUT_DIR / 'bounding_box_test'
    OUTPUT_QUERY = OUTPUT_DIR / 'query'
    
    # Create output directories
    create_output_directories([OUTPUT_TRAIN, OUTPUT_TEST, OUTPUT_QUERY])
    
    # Get all images from input directory
    all_images = get_image_files(INPUT_DIR)
    
    print(f"Total images found: {len(all_images)}")
    
    # Split the data
    train_images, test_images, query_images = split_data(all_images)
    
    print(f"Split sizes: Train={len(train_images)}, Test={len(test_images)}, Query={len(query_images)}")
    
    # Copy images to respective output directories
    print("\nCopying train images...")
    copy_images(train_images, OUTPUT_TRAIN)
    
    print("\nCopying test images...")
    copy_images(test_images, OUTPUT_TEST)
    
    print("\nCopying query images...")
    copy_images(query_images, OUTPUT_QUERY)
    
    print("\nSplit completed successfully!")
    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(test_images)} images")
    print(f"Query: {len(query_images)} images")
    print(f"Output directory: {OUTPUT_DIR}")

def change_filename_pid(directory, new_pid="0001"):
    """Change the person ID (PID) in filenames to a custom value."""
    # Get all image files
    image_files = get_image_files(directory)
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using custom PID: {new_pid}")
    
    # Process each image file
    for img_path in image_files:
        # Extract filename
        filename = img_path.name
        
        # Split the filename into parts
        parts = filename.split('_')
        
        if len(parts) >= 2:
            # Reconstruct the filename with custom PID
            new_filename = f"{new_pid}_{'_'.join(parts[1:])}"
            
            # Create new path
            new_path = img_path.parent / new_filename
            
            # Rename the file
            img_path.rename(new_path)
            print(f"Renamed {filename} to {new_filename}")
        else:
            print(f"Skipping {filename} - invalid format")
    
    print("\nFilename PID modification completed!")


if __name__ == "__main__":
    # Option 1: Run dataset splitting
    create_own_market1501_datasets()
    
    # Option 2: Run filename PID modification with custom PID
    # MODIFY_DIR = Path('/mnt/e/datasets/reid/market1501/30')
    # change_filename_pid(MODIFY_DIR, new_pid="0030")  # Use custom PID "0248"

