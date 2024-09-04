import os
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm

# Get the current working directory and set up paths
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

input_dir = os.path.join('..', 'input', 'asl_alphabet_train', 'asl_alphabet_train')
output_dir = os.path.join('..', 'input', 'preprocessed_images')
csv_file = os.path.join('..', 'input', 'image_labels.csv')

print(f"Input directory: {os.path.abspath(input_dir)}")
print(f"Output directory: {os.path.abspath(output_dir)}")
print(f"CSV file: {os.path.abspath(csv_file)}")

# Function to preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(target_size)
        img = img.convert('RGB')
    return img

# Get sorted list of subdirectories (labels)
labels = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
print(f"Found labels: {labels}")

# Prepare CSV file
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])

    # Process images
    for label_idx, label in enumerate(tqdm(labels, desc="Processing labels", unit="label"), start=1):
        label_input_dir = os.path.join(input_dir, label)
        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        # Get all image files in the directory
        image_files = [img for img in os.listdir(label_input_dir) 
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, img_file in enumerate(tqdm(image_files, desc=f"Processing {label}", unit="image", leave=False)):
            input_path = os.path.join(label_input_dir, img_file)
            img = preprocess_image(input_path)

            base_filename = f"{label}_{idx}"
            output_filename = f"{base_filename}.jpg"
            output_path = os.path.join(label_output_dir, output_filename)
            
            # Save preprocessed image
            img.save(output_path)
            writer.writerow([output_path, label_idx])

print("Preprocessing complete. CSV file created.")
