import cv2
import os
import numpy as np

# Dataset paths
real_path = r"C:\Users\Geresshvaran\Downloads\dataset\real"
fake_path = r"C:\Users\Geresshvaran\Downloads\dataset\fake"

# Image size for AI model
IMG_SIZE = 224

# Lists to store images and labels
data = []
labels = []

# Function to process images
def load_images(folder_path, label):

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        # Read image
        image = cv2.imread(image_path)

        # Skip if image not loaded
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Normalize image
        image = image / 255.0

        # Add image and label
        data.append(image)
        labels.append(label)

# Load real images
load_images(real_path, 0)

# Load fake images
load_images(fake_path, 1)

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Print dataset info
print("Dataset Loaded Successfully")
print(f"Total Images: {len(data)}")
print(f"Data Shape: {data.shape}")
print(f"Labels Shape: {labels.shape}")