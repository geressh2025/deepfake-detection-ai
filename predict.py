import cv2
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/deepfake_model.keras")

# Image size
IMG_SIZE = 224

# Load test image
image = cv2.imread("testing/00JEP4Z36Z.jpg")

# Check image
if image is None:
    print("Error: Image not found")
    exit()

# Resize image
image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

# Normalize image
image_normalized = image_resized / 255.0

# Convert into array
image_array = np.array(image_normalized)

# Expand dimensions
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array)

# Get probability
probability = prediction[0][0]

print(f"\nPrediction Score: {probability:.4f}")

# Classification
if probability >= 0.5:
    print("Result: FAKE IMAGE")
else:
    print("Result: REAL IMAGE")

# Display image
cv2.imshow("Test Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()