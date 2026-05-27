import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Dataset paths
real_path = r"C:\Users\Geresshvaran\Downloads\dataset1\real"
fake_path = r"C:\Users\Geresshvaran\Downloads\dataset1\fake"

# Image size
IMG_SIZE = 224

# Data containers
data = []
labels = []

# Function to load images
def load_images(folder_path, label):

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        # Read image
        image = cv2.imread(image_path)

        # Skip bad images
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Normalize image
        image = image / 255.0

        # Add to dataset
        data.append(image)
        labels.append(label)

# Load datasets
load_images(real_path, 0)
load_images(fake_path, 1)

# Convert to arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# Build CNN model
model = Sequential()

# First convolution layer
model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(224, 224, 3)
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer
model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Dropout layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save trained model
model.save("models/deepfake_model.keras")

print("\nModel saved successfully!")