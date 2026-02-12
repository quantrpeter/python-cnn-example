import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import mnist # A helpful library for loading the dataset

# --- 1. Load and Preprocess Data ---
# Keras can load popular datasets; here we use the `mnist` library for simplicity
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize pixel values from [0, 255] to [-0.5, 0.5]
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape images to add a single color channel dimension (required for CNNs)
# Input shape becomes (num_samples, height, width, channels) -> (60000, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# One-hot encode the labels (e.g., 2 becomes [0, 0, 1, 0, ...])
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# --- 2. Build the CNN Model ---
model = Sequential([
    # First convolutional layer
    Conv2D(filters=8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer to reduce dimensionality
    MaxPooling2D(pool_size=(2, 2)),
    # Second set of layers
    Conv2D(filters=16, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output to feed into dense layers
    Flatten(),
    # Fully connected dense layer for classification
    Dense(units=10, activation='softmax'), # 10 units for 10 digit classes
])

# --- 3. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Appropriate loss for multi-class classification
    metrics=['accuracy'],
)

# --- 4. Train the Model ---
model.fit(
    train_images,
    train_labels_one_hot,
    epochs=3, # Can increase epochs for better accuracy
    validation_data=(test_images, test_labels_one_hot),
)

# --- 5. Evaluate the Model (Optional) ---
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels_one_hot
)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")