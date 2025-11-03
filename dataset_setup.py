import tensorflow as tf
import os
import pathlib

# Path to the dataset (make sure datasets/facades/train and test exist)
dataset_path = pathlib.Path("datasets/facades")

def load_images(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1] // 2

    # Correct split:
    # Left part = real image
    # Right part = sketch (blue sketch)
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    # Convert to float32 and normalize to [-1, 1]
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
    real_image = (tf.cast(real_image, tf.float32) / 127.5) - 1

    return input_image, real_image


# Define dataset paths
train_path = dataset_path / "train"
test_path = dataset_path / "test"

# Create TensorFlow datasets
train_images = tf.data.Dataset.list_files(str(train_path / "*.jpg"), shuffle=True)
train_dataset = train_images.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)

test_images = tf.data.Dataset.list_files(str(test_path / "*.jpg"), shuffle=True)
test_dataset = test_images.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)

# (Optional) Prefetch for performance
train_dataset = train_dataset.batch(1).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

# Print status
print("✅ Local Facades dataset loaded successfully!")
print("Train samples:", len(list(train_images)))
print("Test samples:", len(list(test_images)))
