import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from model_pix2pix import build_generator

# ----------------------------
# Paths
# ----------------------------
checkpoint_path = "checkpoints/generator.weights.h5"
output_dir = "generated_images"
dataset_path = "datasets/facades/test"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Load Generator
# ----------------------------
generator = build_generator()
generator.load_weights(checkpoint_path)
print("✅ Generator weights loaded successfully!")

# ----------------------------
# Helper: Load test image
# ----------------------------
def load_test_image(img_path, img_size=(256, 256)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    w = tf.shape(img)[1] // 2

    # Correct split: left = real image, right = sketch
    target_img = img[:, :w, :]    # real image
    input_img = img[:, w:, :]     # sketch

    input_img = tf.image.resize(input_img, img_size)
    target_img = tf.image.resize(target_img, img_size)

    # Normalize to [-1, 1]
    input_img = (tf.cast(input_img, tf.float32) / 127.5) - 1
    target_img = (tf.cast(target_img, tf.float32) / 127.5) - 1

    return input_img, target_img

# ----------------------------
# Generate and Save Outputs
# ----------------------------
for f in os.listdir(dataset_path):
    if not f.lower().endswith(('.jpg', '.png')):
        continue
    img_path = os.path.join(dataset_path, f)
    input_img, target_img = load_test_image(img_path)
    input_img = tf.expand_dims(input_img, axis=0)  # Add batch dimension

    # Generate output
    gen_output = generator(input_img, training=False)
    gen_image = (gen_output[0] + 1) * 127.5  # Denormalize to [0,255]

    save_path = os.path.join(output_dir, f"gen_{f}")
    save_img(save_path, gen_image.numpy())
    print(f"🖼️ Saved generated image: {save_path}")
