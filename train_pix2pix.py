import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model_pix2pix import build_generator, build_discriminator

# ----------------------------
# Dataset Loader for Facades
# ----------------------------
def load_facades_dataset(folder_path, img_size=(256, 256)):
    input_images = []
    target_images = []

    for f in os.listdir(folder_path):
        if not f.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(folder_path, f)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img)
        w = tf.shape(img)[1] // 2

        # Correct split: left = real image (target), right = sketch (input)
        target_img = img[:, :w, :]
        input_img = img[:, w:, :]

        # Resize and normalize
        input_img = tf.image.resize(input_img, img_size)
        target_img = tf.image.resize(target_img, img_size)
        input_img = (tf.cast(input_img, tf.float32) / 127.5) - 1
        target_img = (tf.cast(target_img, tf.float32) / 127.5) - 1

        input_images.append(input_img)
        target_images.append(target_img)

    dataset = tf.data.Dataset.from_tensor_slices((input_images, target_images))
    dataset = dataset.shuffle(400).batch(1)
    return dataset


# ----------------------------
# Losses
# ----------------------------
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)


# ----------------------------
# Training Setup
# ----------------------------
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss


# ----------------------------
# Train Function
# ----------------------------
def train(dataset, epochs, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step, (input_image, target) in enumerate(dataset):
            gen_loss, disc_loss = train_step(input_image, target)
            if step % 10 == 0:
                print(f"Step {step}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

        # Save model weights after each epoch
        generator.save_weights(os.path.join(checkpoint_dir, "generator.weights.h5"))
        discriminator.save_weights(os.path.join(checkpoint_dir, "discriminator.weights.h5"))
        print(f"✅ Saved model weights for epoch {epoch+1}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    train_dataset = load_facades_dataset("datasets/facades/train")
    train(train_dataset, epochs=150)
