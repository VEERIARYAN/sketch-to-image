import tensorflow as tf
from tensorflow.keras import layers

# ----------------------------
# Generator (U-Net style, simple version)
# ----------------------------
def build_generator():
    inputs = layers.Input(shape=[256, 256, 3])

    # Encoder
    down1 = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(inputs)
    down2 = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(down1)

    # Decoder
    up1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(down2)
    up2 = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(up1)

    return tf.keras.Model(inputs=inputs, outputs=up2, name="Generator")


# ----------------------------
# Discriminator (PatchGAN)
# ----------------------------
def build_discriminator():
    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    tar = layers.Input(shape=[256, 256, 3], name='target_image')

    x = layers.Concatenate()([inp, tar])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x, name="Discriminator")
