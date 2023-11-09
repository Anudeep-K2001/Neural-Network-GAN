# Kolluri, Anudeep
# 1002_116_426
# 2023_11_08
# Assignment_04_01


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Conv2DTranspose, Reshape


def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax


class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator model at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """

    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images*127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images",
                    f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()


def build_discriminator():

    input_shape = input_shape = (28, 28, 1)

    model = tf.keras.models.Sequential()
    # your code here

    model.add(Conv2D(16, (5, 5), strides=(2, 2),
              padding='same', input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def build_generator():

    model = tf.keras.models.Sequential()
    # your code here

    model.add(Dense(7*7*8, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 8)))
    model.add(Conv2DTranspose(8, (5, 5), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, (5, 5), strides=(
        2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(
        2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):

        output = tf.concat([real_output, fake_output], axis=0)
        labels = tf.concat(
            [tf.ones_like(real_output), tf.zeros_like(fake_output)], axis=0)

        loss = self.loss_fn(labels, output)
        return loss

    @tf.function
    def train_step(self, data):
        """
        This method takes a batch of real images (data) and generates fake images using the generator model.
        It then computes the loss for both generator and discriminator models and updates the model weights
        using their respective optimizers.

        By implementing this method, you are overriding the default train_step method of the tf.keras.Model class.
        This allows us to train a GAN using model.fit()

        You can adapt the train_step function from the tensorflow DCGAN tutorial
        https://www.tensorflow.org/tutorials/generative/dcgan

        However, You must make the following changes (IMPORTANT!):
        - Instead of using a normal distribution to generate the noise vector, use a uniform distribution (default params in tensorflow)
        - Instead of separately calculating the discriminator loss for real and fake images and adding them together,
          combine the real and fake images into a batch, label them real and fake, and calculate the loss on the entire batch.
          (This is equivalent to doing it separately as done in the tutorial, and giving each 0.5 weight), either one should pass the tests.

        Args:
            data: a batch of real images

        Returns:
            dict[str, tf.Tensor]: A dictionary containing the generator loss ('g_loss') and
                discriminator loss ('d_loss') for the current training step.
        """

        batch_size = tf.shape(data)[0]
        noise_dim = 100
        # TRAINING CODE START HERE
        noise = tf.random.uniform([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # TRAINING CODE END HERE
        return {"d_loss": d_loss, "g_loss": g_loss}


def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")

    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()

    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(
        generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64,
              callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images*127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')


if __name__ == "__main__":
    train_dcgan_mnist()
