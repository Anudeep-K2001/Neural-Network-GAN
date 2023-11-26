# Deep Convolutional Generative Adversarial Network (DCGAN) for MNIST Dataset

## Overview

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating images from the MNIST dataset using TensorFlow and Keras. The primary goal is to modify specific functions to build and train the DCGAN, as well as to understand the provided testing framework.

## Instructions

### Modifying Functions

1. **build_discriminator:**

   Modify the `build_discriminator` function to construct the discriminator as described in the comments and return it.

2. **build_generator:**

   Modify the `build_generator` function to build the generator as described in the comments and return it.

3. **train_step in the DCGAN class:**

   Implement the training logic for GANs in the `train_step` method of the DCGAN class. By implementing it as a subclass of Keras's Model class, GANs can be trained using Keras's `model.fit()` API.

### Training

The code to load the data and run training is already implemented in the `train_dcgan_mnist()` function. This function is not part of the grading, but you can use it to play with DCGANs and observe the generated images.

- The function may take a long time to train for 50 epochs on the full MNIST dataset (possibly an hour or two).

- You can modify the number of samples you train on or the batch size to make training quicker. Note that if you reduce the size of the dataset, you may need to increase the number of epochs to get comparable results.

### Helper Functions

- `plot_images` is a provided helper function that arranges images into a row/grid. It can be used to visualize generated images.

- `GenerateSamplesCallback` is a provided class that generates images using your generator after every epoch, using the same input values.

### Generated Images

After running the `train_dcgan_mnist` code, check the `generated_images` subdirectory for examples of generated images.

- After one epoch of training on the full dataset, you should expect to see `generated_images_0.png`.

- After ten epochs of training on the full dataset, you should expect to see `generated_images_9.png`.

## Testing

The `Assignment_04_tests.py` file includes three tests:

1. `test_generator`: Tests your generator architecture.
2. `test_discriminator`: Tests your discriminator architecture.
3. `test_dcgan_train_step`: Tests the GAN training logic.

Ensure your implementation passes these tests using the following command:

```bash
py.test --verbose Assignment_04_tests.py
