# GenAI_Hands_on_session_4-5-GAN-
This repository contains an intermediate hands-on that can be used to understand the functioning of GANs

---

### Hands-On for Instructor: Building and Training a GAN for Image Generation

#### Objective:
Students will build and train a Generative Adversarial Network (GAN) to generate images similar to those in a given dataset. They will understand the architecture of GANs, the training process, and common challenges such as mode collapse.

#### Tools:
- TensorFlow or PyTorch
- Python
- Dataset (e.g., MNIST or CIFAR-10)

### Step-by-Step Instructions:

#### Step 1: Setting Up the Environment

Ensure TensorFlow or PyTorch is installed. If not, install via pip:
```bash
pip install tensorflow  # or pip install torch torchvision
```

#### Step 2: Import Necessary Libraries
```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
```

#### Step 3: Load and Prepare the Dataset
```python
# Load the dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

# Batch and shuffle the data
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

#### Step 4: Define the Generator Model
```python
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

generator = make_generator_model()
```

#### Step 5: Define the Discriminator Model
```python
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = make_discriminator_model()
```

#### Step 6: Define the Loss and Optimizers
```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

#### Step 7: Define Training Loop
```python
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Seed for visualizing progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        print ('Epoch {} completed'.format(epoch + 1))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

#### Step 8: Train the GAN
```python
train(train_dataset, EPOCHS)
```

### Explanation and Demonstration

1. **Define Generator Model:** Explain the architecture and function of each layer in the generator.
2. **Define Discriminator Model:** Explain the architecture and function of each layer in the discriminator.
3. **Loss Functions and Optimizers:** Discuss the loss functions used for the generator and discriminator and the role of optimizers.
4. **Training Loop:** Explain the training loop, including how the generator and discriminator are trained in tandem.
5. **Visualizing Progress:** Demonstrate how generated images evolve during training and discuss common challenges such as mode collapse and how they can be addressed.

### Conclusion
By the end of this hands-on project, students will have a practical understanding of how GANs work, including the design and training of both the generator and discriminator, handling of loss functions, and the overall training process. They will also be able to visualize the improvement in generated images over time and understand the key concepts behind generative models.
