# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

### Problem Statement

The objective is to build a Convolutional Autoencoder (CAE) to denoise images from the MNIST dataset. The MNIST dataset contains grayscale images of handwritten digits (0-9) in a 28x28 pixel format. The challenge is to:

1. **Add Noise**: Introduce random noise to the images, effectively corrupting them.
2. **Denoise the Images**: Train the autoencoder to reconstruct clean images from the noisy input by learning compressed and efficient representations of the image data.
3. **Evaluate Performance**: Compare the original images, noisy images, and the reconstructed (denoised) images to assess the autoencoder’s ability to remove noise.

The focus is on achieving an efficient encoding-decoding mechanism using convolutional layers, which are particularly well-suited for image data.

### Dataset

![image](https://github.com/user-attachments/assets/08ccfae2-8025-458c-8e2b-942785afac00)


## Convolution Autoencoder Network Model

![image](https://github.com/user-attachments/assets/1bf17af9-e4c2-4b86-8cea-693c1d1d0a90)


## DESIGN STEPS

### STEP 1:
Data Preparation: Load MNIST dataset, normalize the pixel values to [0, 1], and reshape the images to (28, 28, 1).


### STEP 2:
 Add Gaussian noise to the images and clip the values between 0 and 1 to create noisy versions of the dataset

### STEP 3:
Build a Convolutional Autoencoder with convolutional and pooling layers for encoding, and upsampling layers for decoding to reconstruct clean images from noisy inputs.

## PROGRAM
### Name: B VIJAY KUMAR
### Register Number: 212222230173

```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

```

```
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
```

```
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

```
```
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

```
n = 5
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
```
input_image = keras.Input(shape = (28,28,1))

x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_image)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(16, (3,3), activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_image, decoded)

```

```
print('Name:   B VIJAY KUMAR        Register Number:    212222230173    ')
autoencoder.summary()

```

```
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
decoded_imgs = autoencoder.predict(x_test_noisy)

```

```
n = 5
print('Name:    B VIJAY KUMAR       Register Number:    212222230173    ')
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


```

```
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/f0d791a8-95ce-4faa-a8ba-4e0deade1c83)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/80ec0c83-e1e8-44ef-86d9-d95757cffbbf)


## RESULT

The convolutional autoencoder successfully reconstructed noisy MNIST images, effectively reducing noise while preserving key details. Visual comparisons show significant improvement from the noisy input to the denoised output.

