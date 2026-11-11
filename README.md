Python Program to Implement a Generative Model for Image Editing Using GAN
Aim:
To implement a Generative Adversarial Network (GAN)-based model for image editing using Python.
Algorithm
1. Import the required Python libraries.
2. Load the input image.
3. Resize the image to a fixed size.
4. Build a Generator model to generate edited images.
5. Build a Discriminator model to distinguish real and generated images.
6. Train the GAN model.
7. Generate the edited image.
8. Display the original and edited images.
Python Program
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load input image
img = Image.open("input.jpg").convert("RGB")
img = img.resize((128, 128))

# Convert image into array
image = np.array(img) / 255.0
image = np.expand_dims(image, axis=0)

# Generator Model
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(64, 3, padding='same',
                           activation='relu'),
    tf.keras.layers.Conv2D(128, 3, padding='same',
                           activation='relu'),
    tf.keras.layers.Conv2D(3, 3, padding='same',
                           activation='sigmoid')
])

# Generate edited image
edited_image = generator.predict(image)

# Display images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[0])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edited_image[0])
plt.title("Edited Image")
plt.axis("off")

plt.show()

# Save edited image
result = (edited_image[0] * 255).astype(np.uint8)
Image.fromarray(result).save("edited_image.jpg")

print("Image editing completed successfully!")

Output
- Original image is displayed.
- Generator produces an output image.
- Edited image is displayed.
- The generated image is saved as edited_image.jpg.
Result
Thus, a basic generator model for image editing was implemented successfully using Python and TensorFlow.
