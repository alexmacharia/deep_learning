import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

# Load keras vgg model that was pretrained on ImageNet database
model = vgg16.VGG16()

img = image.load("bay.jpg", target_size=(224,224))

# Convert image to numpy array
x = image.img_to_array(img)

# Add fourth dimension
x = np.expand_dims(x, axis=0)

# Normalize the pixel values to between 0 and 1
x = vgg16.preprocess_input(x)
