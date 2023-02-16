from keras.models import model_from_json
from pathlib import Path
from keras.utils import load_img, img_to_array
import numpy as np
from keras.applications import vgg16

# Load the model's struture
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the model object
model = model_from_json(model_structure)

# Load the model weights
model.load_weights("model_weights.h5")

# Load image to test
img = load_img("not_dog.png", target_size=(64, 64))

# Convert the image to a numpy array
image_array = img_to_array(img)

# Add fourth dimension to the image
images = np.expand_dims(image_array, axis=0)

# Normalize the data
images = vgg16.preprocess_input(images)

# Extract features from test image the same as in training
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)

# Make prediction
results = model.predict(features)

# Check first result of first element
single_result = results[0][0]

# Print the result
print("Likelihood that the image contains a dog: {}%".format(int(single_result * 100)))




