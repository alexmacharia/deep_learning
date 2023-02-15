from pathlib import Path
import numpy as np
import joblib
from keras.utils import load_img, img_to_array
from keras.applications import vgg16

# Path to folders with training data

dog_path = Path("training_data") / "dogs"
not_dog_path = Path("training_data") / "not_dogs"

images = []
labels = []

# Load not dog images
for img in not_dog_path.glob("*.png"):
    # Load image
    img = load_img(img)

    # Convert image to array
    img_array = img_to_array(img)

    # Add image to list of images
    images.append(img_array)

    # Add label 0 for not dog
    labels.append(0)

for img in dog_path.glob("*.png"):
    # Load image
    img = load_img(img)

    # Convert iamge to numpy array
    img_array = img_to_array(img)

    # Add image to list of images
    images.append(img_array)

    # Add label 1 for dog
    labels.append(1)

# Convert list to array
x_train = np.array(images)

# Convert list of labels to array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64,64,3))

# Extract features for each image in one pass
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

#Save matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")