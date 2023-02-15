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

for img in not_dog_path.glob("*.png"):
    img = load_img(img)

    img_array = img_to_array(img)

    images.append(img_array)

    labels.append(0)

for img in dog_path.glob("*.png"):
    img = load_img(img)

    img_array = img_to_array(img)

    images.append(img_array)

    labels.append(1)

x_train = np.array(images)

y_train = np.array(labels)

x_train = vgg16.preprocess_input(x_train)

pretrained_nn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64,64,3))

