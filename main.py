import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications import vgg16

# Load keras vgg model that was pretrained on ImageNet database
model = vgg16.VGG16()

img = load_img("bay.jpg", target_size=(224,224))

# Convert image to numpy array
x = img_to_array(img)

# Add fourth dimension
x = np.expand_dims(x, axis=0)

# Normalize the pixel values to between 0 and 1
x = vgg16.preprocess_input(x)

# Predict
predictions = model.predict(x)


# Look up the names of the predicted classes
predicted_classes = vgg16.decode_predictions(predictions)

print("Top predictions for this image:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))
