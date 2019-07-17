import numpy as np
from import_data import load_data
from PIL import Image
import keras
from keras.models import Model, load_model

# Load trained unet model
loaded_model = load_model("Saved_Model/trained_model.h5")
loaded_model.set_weights(loaded_model.get_weights())
# Retrieve raw test images, label_name, height, and width.
images, label_names, height, width = load_data(mode="Test")
for i in range(len(images)):
    img = images[i]
    # (height, width, channels) -> (1, height, width, channels)
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img, verbose=1)
    prediction = np.squeeze(prediction)
    # Generate binary mask by rounding up values.
    prediction = np.round(prediction)
    prediction = prediction * 255.
    # Generate image.
    img = Image.fromarray(prediction)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize the image to original size.
    img = img.resize((width[i], height[i]), Image.ANTIALIAS)
    print img.size
    img.save(label_names[i])
