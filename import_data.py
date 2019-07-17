import numpy as np
from PIL import Image
import os
import re

def load_data(mode):
    # Takes the mode as Input
    # mode = "Train", returns the raw training images and corresponding ground_truth images.
    # mode = "Test", returns the raw test images, label name, image height, and image width.
    if mode == "Train":
        images_folder = "data/raw"
        labels_folder = "data/gt"
        # Get all the image files from the folder.
        filenames = [os.path.join(images_folder,file) for file in os.listdir(images_folder) if file.endswith(".jpg")]
        images = []
        labels = []
        for f in filenames:
            img = Image.open(f)
            # Resize images to 256x256.
            img = img.resize((256, 256), Image.ANTIALIAS)
            img = np.array(img, dtype=np.float32)
            # Normalize the images.
            img = img / 255.
            images.append(img)
            # Get the corresponding ground truth image.
            id = re.search("data/raw/(.+).jpg", f).group(1)
            label_file = labels_folder+"/"+id+".jpg"
            label = Image.open(label_file)
            label = label.resize((256, 256), Image.ANTIALIAS)
            label = np.array(label, dtype=np.float32)
            # Expand the dimensions to include the number of channels.
            label = np.expand_dims(label, axis=3)
            label = label / 255.
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    if mode == "Test":
        images_folder = "test_raw"
        labels_folder = "test_gt"
        if not os.path.isdir("test_gt"):
            os.makedirs("test_gt")
        filenames = [os.path.join(images_folder,file) for file in os.listdir(images_folder) if file.endswith(".jpg")]
        images = []
        label_name = []
        height = []
        width = []
        for f in filenames:
            img = Image.open(f)
            # Save the width and height of the image.
            width.append(img.size[0])
            height.append(img.size[1])
            # Resize images to 256x256.
            img = img.resize((256, 256), Image.ANTIALIAS)
            img = np.array(img, dtype=np.float32)
            # Normalize the images.
            img = img / 255.
            images.append(img)
            id = re.search("test_raw/(.+).jpg", f).group(1)
            label_file = labels_folder+"/"+id+".jpg"
            # Save the label_name for each test image.
            label_name.append(label_file)

        images = np.array(images)
        label_name = np.array(label_name)
        height = np.array(height)
        width = np.array(width)
        return images, label_name, height, width
