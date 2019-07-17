import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import keras
from google.colab import files
from keras.models import Model
from keras.layers import *
from keras import initializers
from keras import optimizers
from google.colab import drive
from keras.preprocessing import image

def import_data():
    drive.mount('/content/drive')
    images_folder = "drive/My Drive/Unet_Data/raw"
    labels_folder = "drive/My Drive/Unet_Data/gt"

    filenames = [os.path.join(images_folder,file) for file in os.listdir(images_folder) if file.endswith(".jpg")]
    images = []
    labels = []
    for f in filenames:
        img = Image.open(f)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        img = img / 255.
        images.append(img)
        id = re.search("drive/My Drive/Unet_Data/raw/(.+).jpg", f).group(1)
        label_file = labels_folder+"/"+id+".jpg"
        label = Image.open(label_file)
        label = label.resize((256, 256), Image.ANTIALIAS)
        label = np.array(label, dtype=np.float32)
        label = np.expand_dims(label, axis=3)
        label = label / 255.
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def unet_model():
    input_layer = Input(shape=(None, None, 3))
    convolutional_layer_1 = Conv2D(filters=64, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(input_layer)
    convolutional_layer_2 = Conv2D(filters=64, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_1)
    pooling_layer_1 = MaxPooling2D(pool_size=[2,2], strides=2)(convolutional_layer_2)
    convolutional_layer_3 = Conv2D(filters=128, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(pooling_layer_1)
    convolutional_layer_4 = Conv2D(filters=128, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_3)
    pooling_layer_2 = MaxPooling2D(pool_size=[2,2], strides=2)(convolutional_layer_4)
    convolutional_layer_5 =  Conv2D(filters=256, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(pooling_layer_2)
    convolutional_layer_6 = Conv2D(filters=256, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_5)
    pooling_layer_3 = MaxPooling2D(pool_size=[2,2], strides=2)(convolutional_layer_6)
    convolutional_layer_7 = Conv2D(filters=512, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(pooling_layer_3)
    convolutional_layer_8 = Conv2D(filters=512, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_7)
    pooling_layer_4 = MaxPooling2D(pool_size=[2,2], strides=2)(convolutional_layer_8)
    convolutional_layer_9 = Conv2D(filters=1024, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(pooling_layer_4)
    convolutional_layer_10 = Conv2D(filters=1024, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_9)
    transpose_convolutional_layer_1 = Conv2DTranspose(filters=512, kernel_size=[2,2], strides=(2,2), kernel_initializer=initializers.he_normal(seed=1))(convolutional_layer_10)
    merged_layer_1 = concatenate([convolutional_layer_8, transpose_convolutional_layer_1], axis=3)
    convolutional_layer_11 = Conv2D(filters=512, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(merged_layer_1)
    convolutional_layer_12 = Conv2D(filters=512, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_11)
    transpose_convolutional_layer_2 = Conv2DTranspose(filters=256, kernel_size=[2,2], strides=(2,2), kernel_initializer=initializers.he_normal(seed=1))(convolutional_layer_12)
    merged_layer_2 = concatenate([convolutional_layer_6, transpose_convolutional_layer_2], axis=3)
    convolutional_layer_13 = Conv2D(filters=256, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(merged_layer_2)
    convolutional_layer_14 = Conv2D(filters=256, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_13)
    transpose_convolutional_layer_3 = Conv2DTranspose(filters=128, kernel_size=[2,2], strides=(2,2), kernel_initializer=initializers.he_normal(seed=1))(convolutional_layer_14)
    merged_layer_3 = concatenate([convolutional_layer_4, transpose_convolutional_layer_3], axis=3)
    convolutional_layer_15 = Conv2D(filters=128, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(merged_layer_3)
    convolutional_layer_16 = Conv2D(filters=128, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_15)
    transpose_convolutional_layer_4 = Conv2DTranspose(filters=64, kernel_size=[2,2], strides=(2,2), kernel_initializer=initializers.he_normal(seed=1))(convolutional_layer_16)
    merged_layer_4 = concatenate([convolutional_layer_2, transpose_convolutional_layer_4], axis=3)
    convolutional_layer_17 = Conv2D(filters=64, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(merged_layer_4)
    convolutional_layer_18 = Conv2D(filters=64, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_17)
    convolutional_layer_19 = Conv2D(filters=2, kernel_size=[3,3], kernel_initializer=initializers.he_normal(seed=1), activation="relu", padding="same")(convolutional_layer_18)
    convolutional_layer_20 = Conv2D(filters=1, kernel_size=[1,1], activation="sigmoid")(convolutional_layer_19)
    model = Model(input=input_layer, output=convolutional_layer_20)
    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    return model

images, labels = import_data()
model = unet_model()
print model.summary()
pd.DataFrame(model.fit(images,labels, epochs=20, verbose=1).history).to_csv("history.csv")
model.save("trained_model.h5")
files.download("trained_model.h5")
files.download("history.csv")
