# Histology Image Segmentation using UNet Architecture

Semantic Segmentation implemented using Keras.

## Problem Statement

Develop a machine learning model for identifying cell nuclei from histology images. The model should have the ability to generalize across a variety of lighting conditions, cell types, magnifications, and imaging modalities (bright-field vs. fluorescence).

## Approach

As the problem requires generating binary masks of raw histology images, my first thought was the use of a segmentation technique. There are multiple architectures for segmentation, however, UNet architecture works best for small datasets and is highly computationally efficient.


![Unet](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/unet_architecture.png)*The UNet Architecture from the original paper by O Ronneberger et al., 2015.*


The UNet architecture, consists of a contraction path (which is also called an Encoder) and an expanding path (which is also called a Decoder). This network is an end-to-end fully convolutional network, hence, the input to the network can be an image of any size.


The UNet works by downsampling the input image, working in the lower resolution, and then upsampling the image (through the use of Transposed Convolutions for precise localization) to generate the segmented image of proportional size. For better precise locations, the decoder consists of skip connections which basically concatenate the output of the upsampling layer with the feature maps from the encoder at the corresponding level. Finally, a 1x1 convolution is used to reduce the filter dimension and obtain the segmented image.

## Dataset

The dataset for training consists of 590 histology images and its corresponding ground truth label (binary mask).

Training Image           |  Ground Truth Label
:-------------------------:|:-------------------------:
![](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/raw/agxfpoobdlvfpkipcsun.jpg)  |  ![](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/gt/agxfpoobdlvfpkipcsun.jpg)

The dataset for testing consists of 80 histology images similar to the ones provided as training images.


![Test_Image](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/test_raw/zbfwxtfwwhhmqifdvjjl.jpg)*Test Image*


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project is developed using Python 2.7 and uses the following frameworks / tools:

• Keras


• Google Colab**


• Pandas


• Plotly**


** Google Colab has been used for training the network since training the network on a GPU
is temporally efficient.

** Plotly has only been used in the Graph.py file for visualising the training loss and training accuracy. It isn’t mandatory and does not contribute towards the training and prediction of the network.

### Training
For training the network, there are two approaches.


1) Training the network on your local system
• For training the network on your system, simply run the script train.py in your terminal as:
`Python train.py`
• The script imports the data using import_data.py file.
• The script saves the trained network as trained_model.h5 into the Saved_Model folder (Along with the history.csv file).


2) Training the network on the cloud via Google Colab
• Store the data (gt and raw folders) in a folder called Unet_Data on your Google Drive.
• For training the network on GoogleColab, simply copy the google_colab.py script onto a Google Colab Python 2 Notebook.
• Change the Runtime type to GPU accelerator.
• The trained network will be downloaded as trained_model.h5 which needs to be stored into the Saved_Model folder (Along with the history.csv file).

## Training Performance

The Graph is computed using the history.csv file and by running the graph.py script.

### Training Accuracy vs Training Loss
![Graph](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/Training_Graph.png)*Training Accuracy vs Training Loss Graph*


The model is trained for 20 epochs.
`The final accuracy = 85.315927 %`
`The final loss = 0.122685015`

## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [TensorFlow](https://www.tensorflow.org) - Deep Learning Framework

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Understood the concept of UNet through **Olaf Ronneberger's** paper, [*U- net: Convolutional networks for biomedical image segmentation.*](https://arxiv.org/pdf/1505.04597.pdf)

xkrthiidzdormknuowqh
