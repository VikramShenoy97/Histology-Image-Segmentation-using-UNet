# Histology Image Segmentation using UNet Architecture

Semantic Segmentation implemented using Keras.

## Problem Statement

Develop a machine learning model for identifying cell nuclei from histology images. The model should have the ability to generalize across a variety of lighting conditions, cell types, magnifications, and imaging modalities (bright-field vs. fluorescence).

## Approach

As the problem requires generating binary masks of raw histology images, my first thought was the use of a segmentation technique. There are multiple architectures for segmentation, however, UNet architecture works best for small datasets and is highly computationally efficient.


![Unet](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/unet_architecture.png "UNet Architecture")


The UNet architecture, consists of a contraction path (which is also called an Encoder) and an expanding path (which is also called a Decoder). This network is an end-to-end fully convolutional network, hence, the input to the network can be an image of any size.


The UNet works by downsampling the input image, working in the lower resolution, and then upsampling the image (through the use of Transposed Convolutions for precise localization) to generate the segmented image of proportional size. For better precise locations, the decoder consists of skip connections which basically concatenate the output of the upsampling layer with the feature maps from the encoder at the corresponding level. Finally, a 1x1 convolution is used to reduce the filter dimension and obtain the segmented image.

## Dataset

The dataset for training consists of 590 histology images and its corresponding ground truth label (binary mask).

![Training_Image](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/raw/agxfpoobdlvfpkipcsun.jpg) ![Ground_Truth_Label](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/gt/agxfpoobdlvfpkipcsun.jpg)

Training Image           |  Ground Truth Label
:-------------------------:|:-------------------------:
![](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/raw/agxfpoobdlvfpkipcsun.jpg)  |  ![](https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/blob/master/data/gt/agxfpoobdlvfpkipcsun.jpg)

The dataset for testing consists of 80 histology images similar to the ones provided as training images.

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


#### Content Image - A picture taken by me while on my trip to Amsterdam.

![Content_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Input_Images/Amsterdam.jpg)

#### Style Image - Starry Night by Vincent Van Gogh.

![Style_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Input_Images/Starry_Night.jpg)


### Run

Run the script *test.py* in the terminal as follows.

```
Python test.py
```

## Results
The final output is stored in Output Images.

### Intermediate Stages of Style Transfer

Here is the generated image through different intervals of the run.

![Intermediate_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Output_Images/Intermediate_Images.jpg)

### Transition through epochs

![Transition](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Transition/nst.gif)

### Result of Style Transfer

![Final_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Output_Images/Style_Transfer.jpg)


## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [TensorFlow](https://www.tensorflow.org) - Deep Learning Framework

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is based on **Leon A. Gaty's** paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)
* Project is inspired by **Raymond Yuan's** blog, [*Neural Style Transfer*](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
