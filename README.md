# CS5720-Final-Project
Final two visualizations for CS 5720 (Scientific Computing with Visualization). 

## Visualization One:

## Visualization Two:

### Inspiration:

This project was heavily influenced by the excellent visualization work
of [Utku Ozbulak](github.com/utkuozbulak), in particular, his
[pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
code repository.

### Interpretation:

The goal of this visualization is to help de-mistify the inner workings
of a Convolutional Neural Network. By visualizing attributes such as the
gradient of the error function, and the activation thresholds for
decision units, it becomes easier to understand the internal mechanics
of such classification models.

#### Sample Images:
The proceeding table shows the results of a sample image (column one)
being input through the Convolutional Neural Network [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
which has ben pre-trained on the [ImageNet1000 Database](http://image-net.org/challenges/LSVRC/2014/browse-synsets).
Sample images have been resized to 224 x 224 pixels with three color channels (RGB)
. Each channel has been normalized by subtracting the mean of the channel and dividing by its standard deviation.
This is done for a variety of reasons, one such reason is that normalization
ensures the optimzation algorithm will treat the input parameters (pixels)
with equal importance. Without this step, an image with extreeme pixel
intensities (i.e. day vs night) will be the focus of the optimization process.
We don't want the learned feature detectors to focus primarily on
detecting the presence of extreemly dark or light pixels, we want them to
learn to detect patterns of the pixels themselves, not just their relative intensities.

#### Colored Vanilla Backpropagation:

The images in column two are a colored interpretation of the gradients of the neural network.

The process to produce this image is roughly as follows:
1. Instantiate a pre-trained Alexnet classifier.
2. Feed the input image (column one) through the neural network in a forward pass.
    1. The input image must be one of the thousand different classes comprising [ImageNet1000](http://image-net.org/challenges/LSVRC/2014/browse-synsets).
3. Perform backpropagation, thereby computing the gradients (change in the error function with respect to the network's
weights).
4. Display the gradient of the first layer of the network as an RGB image.
    1. The first layer is chosen because after backpropagation this layer houses the feature detectors most relatable to the original input space.

The colored images can be interpreted as follows:
* A grey color indicates that the input pixel did not impact the error in a large manner.
* A bright color indicates that the input pixel did impact the error in a significant way.

If the error is impacted significantly the gradient will appear to have
more extreeme coloration. In this way we can see what parts of the image
play the most significant role in the networks interpretation of the
input image.

#### Vanilla Backpropagation Saliency:
<!--  Absolute value shows which parts are most important to error function. -->

### Gradient Visualization:
<!-- <table border=0> -->
    <!-- <tbody> -->
    <!-- <tr> -->
        <!-- <td></td> -->
        <!-- <td align='center'>Original Image (Preprocessed):</td> -->
        <!-- <td align='center'>Colored Vanilla Backpropagation:</td> -->
        <!-- <td align='center'>Vanilla Backpropagation Saliency: </td> -->
    <!-- </tr> -->
    <!-- <tr> -->
        <!-- <td width='19%' align='center'>Target Class (211) [Vizsla, Hungarian Pointer]:</td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/211/transformed_original.jpg"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/211/vanilla_bp_color.png"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/211/vanilla_bp_gray.png"></td> -->
    <!-- </tr> -->
    <!-- <tr> -->
        <!-- <td width='19%' align='center'>Target Class (56) [King Snake]:</td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/56/transformed_original.jpg"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/56/vanilla_bp_color.png"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/56/vanilla_bp_gray.png"></td> -->
    <!-- </tr> -->
    <!-- <tr> -->
        <!-- <td width='19%' align='center'>Target Class (243) [Mastiff]:</td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/243/transformed_original.jpg"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/243/vanilla_bp_color.png"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/243/vanilla_bp_gray.png"></td> -->
    <!-- </tr> -->
    <!-- <tr> -->
        <!-- <td width='19%' align='center'>Target Class (281) [Tabby Cat]:</td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/281/transformed_original.jpg"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/281/vanilla_bp_color.png"></td> -->
        <!-- <td width="27%" align='center'><img src="https://raw.githubusercontent.com/ccampell/CS5720-Final-Project/master/results/PyTorchCNNVisualizations/281/vanilla_bp_gray.png"></td> -->
    <!-- </tr> -->
    <!-- </tbody> -->
<!-- </table> -->
