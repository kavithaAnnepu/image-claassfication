# PyTorch image-claassfication

This repo contains tutorials covering image classification using PyTorch 1.7, torchvision 0.8, matplotlib 3.3 and scikit-learn 0.24, with Python 3.8. We'll start by implementing a multilayer perceptron (MLP) and then move on to architectures using convolutional neural networks (CNNs). Specifically, we'll implement LeNet, AlexNet, VGG for bengali-CFY daset of digit classification.

Tutorials 

1-Multilayer Perceptron

This tutorial provides an introduction to PyTorch and TorchVision. We'll learn how to: load datasets(preprocessing if cudtom dataset), augment data, define a multilayer perceptron (MLP), train a model, view the outputs of our model, visualize the model's representations, and view the weights of the model. The experiments will be carried out on the Bengali-CFY dataset - a set of 28x28 handwritten digits.

2 - LeNet

In this tutorial we'll implement the classic LeNet architecture. We'll look into convolutional neural networks and how convolutional layers and subsampling (aka pooling) layers work.

3 - AlexNet

In this tutorial we will implement AlexNet, the convolutional neural network architecture that helped start the current interest in deep learning. We will move on to the Bengali-CFY dataset - 32x32 color images in ten classes. We show: how to define architectures using nn.Sequential, how to initialize the parameters of your neural network, and how to use the learning rate finder to determine a good initial learning rate.

4 - VGG

This tutorial will cover implementing the VGG model. However, instead of training the model from scratch we will instead load a VGG model pre-trained on the ImageNet dataset and show how to perform transfer learning to adapt its weights to the Bengali-CFY dataset using a technique called discriminative fine-tuning. We'll also explain how adaptive pooling layers and batch normalization works.

References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

https://github.com/pytorch/tutorials

https://github.com/pytorch/examples

https://colah.github.io/posts/2014-10-Visualizing-MNIST/

https://distill.pub/2016/misread-tsne/

https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

https://github.com/activatedgeek/LeNet-5
