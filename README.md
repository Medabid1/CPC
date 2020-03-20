# CPC
Implementation of [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) Paper.

We Split MNIST images into 3x3 grid, composed of 9, 14x14 images with 50% overlap.

For the autoregressive part we use GRU cell and we predict 5 steps in the future. 

[CPC on Image](https://raw.githubusercontent.com/Medabid1/CPC/master/imgs/vision.png)

###Usage

The Encoder weights are saved after each train using the `train.py` file.
We train a classifier with one hidden layer on top of the features spaces of the freezed Encoder in `classifier.py` file.

The classifier reach 85% accuracy on mnist.

###ToDo :
- [] make the contrastive predicition row wise.
- [] train on cifar-10