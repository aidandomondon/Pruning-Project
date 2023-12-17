# Pruning Project

## Overall Goal

The goal is to (1) train a convolutional neural network to classify articles of clothing, (2) calculate and prune the bottom 0.01<sup>th</sup> percentile of weights when ranked by magnitude, and (3) evaluate the performance of the pruned network.

Exploring the use of weight magnitude as a pruning metric on this concrete example problem will illuminate whether it can be useful for other, more complex image recognition problems that require larger neural networks.

## Motivation

This project was motivated by a desire to become more familiar with PyTorch. While I had seen PyTorch code before, I had not written substantive original code with PyTorch and thus only had a tenuous grasp of how machine learning implementations are created with it.

## Background

***Image Classification***

Our definition of an image classification problem is as follows. Let there be a sample $X$ of images, each of dimension $d \times d$, with corresponding labels $y$. Partition $X$ into $X_{train}$ and $X_{test}$. Design a model that can, given $X_{train}$ and the labels corresponding to $X_{train}$'s images, $\\{ y_i \\; | x_i \in X_{train}\\}$, accurately predict the correct label $y_j$ of an image $x_j$ in $X_{test}$. 

A commonly-employed solution for image classification problems are convolutional neural networks (CNNs)– a variant of neural networks that employ $\textit{convolutions}$ to meaningfully extract information from images into a numerical form.

***Convolutions***

Given a $d \times d$ image $D$, and a $k \times k$ kernel $K$ where $k \leq d$, convolving $D$ with $K$ produces the $(d - k + 1) \times (d - k + 1)$ matrix $D * K$, where $(D*K)_{ij}$ is defined as the element-wise product of the $k \times k$ submatrix $D[i:i+k, j:j+k]$ and $K$.

Since a result of a convolution operation on $D$ is a function of $D$, it is an informative descriptor of $D$. Because of this, convolutional neural networks use convolutions of images to learn things about them. Further, since the result of a convolution operation on $D$ is also a function of $K$, each convolution of $D$ with a different $K\text{s}$ captures something different about $D$. Thus, CNNs usually employ a multitude of convolutions so as to encapsulate many different properties of the images given to them.

***Pruning based on metrics***

With many CNNs employing multiple different convolutions, and with each kernel spanning two dimensions, the storage requirements can blow up. Thus, it is a natural desire to want to reduce the number of parameters needed while maintaining the model's accuracy.


We call the individual entries of each kernel in the model "weights". We consider the problem of choosing 0.01% of the model's weights to set to 0. An informed way of doing so would be to choose based on some metric that identifies "unimportant" weights.

## Project Description

In this project, we consider one such metric, rather simply, the magnitude of the weight.

We prune (set to 0) the weights in the bottom 0.01% of all weights in the model when ranked by magnitude, as these weights should be the 0.01% least impactful.

To evaluate the usefulness of this metric, we must perform pruning with it on a concrete example problem. We consider the problem of classifying images from the Fashion-MNIST dataset ([Xiao et al, 2017](https://arxiv.org/abs/1708.07747)). The Fashion-MNIST dataset is a set of 70,000 28 $\times$ 28 images, each depicting one article of clothing that falls into one of 10 categories.

## Results

Before pruning, the accuracy on test data was 72.86%. After pruning, the accuracy on the test data was 71.31%.

Thus, the tradeoff seen was a 1.55% decrease in accuracy for a 0.01% decrease in the number of saved weights.

Whether this tradeoff is a good one is out of the scope of this paper and likely requires knowledge in computer systems and hardware. Thus, it is a difficult question to evaluate with my current background.