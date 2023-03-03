# Neural Network Implementation

Implementing a neural network from scratch is a valuable exercise for anyone seeking to understand the architecture of these powerful learning systems. By building a neural network from the ground up, I developed  a deeper understanding of the fundamental concepts that underlie this technology. This includes gaining insight into the role of different layers and activation functions, as well as the mathematics and algorithms used to train and optimize the network. Additionally, by coding a neural network from scratch, I developed a more intuitive understanding of the way that data flows through the network and how errors are backpropagated to adjust the weights and biases of the neurons. This knowledge can be invaluable for those seeking to work with neural networks in a professional capacity or simply for those who are interested in the inner workings of these powerful learning systems

## Description

This code implements components of a neural network including the input layer, hidden layers, activation function, weights and biases, an output layer, loss function, and optimization algorithm. 
* The input layer takes in the input data, while the hidden layers process the input and extract relevant features. 
*An activation function is applied to the output of each neuron in the hidden layers to introduce non-linearity. 
* Weights and biases are parameters associated with each neuron that are adjusted during training to optimize the network's performance. 
* The output layer produces the network's prediction or classification result, while the loss function measures the difference between the predicted and actual output. 
* An optimization algorithm is used to adjust the weights and biases during training to minimize the loss function and improve the network's performance.

## Getting Started

### Dependencies

* Install Julia (I prefered to use julia as it has proven to be way easier to compute matrices faster)
* Pkg, MLDatasets: MNIST, ImageCore
using Flux: onehotbatch, onecold, CUDA

### Contents
* activations_and_losses.jl:This file contains activation functions(tanh and RELu) that introduce non-linearity to the output of neurons in the hidden layers of a neural network. Also it contains different loss functions that measure the difference between predicted and actual output. Both activation and losses files are crucial in the development and training of a neural network.
* ModelTraining: I uploaded a famous dataset and trained on my neural network model(get creative :) and have fun)
* dense_network_model.jl:This code defines abstract types and mutable structs for a dense neural network with CPU and GPU implementations. The DenseLayer structs contain arrays for weights, biases, stored activations, stored deltas, activation functions, and their derivatives. The DenseNetwork structs contain input, hidden, and output layers of DenseLayer structs, with the GPU implementation using CuArrays. The function DenseLayerCPU constructs a DenseLayer object with given size, activation function, and weight distribution.
* dense_network_training.jl: This script is for training a dense neural network using mini-batch stochastic gradient descent. It includes functions for predicting the output of the network given an input, computing the gradients based on the difference between the predicted output and the actual output, updating the weights in the network, and training the network over multiple epochs. The training function records and returns the loss on the entire data set at the end of each epoch, and can print the epoch numbers and losses if verbose is set to true
* experimental_results: scratchwork, I used to test various steps



## Help

This project was fun and took me a long time to wrap my head around it. so I'd be happy to help you understand this code and walk you through it! Just let me know what specific questions you have or what parts of the code you would like me to explain.

## Authors

Contributors names and contact info
 
[@MikeRemezo(https://www.linkedin.com/in/mike-remezo/)



## License

This project is licensed under the [Mike Remezo] License - see the LICENSE.md file for details

## Acknowledgments

* Bryce Wiedenbeck
