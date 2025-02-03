# Deep-Neural-Net-from-scratch-in-C
Building a Deep Neural Network from scratch in C, only using high performance math libraries, such as CBLAS, math.h and time.h.

The project has multiple source files since I have been following the [Sendtex "Neural Networks from Scratch" Series](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3).

## The repository contains the following:  
- LayerDense: a library that tries to replicate the most basic aspects of Tensorflow, with LayerDense as an object to work with.
   * Each LayerDense object contains its own set of weights, biases and outputs, giving total transparency and control to the user about the attributes of the layer.  
   * A constructor is provided, called newLayerDense, which returns a pointer to a new LayerDense object.  
   * A forward function that calculates the outputs of the neurons as weights * inputs + biases using CBLAS library sgemm function for the dot product.
   * Activation functions: Reinforcement Linear Unit and Softmax have been implemented.
   * A random input data generator: create_data produces new data as the user requests in the form of matrices whose values are stored linearly in float pointers.
   * A loss function to serve as an accuracy metric for the deep neural network to adjust its weights and biases.
   * An arbitrary function created only for the deep neural network to aim to approximate: (a + b) + ( c / (d + 1e-8)) > 0.5, only works for n_inputs = 4.
   * Additional functions to make working with the LayerDense objects more efficient, such as copy, add and deleteLayer.
- mainLayerDense: works exclusively with the LayerDense objects and its functions, feeding input into a deep neural network and shows the output in the terminal.
- mainOptimization: trains a deep neural network with 3 layers and applies optimization in order to approximate an arbitrary function.
- Source files with the basics of deep neural network, built at the beginning in order to gain a low-level knowledgeable understanding behind the layers from high-level deep learning Python libraries Tensorflow and Pytorch.

## Setup
Tools used:  
- gcc version: 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)  
- CBLAS: C interface for Basic Linear Algebra Subprograms  

To compile programs in this repository, use one of the following orders, making the appropiate changes:  
`gcc src/program.c -o program -lopenblas -lm`  
`gcc src/program.c include/LayerDense.h -o program -lopenblas -lm -I./include`


## Source files
The source files belong to one of the 9 parts that compose the Sendtex Series:
- Part 1: firstNeuralNet.c  
- Part 2: firstLayer.c  
- Part 3: firstEfficientLayer.c and firstDotProduct.c  
- Part 4: LayerWithBatches.c  
- Part 5 to 8: LayerDense.c, LayerDense.h and mainLayerDense.c  
- Part 9: LayerDense.c, LayerDense.h and mainOptimization.c  


## Main programs
mainLayerDense runs a basic deep neural network with no goal whatsoever. mainOptimization applies optimization, as the name shows, in order to approximate a function I created only for the deep neural network to approximate.

mainOptimization uses the same seed always, which gives a value close to 13 as the loss in the first epoch always. If it only shows one epoch, then run it more times and it will appear how loss goes down to 0 after applying optimization in the main program.


