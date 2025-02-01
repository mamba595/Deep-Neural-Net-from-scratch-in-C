# Deep-Neural-Net-from-scratch-in-C
Building a Deep Neural Network from scratch in C, only using high performance math libraries, such as CBLAS, math.h and time.h.

The repository contains the following:  
	1. LayerDense: a library that tries to replicate the most basic aspects of Tensorflow, with LayerDense as an object to work with.  
		* Each LayerDense object contains its own set of weights, biases and outputs, giving total transparency and control to the user about the attributes of the layer.  
   		* A constructor is provided, called newLayerDense, which returns a pointer to a new LayerDense object.  
	 	* A forward function that calculates the outputs of the neurons as weights * inputs + biases using CBLAS library sgemm function for the dot product.   

The project has multiple source files since I have been following the Sendtex "Neural Networks from Scratch" Series:
https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

Dependencies:
	- gcc version: 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)
	- CBLAS: C interface for Basic Linear Algebra Subprograms

To compile most of the programs in this repository, just use the following order, changing "program" with the name of the program you are trying to compile:
gcc src/program.c -o obj/program -lopenblas -lm

Part 1: firstNeuralNet.c
Part 2: firstLayer.c
Part 3: firstEfficientLayer.c and firstDotProduct.c
Part 4: LayerWithBatches.c
Part 5 to 8: LayerDense.c, LayerDense.h and mainLayerDense.c
Part 9: LayerDense.c, LayerDense.h and mainOptimization.c

mainLayerDense runs a basic deep neural network with no goal whatsoever. mainOptimization applies optimization, as the name shows, in order to approximate a function I created only for the deep neural network to approximate.

mainOptimization uses the same seed always, which gives a value close to 13 as the loss in the first epoch always. If it only shows one epoch, then run it more times and it will appear how loss goes down to 0 after applying optimization in the main program.


