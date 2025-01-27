#include <stdio.h>
#include <stdlib.h>

// for random number generation
#include <time.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

// basic math library
#include <math.h>

#ifndef LAYER_DENSE
#define LAYER_DENSE

typedef struct LayerDense LayerDense;

// struct representing each layer of the neural network
// similar to layer.Dense() from Tensorflow.keras

struct LayerDense {
	float * weights;
	float * biases;
	float * output;
	int n_inputs;
	int n_neurons;
};

// creates a new layer object, allocating memory and initializing it

LayerDense * newLayerDense(int n_inputs, int n_neurons);


// calculates the outputs for each neuron such as weights * inputs + biases

void forward( LayerDense * layer, float * inputs, int n_batches );


// rectified Linear Unit activation function
// if ( x < 0 ) x = 0, eliminates negative values

void activation_ReLU( LayerDense * layer, int n_batches );


// softmax activation function

void activation_Softmax( LayerDense * layer, int n_batches );


// C equivalent of overloading C++ << operator

void getOutput( LayerDense * layer, int n_batches );


// creates new random data for inputs to feed the neural network

float * create_data( int n_inputs, int n_batches );


// selects classification values randomly

int * create_classification_target( int n_neurons, int n_batches );


// function to approximate in the mainOptimization program
// (inputs[0] + inputs[1]) + ( inputs[2] / (inputs[3] + 1e-8)) > 0.5

int * function_to_aproximate( float * inputs, int n_inputs, int n_batches );


// loss function

float calculate_loss( LayerDense * layer, int * targets, int n_batches );


// frees memory heap, allowing the user to run different neural nets in the same program run

void deleteLayer(LayerDense * layer);

#endif
