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

struct LayerDense {
	float * weights;
	float * biases;
	float * output;
	int n_inputs;
	int n_neurons;
};

LayerDense * newLayerDense(int n_inputs, int n_neurons);

void forward( LayerDense * layer, float * inputs, int n_batches );

void activation_ReLU( LayerDense * layer, int n_batches );

void activation_Softmax( LayerDense * layer, int n_batches );

void getOutput( LayerDense * layer, int n_batches );

float * create_data( int n_inputs, int n_batches );

int * create_classification_target( int n_neurons, int n_batches );

float calculate_loss( LayerDense * layer, int * targets, int n_batches );

void deleteLayer(LayerDense * layer);

#endif
