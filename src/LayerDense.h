#include <stdio.h>
#include <stdlib.h>

// for random number generation
#include <time.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

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

#endif
