#include "LayerDense.h"

struct LayerDense {
	float * weights;
	float * biases;
	float * output;
	int n_inputs;
	int n_neurons;
}

LayerDense * newLayerDense(int n_inputs, int n_neurons) {
	// new layer created
	LayerDense layer;
	
	layer.n_inputs = n_inputs;
	layer.n_neurons = n_neurons;
	
}
