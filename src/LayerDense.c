#include "LayerDense.h"

LayerDense * newLayerDense(int n_inputs, int n_neurons) {
	// new layer created
	LayerDense * layer = malloc(sizeof(LayerDense));
	
	// initialization of attributes
	layer->n_inputs = n_inputs;
	layer->n_neurons = n_neurons;
	
	// seed for the random number generator
	srand(time(NULL));
	
	// malloc for weights
	layer->weights = (float *)malloc(n_inputs * n_neurons * sizeof(float));
	
	// weights initialized randomly to values between 0.0 and 1.0
	for ( int i = 0; i < n_inputs * n_neurons; i++ ) {
		layer->weights[i] = (float)rand() / RAND_MAX;
	}
	
	// malloc for biases
	layer->biases = (float *)malloc(n_neurons * sizeof(float));
	
	//  biases initiliazed to random values between 0.0 and 1.0
	for ( int i = 0; i < n_neurons; i++)
		layer->biases[i] = (float)rand() / RAND_MAX;
		
	return layer;
}

void forward( LayerDense * layer, float * inputs, int n_batches ) {
	// malloc for output
	layer->output = (float *)malloc(layer->n_neurons * n_batches * sizeof(float));
	
	// matrix-array multiplication using CBLAS
	cblas_sgemm(
		CblasRowMajor,	 // row-major layout
		CblasNoTrans,	 // no transpose for inputs
		CblasTrans,	 // transpose weights
		n_batches,	 // rows in inputs
		layer->n_neurons,// columns of transposed weights
		layer->n_inputs, // columns in inputs
		1.0,		 // scaling factor for inputs
		inputs,		 // pointer to inputs
		layer->n_inputs, // leading dimension of inputs
		layer->weights,	 // pointer to weights
		layer->n_inputs, // leading dimension of weights
		0.0,		 // scaling factor for outputs
		layer->output,	 // pointer to outputs
		3		 // leading dimension of outputs
	);
		
	// low-level efficient loop for outputs calculation
	for ( int i = 0; i < n_batches; i++ ) {
		// biases added
		cblas_saxpy(layer->n_neurons,1.0,layer->biases,1,layer->output+(i*layer->n_neurons),1);
	}
}
