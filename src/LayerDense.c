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
	
	// weights initialized randomly to values between -1.0 and 1.0
	for ( int i = 0; i < n_inputs * n_neurons; i++ ) {
		layer->weights[i] = (-1.0) + (((float)rand() / RAND_MAX) * 2);
	}
	
	// malloc for biases
	layer->biases = (float *)malloc(n_neurons * sizeof(float));
	
	//  biases initiliazed to random values between -1.0 and 1.0
	for ( int i = 0; i < n_neurons; i++)
		layer->biases[i] = (-1.0) + (((float)rand() / RAND_MAX) * 2);
		
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
		cblas_saxpy(layer->n_neurons,
			    1.0,
			    layer->biases,
			    1,
			    layer->output+(i*layer->n_neurons),
			    1
		);
	}
}

void activation_ReLU( LayerDense * layer, int n_batches ) {
	for ( int i = 0; i < n_batches * layer->n_neurons; i++ )
		if ( layer->output[i] < 0 ) layer->output[i] = 0;
}

void activation_Softmax( LayerDense * layer, int n_batches ) {
	float e = 2.71828182845904,
	      max = layer->output[0],
	      sum = 0;
	
	for ( int i = 0; i < n_batches * layer->n_neurons; i++ ) {
		layer->output[i] = pow(e, layer->output[i]);
		if ( layer->output[i] > max )
			max = layer->output[i];
	}
	
	for ( int i = 0; i < n_batches * layer->n_neurons; i++ ) {
		layer->output[i] = layer->output[i] - max;
		sum += layer->output[i];
	}
	
	for ( int i = 0; i < n_batches * layer->n_neurons; i++ )
		layer->output[i] = layer->output[i] / sum;
}

void getOutput( LayerDense * layer, int n_batches ) {
	printf("\n\n");

	for ( int i = 0; i < n_batches; i++ ) {
		printf("\nBatch number %d:\n",i+1);
		
		for ( int j = 0; j < layer->n_neurons; j++ )
			printf("Neuron number %d: %.3f\n",j+1,layer->output[j + (i*layer->n_neurons)]);
	}
}

float * create_data( int n_inputs, int n_batches ) {
	// malloc for inputs
	float * inputs = malloc( n_inputs * n_batches * sizeof(float));
	
	// seed for the random number generator
	srand(time(NULL));
	
	// inputs initialized to random values between -1.0 and 1.0
	for ( int i = 0; i < n_inputs * n_batches; i++ ) {
		inputs[i] = (-1.0) + (((float)rand() / RAND_MAX) * 2);
	}

	return inputs;
}

void deleteLayer(LayerDense * layer) {
	free(layer->weights);
	free(layer->biases);
	free(layer->output);
	free(layer);
}
