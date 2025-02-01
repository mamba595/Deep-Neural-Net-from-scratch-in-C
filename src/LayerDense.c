#include "LayerDense.h"

LayerDense * newLayerDense(int n_inputs, int n_neurons) {
	// new layer created
	LayerDense * layer = malloc(sizeof(LayerDense));
	
	// initialization of attributes
	layer->n_inputs = n_inputs;
	layer->n_neurons = n_neurons;
	
	// initizalition of std dev for ReLU compatibility
	float std = sqrtf(2.0f / n_inputs);
	
	// malloc for weights
	layer->weights = (float *)malloc(n_inputs * n_neurons * sizeof(float));
	
	// weights initialized randomly to values between -1.0 and 1.0
	for ( int i = 0; i < n_inputs * n_neurons; i++ ) {
		layer->weights[i] = std * ((-1.0) + (((float)rand() / RAND_MAX) * 2));
	}
	
	// malloc for biases
	layer->biases = (float *)malloc(n_neurons * sizeof(float));
	
	//  biases initiliazed to random values between -1.0 and 1.0
	for ( int i = 0; i < n_neurons; i++)
		layer->biases[i] = std * ((-1.0) + (((float)rand() / RAND_MAX) * 2));
		
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
		layer->n_neurons, // columns of transposed weights
		layer->n_inputs, // columns in inputs
		1.0,		 // scaling factor for inputs
		inputs,		 // pointer to inputs
		layer->n_inputs, // leading dimension of inputs
		layer->weights,	 // pointer to weights
		layer->n_inputs,// leading dimension of weights
		0.0,		 // scaling factor for outputs
		layer->output,	 // pointer to outputs
		layer->n_neurons // leading dimension of outputs
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
	for ( int batch = 0; batch < n_batches; batch++ ) {
	
		float e = 2.71828182845904,
		      max = layer->output[0],
		      sum = 0;
		
		int offset = batch * layer->n_neurons;
		
		for ( int i = 0; i < layer->n_neurons; i++ ) {
			layer->output[offset+i] = pow(e, layer->output[offset+i]);
			if ( layer->output[offset+i] > max )
				max = layer->output[i];
		}
		
		for ( int i = 0; i < layer->n_neurons; i++ ) {
			layer->output[offset+i] = layer->output[offset+i] - max;
			sum += layer->output[offset+i];
		}
		
		for ( int i = 0; i < layer->n_neurons; i++ )
			layer->output[offset+i] = layer->output[offset+i] / sum;	
	}
}

void getOutput( LayerDense * layer, int n_batches ) {
	printf("\n\n");

	for ( int i = 0; i < n_batches; i++ ) {
		printf("\nBatch number %d:\n",i+1);
		
		for ( int j = 0; j < layer->n_neurons; j++ )
			printf("Neuron number %d: %.3f\n",j+1,layer->output[j + (i*layer->n_neurons)]);
	}
}

void copy(float * a, float * b, int rows, int cols) {
	for ( int i = 0; i < rows * cols; i++ )
		b[i] = a[i];
}

void add(float * M, float b, int rows, int cols) {
	for ( int i = 0; i < rows * cols; i++ )
		M[i] += b;
}

float * create_data( int n_inputs, int n_batches ) {
	// malloc for inputs
	float * inputs = malloc( n_inputs * n_batches * sizeof(float));
	
	// inputs initialized to random values between 0.1 and 0.9
	for ( int i = 0; i < n_inputs * n_batches; i++ ) {
		inputs[i] = 0.1 + (((float)rand() / RAND_MAX) * 0.8);
	}

	return inputs;
}

int * create_classification_target( int n_neurons, int n_batches ) {
	int * targets = (int *)malloc(n_batches * sizeof(int));
	
	for ( int i = 0; i < n_batches; i++ )
		targets[i] = rand() % n_neurons;
		
	return targets;
}

int * function_to_aproximate( float * inputs, int n_inputs, int n_batches ) {
	int * targets = (int *)malloc(n_batches * sizeof(int));
	
	for ( int i = 0; i < n_batches; i++ ) {
		float val = (inputs[i*n_inputs+0] + inputs[i*n_inputs+1]) + 
				( inputs[i*n_inputs+2] / (inputs[i*n_inputs+3] + 1e-7));
		
		if ( val > 0.5 )
			targets[i] = 0;
		else
			targets[i] = 1;
	}
	
	return targets;
}

float calculate_loss( LayerDense * layer, int * targets, int n_batches ) {
	float loss = 0;
	const float epsilon = 1e-6;
	
	for ( int i = 0; i < n_batches; i++ ) {
		float val = layer->output[i*layer->n_neurons + targets[i]];
		
		// makes sure log(0) is not computed
		val = fmaxf(val,epsilon);
		
		loss += -(log(val));
	}
		
	return loss / n_batches;
}

void deleteLayer(LayerDense * layer) {
	free(layer->weights);
	free(layer->biases);
	free(layer->output);
	free(layer);
}
