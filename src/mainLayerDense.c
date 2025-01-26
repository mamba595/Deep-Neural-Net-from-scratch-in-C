#include <stdio.h>
#include <stdlib.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

// LayerDense struct object
#include "LayerDense.c"

int main() {
	/* First layer with 3 different batches of inputs */
	LayerDense * layer = newLayerDense(4,3);

	// inputs
	float inputs[] = {1.0, 2.0, 3.0, 2.5,	// first batch
			  2.0, 5.0, -1.0,2.0,	// second batch
			  -1.5, 2.7, 3.3, -0.8};// third batch
	    
	forward(layer,inputs,3);
	
	// output for each neuron
	float * output;
	
	output = (layer->output);
	
	// outputs of the first layer shown
	for ( int i = 0; i < 3; i++ ) {
		printf("Batch number %d:\n",i+1);
		
		for ( int j = 0; j < layer->n_neurons; j++ )
			printf("Neuron number %d: %.3f\n",j+1,layer->output[j + (i*layer->n_neurons)]);
	}
	
	/* Second layer */
	LayerDense * layer2 = newLayerDense(3,2);
	
	// takes output from last layer
	forward(layer2,layer->output,3);
	
	printf("\n\n");
	
	// outputs of the second layer
	for ( int i = 0; i < 3; i++ ) {
		printf("Batch number %d:\n",i+1);
		
		for ( int j = 0; j < layer2->n_neurons; j++ )
			printf("Neuron number %d: %.3f\n",j+1,layer2->output[j + (i*layer2->n_neurons)]);
	}
}
