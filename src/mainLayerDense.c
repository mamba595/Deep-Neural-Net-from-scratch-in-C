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
	/*float inputs[] = {1.0, 2.0, 3.0, 2.5,	// first batch
			  2.0, 5.0, -1.0,2.0,	// second batch
			  -1.5, 2.7, 3.3, -0.8};// third batch
			  
	*/
	
	// aritificial data created
	float * inputs = create_data(4,3);
	    
	forward(layer,inputs,3);
	
	// output for each neuron
	float * output;
	
	output = (layer->output);
	
	// outputs of the first layer before activation reLU shown
	getOutput(layer,3);
	
	// activation reLU function
	activation_ReLU(layer,3);
	
	// output after activation function
	getOutput(layer,3);
	
	/* Second layer */
	LayerDense * layer2 = newLayerDense(3,2);
	
	// takes output from last layer
	forward(layer2,layer->output,3);
	
	// this layer is not used anymore
	deleteLayer(layer);
	
	printf("\n\n");
	
	// output before activationSoftmax
	getOutput(layer2,3);
	
	// activation softmax function
	activation_Softmax(layer2,3);
	
	// outputs of the second layer
	getOutput(layer2,3);
	
	// loss function
	printf("\n\nLoss function: \n");
	int * target = create_classification_target(layer2->n_neurons,3);
	printf("Loss: %.3f\n\n", calculate_loss(layer2,target,3));
	
	// free dynamic memory
	deleteLayer(layer2);
	free(target);
}
