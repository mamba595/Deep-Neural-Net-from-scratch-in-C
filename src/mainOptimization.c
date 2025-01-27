#include <stdio.h>
#include <stdlib.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

// LayerDense struct object
#include "LayerDense.c"

int main() {
	/* 
	  	deep neural network layers: Classification task
			input layer -> 3 neurons
			activation_Softmax
			1 hidden layer -> 6 neurons
			activation_ReLU
			output layer -> 2 neurons
	*/
	
	LayerDense * input_layer = newLayerDense(4, 3),
		   * hidden_layer = newLayerDense(3,6),
		   * output_layer = newLayerDense(6,2);
		   
	float * best_weigths_input = input_layer->weights,
	      * best_weight_hidden = hidden_layer->weights,
	      * best_weight_output = output_layer->weights,
	      * best_biases_input  = input_layer->biases,
	      * best_biases_hidden = hidden_layer->biases,
	      * best_biases_output = output_layer->biases;
		
	float lowest_loss = 1e7;
	
	// optimization
		   
	for ( int i = 0; i < 1000; i++ ) {
		// adjust weights and biases
		/*
		input_layer->weights  += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)),
		hidden_layer->weights += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)),
		output_layer->weights += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)),
		input_layer->biases   += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)),
		hidden_layer->biases  += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)),
		output_layer->biases  += 0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2));
		*/
		
		// creates new random data for 3 batches each
		float * inputs = create_data(4,3);	
		
		// feeds the input to the neural net
		forward(input_layer, inputs, 3);
		forward(hidden_layer, input_layer->output, 3);
		forward(output_layer, hidden_layer->output, 3);
		
		// calculates targets
		int * targets = function_to_aproximate(inputs, 4, 3);
		
		// calculates loss
		float loss = calculate_loss(output_layer, targets, 3);
		
		// optimizes the neural net
		if ( loss < lowest_loss ) {
			best_weigths_input = input_layer->weights,
			best_weight_hidden = hidden_layer->weights,
			best_weight_output = output_layer->weights,
			best_biases_input  = input_layer->biases,
			best_biases_hidden = hidden_layer->biases,
			best_biases_output = output_layer->biases;
			
			lowest_loss = loss;
			
			printf("Iteration: %d  Loss: %.3f\n",i,loss);
		} else {
			// returns to the best weights and biases
			input_layer->weights  = best_weigths_input,
			hidden_layer->weights = best_weight_hidden,
			output_layer->weights = best_weight_output ,
			input_layer->biases   = best_biases_input,
			hidden_layer->biases  = best_biases_hidden,
			output_layer->biases  = best_biases_output;
		}
	}
	
	
}
