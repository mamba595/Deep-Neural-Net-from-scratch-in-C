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
	
	LayerDense * input_layer = newLayerDense(4,3),
		   * hidden_layer = newLayerDense(3,6),
		   * output_layer = newLayerDense(6,2);
		   
	float * best_weights_input = malloc(input_layer->n_inputs * input_layer->n_neurons * sizeof(float)),
	      * best_weights_hidden = malloc(hidden_layer->n_inputs * hidden_layer->n_neurons * sizeof(float)),
	      * best_weights_output = malloc(output_layer->n_inputs * output_layer->n_neurons * sizeof(float)),
	      * best_biases_input  = malloc(input_layer->n_neurons * sizeof(float)),
	      * best_biases_hidden = malloc(hidden_layer->n_neurons * sizeof(float)),
	      * best_biases_output = malloc(output_layer->n_neurons * sizeof(float));
	      
	copy(input_layer->weights, best_weights_input, input_layer->n_inputs, input_layer->n_neurons);
	copy(hidden_layer->weights, best_weights_hidden, hidden_layer->n_inputs, hidden_layer->n_neurons);
	copy(output_layer->weights, best_weights_output, output_layer->n_inputs, output_layer->n_neurons);
	copy(input_layer->biases, best_biases_input, input_layer->n_neurons, 1);
	copy(hidden_layer->biases, best_biases_hidden, hidden_layer->n_neurons, 1);
	copy(output_layer->biases, best_biases_output, output_layer->n_neurons, 1);
		
	float lowest_loss = 1e7;
	
	// optimization
		   
	for ( int i = 0; i < 1000; i++ ) {
		// adjust weights and biases
		add( input_layer->weights,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			input_layer->n_inputs, input_layer->n_neurons);
		add( hidden_layer->weights,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			hidden_layer->n_inputs, hidden_layer->n_neurons);
		add( output_layer->weights,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			output_layer->n_inputs, output_layer->n_neurons);
		add( input_layer->biases,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			input_layer->n_neurons, 1);
		add( hidden_layer->biases,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			hidden_layer->n_neurons, 1);
		add( output_layer->biases,  0.05 * ((-1.0) + (((float)rand() / RAND_MAX) * 2)), 
			output_layer->n_neurons, 1);
		
		// creates new random data for 3 batches each
		float * inputs = create_data(4,3);
		
		// calculates targets
		int * targets = function_to_aproximate(inputs, 4, 3);
		
		// training the neural net
		forward(input_layer, inputs, 3);
		activation_ReLU( input_layer, 3 );
		forward(hidden_layer, input_layer->output, 3);
		activation_Softmax( hidden_layer, 3 );
		forward(output_layer, hidden_layer->output, 3);
		
		free(inputs);
		
		// calculates loss
		float loss = calculate_loss(output_layer, targets, 3);
		
		free(targets);
		
		// optimizes the neural net
		if ( loss < lowest_loss ) {
			copy(input_layer->weights, best_weights_input, 
				input_layer->n_inputs, input_layer->n_neurons);
			copy(hidden_layer->weights, best_weights_hidden, 
				hidden_layer->n_inputs, hidden_layer->n_neurons);
			copy(output_layer->weights, best_weights_output, 
				output_layer->n_inputs, output_layer->n_neurons);
			copy(input_layer->biases, best_biases_input, input_layer->n_neurons, 1);
			copy(hidden_layer->biases, best_biases_hidden, hidden_layer->n_neurons, 1);
			copy(output_layer->biases, best_biases_output, output_layer->n_neurons, 1);
			
			lowest_loss = loss;
			
			//printf("Epoch: %d/%d  Loss: %.3f\n",i,1000,loss);
		} else {
			// returns to the best weights and biases
			copy(best_weights_input, input_layer->weights, 
				input_layer->n_inputs, input_layer->n_neurons);
			copy(best_weights_hidden, hidden_layer->weights, 
				hidden_layer->n_inputs, hidden_layer->n_neurons);
			copy(best_weights_output, output_layer->weights, 
				output_layer->n_inputs, output_layer->n_neurons);
			copy(best_biases_input, input_layer->biases, input_layer->n_neurons, 1);
			copy(best_biases_hidden, hidden_layer->biases, hidden_layer->n_neurons, 1);
			copy(best_biases_output, output_layer->biases, output_layer->n_neurons, 1);
		}
		
		printf("Epoch: %d/%d  Loss: %.3f\n",i+1,1000,loss);
	}
	
	deleteLayer(input_layer);
	deleteLayer(hidden_layer);
	deleteLayer(output_layer);
}
