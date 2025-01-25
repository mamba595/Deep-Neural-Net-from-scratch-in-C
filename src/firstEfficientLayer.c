#include <stdio.h>
#include <stdlib.h>

int main() {
	/* An efficient layer of 3 neurons */

	// inputs
	float inputs[] = {1.0, 2.0, 3.0, 2.5};
	
	// matrix of weights
	float weights[][4] = {{0.2, 0.8, -0.5, 1.0},
			     {0.5, -0.91, 0.26, -0.5},
	      		     {-0.26, -0.27, 0.17, 0.87}};
	
	// biases
	float biases[] = {2.0, 3.0, 0.5};
	    
	// output for each neuron
	float output[] = {0,0,0};
	
	// clean loop for outputs calculation
	for ( int i = 0; i < 3; i++ ) {
		for ( int j = 0; j < 4; j++ )
			output[i] += inputs[j] * weights[i][j];
			
		output[i] += biases[i];
		
		// outputs
		printf("Neuron number %d: %.3f\n",i+1,output[i]);
	}
}
