#include <stdio.h>
#include <stdlib.h>

int main() {
	/* A layer of 3 neurons */

	// inputs
	float inputs[] = {1.0, 2.0, 3.0, 2.5};
	
	// weights
	float weights1[] = {0.2, 0.8, -0.5, 1.0},
	      weights2[] = {0.5, -0.91, 0.26, -0.5},
	      weights3[] = {-0.26, -0.27, 0.17, 0.87};
	
	// biases
	float bias1 = 2.0,
	      bias2 = 3.0,
	      bias3 = 0.5;
	    
	// output for each neuron
	float output[] = {bias1,bias2,bias3};
	
	// first neuron
	for ( int i = 0; i < 4; i++ )
		output[0] += inputs[i] * weights1[i];
		
	// second neuron
	for ( int i = 0; i < 4; i++ )
		output[1] += inputs[i] * weights2[i];
	
	// third neuron
	for ( int i = 0; i < 4; i++ )
		output[2] += inputs[i] * weights3[i];
		
	// outputs
	printf("First neuron: %.3f\n",output[0]); 
	printf("Second neuron: %.3f\n",output[1]);
	printf("Third neuron: %.3f\n",output[2]);
}
