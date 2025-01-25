#include <stdio.h>
#include <stdlib.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

int main() {
	/* First layer of 3 neurons using dot products */

	// inputs
	float inputs[] = {1.0, 2.0, 3.0, 2.5};
	
	// matrix 3 x 4 of weights
	float weights[] = {0.2, 0.8, -0.5, 1.0,       // first neuron
			   0.5, -0.91, 0.26, -0.5,    // second neuron
	      		   -0.26, -0.27, 0.17, 0.87}; // third neuron
	
	// biases
	float biases[] = {2.0, 3.0, 0.5};
	    
	// output for each neuron
	float output[] = {0,0,0};
	
	// low-level efficient loop for outputs calculation
	for ( int i = 0; i < 3; i++ ) {
		// matrix-array multiplication using CBLAS
		cblas_sgemv(
			CblasRowMajor,	// row-major layout
			CblasNoTrans,	// No transpose
			3,		// rows in weights
			4,		// columns in weights
			1.0,		// scaling factor for weights
			weights,	// pointer to weights
			4,		// columns for row-major
			inputs,		// pointer to inputs
			1,		// step for inputs
			0.0,		// scaling factor for outputs
			output,		// pointer to outputs
			1		// step for outputs
		);
		
		// biases added
		output[i] += biases[i];
		
		// outputs shown
		printf("Neuron number %d: %.3f\n",i+1,output[i]);
	}
}
