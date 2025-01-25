#include <stdio.h>
#include <stdlib.h>

// C interface for Basic Linear Algebra Subprograms
#include <cblas.h>

int main() {
	/* Layer with 3 different batches of inputs */

	// inputs
	float inputs[] = {1.0, 2.0, 3.0, 2.5,	// first batch
			  2.0, 5.0, -1.0,2.0,	// second batch
			  -1.5, 2.7, 3.3, -0.8};// third batch
	
	// matrix 3 x 4 of weights
	float weights[] = {0.2, 0.8, -0.5, 1.0,       // first neuron
			   0.5, -0.91, 0.26, -0.5,    // second neuron
	      		   -0.26, -0.27, 0.17, 0.87}; // third neuron
	
	// biases
	float biases[] = {2.0, 3.0, 0.5};
	    
	// output for each neuron
	float output[9] = {0.0};
	
	// low-level efficient loop for outputs calculation
	for ( int i = 0; i < 3; i++ ) {
		// matrix-array multiplication using CBLAS
		cblas_sgemm(
			CblasRowMajor,	// row-major layout
			CblasNoTrans,	// no transpose for inputs
			CblasTrans,	// transpose weights
			3,		// rows in inputs
			3,		// columns of transposed weights
			4,		// columns in inputs
			1.0,		// scaling factor for inputs
			inputs,		// pointer to inputs
			4,		// leading dimension of inputs
			weights,	// pointer to weights
			4,		// leading dimension of weights
			0.0,		// scaling factor for outputs
			output,		// pointer to outputs
			3		// leading dimension of outputs
		);
		
		// biases added
		cblas_saxpy(3,1.0,biases,1,output+(i*3),1);
		
		// outputs shown
		printf("Batch number %d:\n",i+1);
		
		for ( int j = 0; j < 3; j++ )
			printf("Neuron number %d: %.3f\n",j+1,output[j + (i*3)]);
	}
}
