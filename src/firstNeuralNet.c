#include <stdio.h>
#include <stdlib.h>

int main() {
	float inputs[3] = {1.2, 5.1, 2.1},
	      weights[3] = {3.1, 2.1, 8.7},
	      bias = 3.0;
	    
	float output = bias;
	
	for ( int i = 0; i < 3; i++ )
		output += inputs[i] * weights[i];
		
	printf("First output: %.1f",output); 
}
