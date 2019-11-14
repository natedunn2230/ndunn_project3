/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define N 8
#define BLOCK_SIZE 2


/**
 * Performs Prefix Sum on a Vector using the CPU
 * 
*/
void hostPrefixSum(int *y, int *x, int length){
    y[0] = x[0];
	for (int i = 1; i < length; i++) 
		y[i] = y [i-1] + x[i];
}

/**
 * Compares two vectors a and b for equality
*/
int verify(int *a, int *b, int length){
	for(int i = 0; i < length; i++){
		if(a[i] != b[i])
			return 0;
	}
	
	return 1;
}

int main(void){
	int *a, *b;
	
	a = (int*)malloc(sizeof(int) * N);
	b = (int*)malloc(sizeof(int) * N);
	
	// initialize a
	int init = 1325;
	for (int i=0; i<N; i++){
		init=3125*init%65521;
		a[i]=(init-32768)/16384;
	}
	
	// perform prefix sum on cpu
	hostPrefixSum(b, a, N);

	printf("\n");
	for(int i = 0; i < N; i++){
		printf("%d ", b[i]);
	}

	
	return 0;	
}