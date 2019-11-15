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
* Performs Prefix Sum on a vector using GPU
*/
__global__ void work_efficient_scan_kernel (int *x, int *y, int InputSize){

 	__shared__ int scan_array[2 * BLOCK_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x * blockDim.x;
	scan_array[t] = x[start + t];
	scan_array[blockDim.x + t] = x[start + blockDim.x + t];

	__syncthreads();

	// Perform reduction step
   int reduction_stride = 1;
   while(reduction_stride <= BLOCK_SIZE){
        int index = (threadIdx.x + 1) * reduction_stride * 2 - 1;
		
        if(index < 2 * BLOCK_SIZE)
            scan_array[index] += scan_array[index-reduction_stride];
		
        reduction_stride = reduction_stride*2;

        __syncthreads();
    }

	// Perform post scan step
    int post_stride = BLOCK_SIZE/2;
    while(post_stride > 0){
        int index = (threadIdx.x + 1) * post_stride * 2 - 1;
		
        if(index + post_stride < 2 * BLOCK_SIZE)
			scan_array[index + post_stride] += scan_array[index];
		
        post_stride = post_stride / 2;
        __syncthreads();
    }

	__syncthreads();

	x[start + t] = scan_array[t];
	x[start+ blockDim.x + t] = scan_array[blockDim.x + t];

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

/**
 * Print the given vector a
*/
void printVector(int *a, int length){

	for(int i = 0; i < length; i++){
		printf("%d ", a[i]);
	}
	printf("\n");
}

int main(void){
	int *a, *b;
	
	// initialize cpu vectors
	a = (int*)malloc(sizeof(int) * N); // original vector
	b = (int*)malloc(sizeof(int) * N); // stores cpu prefix sum
	
	// initialize a
	int init = 1325;
	for (int i=0; i<N; i++){
		init=3125*init%65521;
		a[i]=(init-32768)/16384;
	}
	
	// perform prefix sum on cpu
	hostPrefixSum(b, a, N);
	
	
	

	
	return 0;	
}