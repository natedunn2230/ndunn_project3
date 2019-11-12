/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define  N 16// Length of vector that will be summed
#define BLOCK_SIZE 4// size of thread blocks

/**
 * Performs CPU Sum Reduction
 * x: vector to be summed
 * length: width of vector x
*/
int hostSumReduction(int* x, int length){

	for (int i = 1; i < length; i++)
		x[0] =x [0] + x[i];

	int overallSum = x[0];

	return overallSum;
}

/**
 * Performs GPU Sum Reduction
 * a: vector to be summed
 * length: Length of the vector to be added
*/
__global__ void deviceSumReduction(int *input, int *sum, int length){
	__shared__ int partialSum[2*BLOCK_SIZE];
	unsigned int tx = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;

	partialSum[tx] = input[start + tx];
	partialSum[blockDim.x+tx] = input[start+ blockDim.x+tx];

	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2){
		__syncthreads();
		if (tx < stride)
			partialSum[tx] += partialSum[tx+stride];
	}
	
	sum[blockIdx.x] = partialSum[0];
	printf("(%d, %d)\t%d\n", blockIdx.x, threadIdx.x, sum[blockIdx.x]); 
}



int main(void){
	int *a, *b, *dev_a, *dev_b;
	
	// used to keep track of sum array size (if n is larger than 2 * blocksize), then 
	// multiple kernel calls will have to be made
	int sumSize = N / (2 * BLOCK_SIZE); 
	
	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(N / dimBlock.x), 1, 1);
	
	// allocate vectors for cpu
	a = (int*)malloc(sizeof(int)* N);
	b = (int*)malloc(sizeof(int)* sumSize);
	
	// allocate vectors for gpu
	cudaMalloc((void **)(&dev_a), N* sizeof(int));
	cudaMalloc((void **)(&dev_b), sumSize * sizeof(int));
	
	// initialize vector
	int init =1325;
	for(int i=0;i<N;i++){
		// init=3125*init%65521;
		// a[i]=(init-32768)/16384;
		a[i] = i;
	}
	
	// copy array a (host) to dev_a (device)
	cudaMemcpy(dev_a,a,N * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Launch kernels for initial reduction
	deviceSumReduction<<<dimGrid, dimBlock>>>(dev_a, dev_b, N);
	cudaDeviceSynchronize();
	
	// copy results from gpu back to host
	cudaMemcpy(b, dev_b, sumSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// run sum reducton on host
	int sum = hostSumReduction(a, N);
	
	printf("Hello world!\n");
	printf("%d\n", sum);
	
	//debugging b
	for(int i =0; i < sumSize; i++){
		printf("%d ", b[i]);
	}
	printf("\n");
	
	
	// free system and device memory
	free(a);
	free(b);
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	
	return 0;
}