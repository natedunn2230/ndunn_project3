/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define  N 1048576// Length of vector that will be summed
#define BLOCK_SIZE 256// size of thread blocks

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
 * input: vector to be summed
 * sum: sum array of input stored
 * length: Length of the vector to be added
*/
__global__ void deviceSumReduction(int *input, int length){
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
	
	input[blockIdx.x] = partialSum[0];
	printf("(%d, %d)\t%d\n", blockIdx.x, threadIdx.x, input[blockIdx.x]); 
}

/**
 * Calls cuda kernel function recursively to get total sum reduction
 * a: Array to be summed
 * length: length of array to be summed
*/
void applyReduction(int *a, int length){
	
	// make a copy of incoming array to be used for current sum
	int *vect = (int*)malloc(sizeof(int) * length);
	memcpy(vect, a, sizeof(int) * length);

	int *vect_dev;
	int sumSize = ceil((float)length / (2 * BLOCK_SIZE)); // size of sum array after each iteration
	
	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil((float)length / dimBlock.x), 1, 1);
	
	// allocate vectors for gpu
	cudaMalloc((void **)(&vect_dev), length * sizeof(int));
	
	// copy array a (host) to dev_a (device)
	cudaMemcpy(vect_dev, vect, length * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Launch kernels for reduction
	deviceSumReduction<<<dimGrid, dimBlock>>>(vect_dev, length);
	cudaDeviceSynchronize();
	
	// copy results from gpu back to host
	cudaMemcpy(vect, vect_dev, sumSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// free allocated device memory
	cudaFree(vect_dev);
	cudaDeviceSynchronize();
	
	printf("\nFOR sum size of %d\n", sumSize);
	for(int i = 0; i < sumSize; i++){
		
		printf("%d ", vect[i]);
	}
	printf("\n");
	
	// apply reduction again on sum array, if applicable
	// if(sumSize > 1)
		// //return 0 + applyReduction(vect, sumSize);
	// else{
		// int sum = vect[0];
		// free(vect);
		// //return sum;
	// }
}

int main(void){
	printf("VECTOR OF SIZE: %d\nBLOCK SIZE: %d\n\n", N, BLOCK_SIZE);

	// allocate vector for cpu
	int *a = (int*)malloc(sizeof(int)* N);
	
	// initialize vector
	int init =1325;
	for(int i=0;i<N;i++){
		init=3125*init%65521;
		a[i]=(init-32768)/16384;
	}
	
	// run reduction on device
	//int gpuSum = applyReduction(a, N);
	applyReduction(a, N);
	// variables used to measure cpu computation time
	clock_t cpuStart, cpuEnd;
	float cpuTimeTaken;
	
	// start measuring cpu computation time
	cpuStart = clock();
	
	// run sum reducton on host
	int cpuSum = hostSumReduction(a, N);
	
	// stop measuring cpu computation time
	cpuEnd = clock();
	cpuTimeTaken = ((float)cpuEnd - cpuStart)/CLOCKS_PER_SEC; // in seconds 
	
	
	//printf("GPU SUM: %d\n", gpuSum);
	printf("CPU SUM: %d\n", cpuSum);
	//printf("GPU SUM: %d\n", gpuSum);
	
	printf("\nCPU Time: %f\n", cpuTimeTaken);
	
	// if(cpuSum == gpuSum)
		// printf("TEST PASSED!\n");
	// else 
		// printf("TEST FAILED!\n");
	
	// free system memory
	free(a);
		
	return 0;
}