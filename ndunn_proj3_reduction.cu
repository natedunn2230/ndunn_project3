/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define  N 262144// Length of vector that will be summed
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
	//printf("(%d, %d)\t%d\n", blockIdx.x, threadIdx.x, sum[blockIdx.x]); 
}

/**
 * Prints the specified vector
 * vect: vector to be printed
 * length: length of vector
*/
void printVector(int *vect, int length){
	for(int i = 0; i < length; i++)
		printf("%d\t", vect[i]);
	printf("\n");
}

int main(void){
	printf("\nSTARTING REDUCTION SUM ON VECTOR OF SIZE: %d\n\n", N);
	
	int *a, *sumArray, *dev_a, *dev_sumArray;
	
	// used to keep track of sum array size (if n is larger than 2 * blocksize), then 
	// multiple kernel calls will have to be made
	int sumSize = ceil((float)N / (2 * BLOCK_SIZE)); 

	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(N / dimBlock.x), 1, 1);
	
	// allocate vectors for cpu
	a = (int*)malloc(sizeof(int)* N);
	sumArray = (int*)malloc(sizeof(int)* sumSize); // holds initial sum array
	
	// allocate vectors for gpu
	cudaMalloc((void **)(&dev_a), N* sizeof(int));
	cudaMalloc((void **)(&dev_sumArray), sumSize * sizeof(int));
	
	// initialize vector
	int init =1325;
	for(int i=0;i<N;i++){
		init=3125*init%65521;
		a[i]=(init-32768)/16384;
	}
	//printVector(a, N);
	
	// copy array a (host) to dev_a (device)
	cudaMemcpy(dev_a,a,N * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Launch kernels for initial reduction
	deviceSumReduction<<<dimGrid, dimBlock>>>(dev_a, dev_sumArray, N);
	cudaDeviceSynchronize();
	
	// copy results from gpu back to host
	cudaMemcpy(sumArray, dev_sumArray, sumSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// free memory used for initial sum array generation
	cudaFree(dev_a);
	cudaFree(dev_sumArray);
	cudaDeviceSynchronize();
	
	// keep performing reduction on sum (if applicable)
	int gpuSum = sumArray[0];
	while(sumSize > 1){
		dim3 dimGrid(ceil(sumSize / dimBlock.x), 1, 1);
		// printf("sum: %d\n", gpuSum);
		// printf("sum length: %d\n", sumSize);
		int *dev_sumCpy; 
		
		// allocate vectors for gpu (holds 
		cudaMalloc((void **)(&dev_sumArray), sumSize * sizeof(int));
		cudaDeviceSynchronize();
		
		int newSumSize = ceil((float)sumSize / (2 * BLOCK_SIZE));
		cudaMalloc((void **)(&dev_sumCpy), newSumSize * sizeof(int));
		cudaDeviceSynchronize();
		
		// copy array b (host) to dev_b (device)
		cudaMemcpy(dev_sumArray,sumArray, sumSize * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
 		
		// Launch kernels for initial reduction
		deviceSumReduction<<<dimGrid, dimBlock>>>(dev_sumArray, dev_sumCpy, sumSize);
		cudaDeviceSynchronize();
			
		// copy results from gpu back to host
		cudaMemcpy(sumArray, dev_sumCpy, newSumSize * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		
		cudaFree(dev_sumArray);
		cudaFree(dev_sumCpy);
		cudaDeviceSynchronize();
		
		sumSize = newSumSize;
		gpuSum = sumArray[0];
	}
	
	// run sum reducton on host
	int cpuSum = hostSumReduction(a, N);
	
	printf("GPU SUM: %d\n", gpuSum);
	printf("CPU SUM: %d\n", cpuSum);
	
	if(cpuSum == gpuSum)
		printf("TEST PASSED!\n");
	else 
		printf("TEST FAILED!\n");
	
	// free system and remaining device memory
	free(a);
	free(sumArray);
	
	return 0;
}