/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define  N 16777216  // Length of vector that will be summed
#define BLOCK_SIZE 128// size of thread blocks

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
	//printf("(%d, %d)\t%d\n", blockIdx.x, threadIdx.x, input[blockIdx.x]); 
}

/**
 * Calls cuda kernel function recursively to get total sum reduction
 * a: Array to be summed
 * length: length of array to be summed
*/
void applyReduction(int *vect, int length, float *gpuTimes){
	cudaEvent_t gpuStart,gpuStop;
	
	// holds each time for computation / copy of each kernel call
	float copyTo, computationTime, copyFrom;
	
	int *vect_dev;
	int sumSize = ceil((float)length / (2 * BLOCK_SIZE)); // size of sum array after each iteration
	
	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil((float)length / dimBlock.x), 1, 1);
	
	// allocate vectors for gpu
	cudaMalloc((void **)(&vect_dev), length * sizeof(int));
	
	// Begin measuring time for copying memory over to device
	// cudaEventCreate(&gpuStart);
	// cudaEventCreate(&gpuStop);
	// cudaEventRecord(gpuStart,0);
	
	// copy array a (host) to dev_a (device)
	cudaMemcpy(vect_dev, vect, length * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Finish measuring time for copying memory over to device
	// cudaEventRecord(gpuStop,0);
	// cudaEventSynchronize(gpuStop);
	// cudaEventElapsedTime(&copyTo,gpuStart,gpuStop);
    // cudaEventDestroy(gpuStart);
    // cudaEventDestroy(gpuStop);
	
	
	// Begin measuring GPU computation time
	// cudaEventCreate(&gpuStart);
	// cudaEventCreate(&gpuStop);
	// cudaEventRecord(gpuStart,0);
	
	// Launch kernels for reduction
	deviceSumReduction<<<dimGrid, dimBlock>>>(vect_dev, length);
	cudaDeviceSynchronize();
	
	// Finish measuring GPU computation time
	// cudaEventRecord(gpuStop,0);
	// cudaEventSynchronize(gpuStop);
	// cudaEventElapsedTime(&computationTime,gpuStart,gpuStop);
    // cudaEventDestroy(gpuStart);
    // cudaEventDestroy(gpuStop);
	
	// Begin measuring time for copying memory back to host
	// cudaEventCreate(&gpuStart);
	// cudaEventCreate(&gpuStop);
	// cudaEventRecord(gpuStart,0);
	
	// copy results from gpu back to host
	cudaMemcpy(vect, vect_dev, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// Finish measuring time for copying memory back to host
	// cudaEventRecord(gpuStop,0);
	// cudaEventSynchronize(gpuStop);
	// cudaEventElapsedTime(&copyFrom,gpuStart,gpuStop);
    // cudaEventDestroy(gpuStart);
    // cudaEventDestroy(gpuStop);
	
	// free allocated device memory
	cudaFree(vect_dev);
	cudaDeviceSynchronize();
	
	printf("\nFOR sum size of %d\n", sumSize);
	for(int i = 0; i < sumSize; i++){
		
		printf("%d ", vect[i]);
	}
	printf("\n");
	
	// update total times
	// gpuTimes[0] += copyTo;
	// gpuTimes[1] += computationTime;
	// gpuTimes[2] += copyFrom;
	
	// apply reduction again on sum array, if applicable
	if(sumSize > 1)
		return applyReduction(vect, sumSize, gpuTimes);
}

int main(void){
	printf("VECTOR OF SIZE: %d\nBLOCK SIZE: %d\n\n", N, BLOCK_SIZE);
	
	// passed to helper function "apply reduction" to get transfer to time [0],
	// computation time [1] and transfer back time [2]
	float gpuTimes[3];

	// allocate vector for cpu
	int *a = (int*)malloc(sizeof(int)* N);
	int *b = (int*)malloc(sizeof(int)* N);
	
	// initialize vector
	int init =1325;
	for(int i=0;i<N;i++){
		init=3125*init%65521;
		a[i]=(init-32768)/16384;
	}
	
	memcpy(b, a, sizeof(int)* N);
	
	// run reduction on gpu device
	applyReduction(b, N, gpuTimes);
	int gpuSum = b[0];
	
	// variables used to measure cpu computation time
	clock_t cpuStart, cpuEnd;
	float cpuTimeTaken;
	
	// start measuring cpu computation time
	cpuStart = clock();
	
	// run sum reduction on host
	int cpuSum = hostSumReduction(a, N);
	
	// stop measuring cpu computation time
	cpuEnd = clock();
	cpuTimeTaken = ((float)cpuEnd - cpuStart)/CLOCKS_PER_SEC; // in seconds 
	
	
	printf("GPU SUM: %d\n", gpuSum);
	printf("CPU SUM: %d\n", cpuSum);
	
	printf("\nCPU Time: %f\n", cpuTimeTaken);
	printf("GPU Time: %f\n", gpuTimes[1]);
	printf("Memory Transfer Time: %f\n", gpuTimes[0] + gpuTimes[2]);
	
	if(cpuSum == gpuSum)
		printf("TEST PASSED!\n");
	else 
		printf("TEST FAILED!\n");
	
	// free system memory
	free(a);
	free(b);

	return 0;
}