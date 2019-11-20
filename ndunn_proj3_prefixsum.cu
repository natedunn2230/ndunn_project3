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
#define BLOCK_SIZE 1


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
__global__ void work_efficient_scan_kernel(int *x, int *y, int *block_sum, int InputSize){

 	__shared__ int scan_array[2 * BLOCK_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	scan_array[t] = y[start + t];
	scan_array[blockDim.x + t] = y[start + blockDim.x + t];
	
	__syncthreads();

	// Perform reduction step
   int reduction_stride = 1;
   while(reduction_stride <= BLOCK_SIZE){
        int index = (threadIdx.x + 1) * reduction_stride * 2 - 1;
		
        if(index < 2 * BLOCK_SIZE)
            scan_array[index] += scan_array[index-reduction_stride];

        reduction_stride = reduction_stride * 2;

        __syncthreads();
    }

	// Perform post scan step
    int post_stride = BLOCK_SIZE / 2;
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
	
	block_sum[blockIdx.x] = x[start + blockDim.x + t];
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

/**
	Performs prefix sum on vector 'a'
*/
void applyPrefixSum(int *vect, int *gpu_sum, int length){
	int *block_sum, *vect_dev, *gpu_sum_dev, *block_sum_dev;
	
	block_sum = (int*)malloc(sizeof(int) * (N / ( 2 * BLOCK_SIZE))); // stores copied gpu prefix sum
	
	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(N / dimBlock.x), 1, 1);
	
	// allocate device memory
	cudaMalloc((void **)(&vect_dev), N * sizeof(int));
	cudaMalloc((void **)(&gpu_sum_dev), N * sizeof(int));
	cudaMalloc((void **)(&block_sum_dev), (N / ( 2 * BLOCK_SIZE)) * sizeof(int));
	
	// copy vector on host to gpu device
	cudaMemcpy(vect_dev, vect, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Launch kernels for sum
	work_efficient_scan_kernel<<<dimGrid, dimBlock>>>(gpu_sum_dev, vect_dev, block_sum_dev, N);
	cudaDeviceSynchronize();
	
	// copy sum scan vector on device back to host
	cudaMemcpy(gpu_sum, gpu_sum_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// copy block sum vector on device back to host
	cudaMemcpy(block_sum, block_sum_dev, (N / ( 2 * BLOCK_SIZE)) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	printf("Block sum: \n");
	printVector(block_sum, (N / (BLOCK_SIZE * 2)));
	
	// free system and device memory
	free(block_sum);
	cudaFree(vect_dev);
	cudaFree(gpu_sum_dev);
	cudaFree(block_sum_dev);
}



int main(void){
	
	printf("\nVECTOR SIZE: %d\nBLOCK SIZE: %d\n\n", N, BLOCK_SIZE);
	
	int *vect, *cpu_sum, *gpu_sum;
	
	// initialize cpu vectors
	vect = (int*)malloc(sizeof(int) * N); // original vector
	cpu_sum = (int*)malloc(sizeof(int) * N); // stores cpu prefix sum
	gpu_sum = (int*)malloc(sizeof(int) * N); // stores copied gpu prefix sum
	
	// initialize vect
	int init = 1325;
	for (int i = 0; i < N; i++){
		init = 3125 * init % 65521;
		vect[i] = (init - 32768) / 16384;
	}
	
	applyPrefixSum(vect, gpu_sum, N);
	
	// perform prefix sum on cpu
	hostPrefixSum(cpu_sum, vect, N);
	
	printf("GPU prefix sum:\n");
	printVector(gpu_sum, N);
	
	printf("CPU prefix sum:\n");
	printVector(cpu_sum, N);

	
	if(verify(gpu_sum, cpu_sum, N))
		printf("\nTEST PASSED!\n");
	else
		printf("\nTEST FAILED!\n");
	
	// free system memory
	free(vect);
	free(cpu_sum);
	free(gpu_sum);

	return 0;	
}