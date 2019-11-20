/**
 * Nathan Dunn
 * Project 3: Work Efficient Parallel Reduction and Work Efficient Parallel Prefix Sum
 * Professor Liu
 * CS-4370-90
 * 11-18-19
*/

#include <stdio.h>
#include <cuda.h>

#define N 2048
#define BLOCK_SIZE 1024


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
__global__ void work_efficient_scan_kernel(int *x, int *y, int *sum_arr, int InputSize){

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
	
	sum_arr[blockIdx.x] = x[start + blockDim.x + t];
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
		printf("|%d", a[i]);
	}
	printf("|\n");
}

/**
	Performs the initial prefix sum on the vector.
*/
void initialPrefixSum(int *vect, int *gpu_sum, int *sum_arr, int length){
	
	int sumArraySize = ceil((float)N / (2 * BLOCK_SIZE));
	
	int *vect_dev, *gpu_sum_dev, *sum_arr_dev;
	
	// block and grid initialization for gpu
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(N / dimBlock.x), 1, 1);
	
	// allocate device memory
	cudaMalloc((void **)(&vect_dev), N * sizeof(int));
	cudaMalloc((void **)(&gpu_sum_dev), N * sizeof(int));
	cudaMalloc((void **)(&sum_arr_dev), sumArraySize * sizeof(int));
	
	// copy vector on host to gpu device
	cudaMemcpy(vect_dev, vect, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	// Launch kernels for sum
	work_efficient_scan_kernel<<<dimGrid, dimBlock>>>(gpu_sum_dev, vect_dev, sum_arr_dev, N);
	cudaDeviceSynchronize();
	
	// copy sum scan vector on device back to host
	cudaMemcpy(gpu_sum, gpu_sum_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// copy block sum vector on device back to host
	cudaMemcpy(sum_arr, sum_arr_dev, sumArraySize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// free system and device memory
	cudaFree(vect_dev);
	cudaFree(gpu_sum_dev);
	cudaFree(sum_arr_dev);
}



int main(void){
	
	printf("\nVECTOR SIZE: %d\nBLOCK SIZE: %d\n\n", N, BLOCK_SIZE);
	int sumArraySize = ceil((float)N / (2 * BLOCK_SIZE));
	int *vect, *cpu_sum, *gpu_sum, *sum_arr, *sum_arr_prefix;
	
	// initialize cpu vectors
	vect = (int*)malloc(sizeof(int) * N); // original vector
	cpu_sum = (int*)malloc(sizeof(int) * N); // stores cpu prefix sum
	gpu_sum = (int*)malloc(sizeof(int) * N); // stores copied gpu prefix sum
	sum_arr = (int*)malloc(sizeof(int) * sumArraySize); // stores block sums
	sum_arr_prefix = (int*)malloc(sizeof(int) * 2 * BLOCK_SIZE); // stores prefix sum of block sum
	
	// initialize vect
	int init = 1325;
	for (int i = 0; i < N; i++){
		init = 3125 * init % 65521;
		vect[i] = (init - 32768) / 16384;
	}
	
	// perform initial sum on vector and then prefix sum on the sum array (if applicable)
	initialPrefixSum(vect, gpu_sum, sum_arr, N);
	
	// perform prefix sum on cpu
	hostPrefixSum(cpu_sum, vect, N);
	
	printf("GPU prefix sum:\n");
	//printVector(gpu_sum, N);
	
	printf("CPU prefix sum:\n");
	//printVector(cpu_sum, N);

	if(verify(gpu_sum, cpu_sum, N))
		printf("\nTEST PASSED!\n");
	else
		printf("\nTEST FAILED!\n");
	
	// free system memory
	free(vect);
	free(cpu_sum);
	free(gpu_sum);
	free(sum_arr);
	free(sum_arr_prefix);
	
	return 0;	
}