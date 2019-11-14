# Project 3: Work Efficient Parallel Reduction / Work Efficient Parallel Prefix Sum
#### CS-4370-90
#### Nathan Dunn
#### Professor Liu
#### 11-18-19
### How To Build / Run

Two files can be found for this project:

1. ndunn_proj3_prefixsum.cu
2. ndunn_proj3_reduction.cu

In the Fry environment, enter the following command to compile the source files: 

1. ` singularity exec --nv /home/containers/cuda92.sif nvcc ndunn_proj3_reduction.cu -o dunn_reduc`
2. ` singularity exec --nv /home/containers/cuda92.sif nvcc ndunn_proj3_prefixsum.cu -o dunn_prefix`

To run the programs:

1. `./dunn_reduc`
2. `./dunn_prefix`