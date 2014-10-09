#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI 3.1415926535


__device__ float normalDistribution(float* x, float* mu, float* diagonalCovariance, int dim){
	/*
	x:  individual point being evaluated, x[dim]
	mu: mean of the normal distribution being evaluated  mu[dim]
	diagonalCovariance: for the norm dist  diagonalCovariance[dim]
	
	dim: dimensionality of the distribution, also equal to length of the previous vectors

	Evaluates the normalDistribution on the GPU, for diagonal Covariance Matrices only.

	*/
	float total = 0;
	float det = 1;
	float finval = 0;
	float denom = 0;
	float temp = 0;

	for (int i = 0; i < dim; ++i)
	{
		temp = (x[i]-mu[i]);
		temp *= temp; // Square it
		total += temp * diagonalCovariance[i];
		//Note this is the stuff that goes inside the normal
		det *= diagonalCovariance[i];
	}
	
	total*=-1/2.0;

	finval = expf(total);

	denom = powf(2*PI, dim) * det; 

	return (rsqrtf(denom) * finval);
}

__global__ void kernel1(float *array)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float x[2] = {0.1* threadIdx.x, 0.1* threadIdx.x};
	float mu[2] = {-0.1*blockIdx.x,-0.1*blockIdx.x};
	float diagonalCovariance[2] = {1,1};

	array[index] = normalDistribution(x, mu, diagonalCovariance, 2);

}



int main(void){
	printf("Hello World!\n");


	int num_elements = 100;
	int num_bytes = num_elements * sizeof(int);

	float *device_array = 0;
	float *host_array = 0;

	// allocate memory
	host_array = (float*)malloc(
		);
	cudaMalloc((void**)&device_array, num_bytes);


	kernel1<<<10,10>>>(device_array);
	
	cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

	printf("kernel1 results:\n");
	for(int i = 0; i < num_elements; ++i)
	{
		printf("%f \n", host_array[i]);
	}
	printf("\n\n");


	return 0;
}