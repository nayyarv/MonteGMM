#include <stdlib.h>
#include <math.h>


#define MAXDIM 30
#define MAXTHREADS 512

#define PI 3.1415926535

__device__ float normalDistribution(float* x, float* mu, float* diagonalCovariance, unsigned int dim){
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


__global__ void likelihoodKernel(float **Xpoints, float **means, float **diagCovs, 
	float* weights, unsigned int dim, unsigned int numPoints, unsigned int numMixtures, float* finalLikelihood)
{
	/*
	Xpoints - 2d array of points, numPoints rows of vectors of dim length
		Xpoints[numPoints][dim]
	Means - 2d array of means, numMixtures rows of vectors of dim
		Means[numMixtures][dim]
	diagCovs - 2d array of cov diagonals, ditto
		diagCovs[numMixtures][dim]
	weights - 1d array of length numMixtures
		weights[numMixtures]

	finalLikelihood: Likelihood value that we want to return
		finalLikelihood[blockIdx.x]

	Since threads are usually a power of 2, we have to check if we're out of bounds 
	with regards to the data.
	*/

	__shared__ float sarray[MAXTHREADS]; //Nthreads
	int index = blockIdx.x * blockDim.x + threadIdx.x; 	
	int threadIndex = threadIdx.x;

	sarray[index] = 0;
	__syncthreads();
	//Following CUDA guidelines here for quick reduction
	//TODO: Speed up computation by having a block per mixture? 


	if (numPoints>threadIndex)
	{
		// Just make sure we have stuff to compute
		
		//Will contain the id of the x value

		float value = 0;

		for (int i = 0; i < numMixtures; ++i)
		{
			value += weights[i] * normalDistribution(Xpoints[index], means[i], diagCovs[i], dim);
		}

		sarray[threadIndex] = logf(value);
	} 
	else 
	{
		sarray[threadIndex] = 0; //I.e it's zero
	}

	//Reduction 
	// Courtesy Vidhya Sethu 2014 
	// My version was terrible
	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s>>=1)
	{
		if (threadIndex<s)
		{
			sarray[threadIndex] += sarray[threadIndex+s];
		}
		__syncthreads();
	}

	if (threadIndex==0) //Since everything has been synced, sarray[0] now holds our result
	{
		finalLikelihood[blockIdx.x] = sarray[0];
	}



}

int main(void){
	return 0;
}

/*
int main(void){
	printf("Hello World!\n");

	//Code to read in data/ Generate
	//Need Cython Wrappers ASAP!!


	int num_elements = 100;
	int num_bytes = num_elements * sizeof(float);

	float *device_array = 0;
	float *host_array = 0;

	// allocate memory
	host_array = (float*)malloc(
		);

	cudaMalloc((void**)&device_array, num_bytes);


	likelihoodKernel<<<10,10>>>(device_array);
	
	cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

	printf("likelihoodKernel results:\n");
	for(int i = 0; i < num_elements; ++i)
	{
		printf("%f \n", host_array[i]);
	}
	printf("\n\n");


	return 0;
}

*/