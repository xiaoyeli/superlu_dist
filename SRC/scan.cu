#include <iostream>
#include <cstdio>

// typedef float pfx_dtype ; 

int nextpow2(int v)

{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

__device__ int dnextpow2(int v)

{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}



typedef int pfx_dtype ; 
__global__ void prescan(pfx_dtype *outArr, pfx_dtype *inArr, int n)
{
    extern __shared__ pfx_dtype temp[];
    int n_original = n;
    n = (n & (n - 1)) == 0? n: dnextpow2(n);
    int thread_id = threadIdx.x;
    int offset = 1;
    if(2*thread_id  < n_original)
        temp[2*thread_id] = inArr[2*thread_id]; 
    else 
        temp[2*thread_id] =0;


    if(2*thread_id+1 <n_original)
        temp[2*thread_id+1] = inArr[2*thread_id+1];
    else 
        temp[2*thread_id+1] =0;
    
    for (int d = n>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset*(2*thread_id+1)-1;
            int bi = offset*(2*thread_id+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (thread_id == 0) { temp[n - 1] = 0; } 
    for (int d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset*(2*thread_id+1)-1;
            int bi = offset*(2*thread_id+2)-1;
            pfx_dtype t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if(2*thread_id  < n_original)
    outArr[2*thread_id] = temp[2*thread_id]+ inArr[2*thread_id]; // write results to device memory
    if(2*thread_id+1  < n_original)
    outArr[2*thread_id+1] = temp[2*thread_id+1]+ inArr[2*thread_id+1];
    __syncthreads();
    if(2*thread_id  < n_original)
    printf("xA[%d] = %d \n",2*thread_id , outArr[2*thread_id]);
    if(2*thread_id+1  < n_original)
    printf("xA[%d] = %d \n",2*thread_id+1 , outArr[2*thread_id+1]);
    __syncthreads();
} 

#define SELF_TEST 
#ifdef SELF_TEST

#include <iostream>
#include "cub/cub.cuh"

#define THREAD_BLOCK_SIZE 8


// __global__
// void cub_scan_test(int N)
// {
// 	int thread_id = threadIdx.x;
// 	typedef cub::BlockScan<int, THREAD_BLOCK_SIZE > BlockScan; /*1D int data type*/

// 	__shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

// 	extern __shared__ int* IndirectJ1;
// 	extern __shared__ int* IndirectJ2= IndirectJ1+ N*sizeof(int);

// 	if (thread_id < N)
// 	{
// 		IndirectJ1[thread_id] = 2*thread_id +1;
// 	}

// 	__syncthreads();
// 	if (thread_id < THREAD_BLOCK_SIZE)
// 		BlockScan(temp_storage).InclusiveSum (IndirectJ1[thread_id], IndirectJ2[thread_id]);


// 	if (thread_id < THREAD_BLOCK_SIZE)
// 		printf("%d %d\n", thread_id, IndirectJ2[thread_id]);

// }



// extern __shared__
// #define THREAD_BLOCK_SIZE 7

__global__ void initData(pfx_dtype* A, int n)
{
    int threadId = threadIdx.x;   
    if(threadId<n)
        A[threadId] = 2*threadId+1;
        printf("A[%d] = %d \n",threadId,A[threadId]);
}



int main(int argc, char* argv[])
{
    if(argc<2) 
    {
        std::cout<<"Error with number of arguments\n";
        return -1;
    }
    int N = atoi(argv[1]);
    int N2=N;
    if((N & (N - 1)) == 0)
    {
        std::cout<<"Power of Two\n";
    }
    else
    {
        std::cout<<"Not a power of Two\n";
        N2 = nextpow2(N);
        std::cout<<"Using "<<N2<<"\n";
    } 
        
    
    
    
    
    pfx_dtype *A, *xA;
    cudaMalloc(&A, sizeof(pfx_dtype)*N);
    cudaMalloc(&xA, sizeof(pfx_dtype)*N);
    

    initData<<< 1,THREAD_BLOCK_SIZE >>> (A,N);
    if(cudaDeviceSynchronize() != cudaSuccess)
        std::cout<<"Error- 0\n";
    // prescan<<<  1,THREAD_BLOCK_SIZE/2,2*THREAD_BLOCK_SIZE*sizeof(pfx_dtype) >>> (xA, A, N);
    
    prescan<<<  1,(N+1)/2,2*N*sizeof(pfx_dtype) >>> (xA, A, N);
    prescan<<<  1,N2,2*N*sizeof(pfx_dtype) >>> (xA, A, N);
    if(cudaDeviceSynchronize() != cudaSuccess)
        std::cout<<".....EXITING\n";   
    else
        std::cout<<"No errors reported\n";


    // typedef cub::BlockScan<int, THREAD_BLOCK_SIZE> BlockScan; /*1D int data type*/
	// __shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

    // cub_scan_test <<<  1,THREAD_BLOCK_SIZE >>> (N);

    return 0;
}

#endif 