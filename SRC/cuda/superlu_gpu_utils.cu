#include "superlu_defs.h"
#include "gpu_api_utils.h"

#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <cooperative_groups.h>
#include <nvml.h>
#endif

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
     do {                                                                          \
         cudaError_t result = (stmt);                                              \
         if (cudaSuccess != result) {                                              \
             fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                     cudaGetErrorString(result));                                  \
             exit(-1);                                                             \
         }                                                                         \
         assert(cudaSuccess == result);                                            \
     } while (0)

#undef MPI_CHECK
#define MPI_CHECK(stmt)                                 \
 do {                                                    \
     int result = (stmt);                                \
     if (MPI_SUCCESS != result) {                        \
         fprintf(stderr, "[%s:%d] MPI failed with error %d \n",\
          __FILE__, __LINE__, result);                   \
         exit(-1);                                       \
     }                                                   \
 } while (0)

#define NVSHMEM_CHECK(stmt)                               \
 do {                                                    \
     int result = (stmt);                                \
     if (cudaSuccess != result) {                      \
         fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n",\
          __FILE__, __LINE__, result);                   \
         exit(-1);                                       \
     }                                                   \
 } while (0)



// /*error reporting functions */
// gpuError_t checkGPU(gpuError_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
//     if (result != gpuSuccess) {
//         fprintf(stderr, "GPU Runtime Error: %s\n", gpuGetErrorString(result));
//         assert(result == gpuSuccess);
//     }
// #endif
//     return result;
// }


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
__device__ void incScan(pfx_dtype *inOutArr, pfx_dtype *temp, int n)
{
    // extern __shared__ pfx_dtype temp[];
    int n_original = n;
    n = (n & (n - 1)) == 0? n: dnextpow2(n);
    int thread_id = threadIdx_x;
    int offset = 1;
    if(2*thread_id  < n_original)
        temp[2*thread_id] = inOutArr[2*thread_id]; 
    else 
        temp[2*thread_id] =0;


    if(2*thread_id+1 <n_original)
        temp[2*thread_id+1] = inOutArr[2*thread_id+1];
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
    inOutArr[2*thread_id] = temp[2*thread_id]+ inOutArr[2*thread_id]; // write results to device memory
    if(2*thread_id+1  < n_original)
    inOutArr[2*thread_id+1] = temp[2*thread_id+1]+ inOutArr[2*thread_id+1];
    __syncthreads();
    
} /* end incScan */ 


#if 0 // Not used
__global__ void gExScan(pfx_dtype *inArr, int n)
{
    extern __shared__ pfx_dtype temp[];
    incScan(inArr, temp, n);
    
}
#endif



#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CUDA
void nv_init_wrapper( MPI_Comm mpi_comm)
{
#ifdef HAVE_NVSHMEM    
    int rank, nranks, ndevices;
    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;

    // MPI_CHECK(MPI_Init(c, &v));
    //MPI_CHECK(MPI_Init_thread( c, &v, MPI_THREAD_MULTIPLE, omp_mpi_level));
    MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));
    MPI_CHECK(MPI_Comm_size(mpi_comm, &nranks));


    attr.mpi_comm = &mpi_comm;
    NVSHMEM_CHECK(nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    
    /* Yang: is it safe to call it here?; commenting this out will cause "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed" */
    // CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    // CUDA_CHECK(cudaSetDevice(rank%ndevices));

    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_CHECK(MPI_Get_processor_name(name, &resultlength));
    int get_cur_dev;
    CUDA_CHECK(cudaGetDevice(&get_cur_dev));

    // cudaDeviceProp prop;
    //CUDA_CHECK(cudaGetDeviceProperties(&prop, rank%ndevices));
    // CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node)); // Yang Liu: this line is causing runtime error
    // //int status=nvshmemx_init_status();
    // printf("** MPI %d/%d, NVSHMEM %d/%d,device name: %s bus id: %d, "
    //        "ndevices=%d,cur=%d, node=%s **\n",
    //        rank,nranks,mype,npes,prop.name, prop.pciBusID,
    //        ndevices,get_cur_dev,name);
    // fflush(stdout);


    //int *target;
    //target = (int *)nvshmem_malloc(sizeof(int)*256);
    //printf("(%d) nvshmem malloc target success\n",mype);
    //fflush(stdout);
    //simple_shift<<<1, 256>>>(target, mype, npes);
    //CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
#endif


// void nv_init_wrapper(int* c, char *v[], int* omp_mpi_level)
// {
//     int rank, nranks, ndevices;
//     MPI_Comm mpi_comm;
//     nvshmemx_init_attr_t attr;
//     int mype, npes, mype_node;

//     MPI_CHECK(MPI_Init(c, &v));
//     //MPI_CHECK(MPI_Init_thread( c, &v, MPI_THREAD_MULTIPLE, omp_mpi_level));
//     MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
//     MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));


//     mpi_comm = MPI_COMM_WORLD;
//     attr.mpi_comm = &mpi_comm;
//     NVSHMEM_CHECK(nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
//     mype = nvshmem_my_pe();
//     npes = nvshmem_n_pes();
//     mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
//     CUDA_CHECK(cudaSetDevice(mype_node));

//     char name[MPI_MAX_PROCESSOR_NAME];
//     int resultlength;
//     MPI_CHECK(MPI_Get_processor_name(name, &resultlength));
//     int get_cur_dev;
//     CUDA_CHECK(cudaGetDeviceCount(&ndevices));
//     CUDA_CHECK(cudaGetDevice(&get_cur_dev));

//     cudaDeviceProp prop;
//     //CUDA_CHECK(cudaGetDeviceProperties(&prop, rank%ndevices));
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node));
//     //int status=nvshmemx_init_status();
//     printf("** MPI %d/%d, NVSHMEM %d/%d, mype_node=%d, device name: %s bus id: %d, "
//            "ndevices=%d,cur=%d, node=%s **\n",
//            rank,nranks,mype,npes,mype_node, prop.name, prop.pciBusID,
//            ndevices,get_cur_dev,name);
//     fflush(stdout);


//     //int *target;
//     //target = (int *)nvshmem_malloc(sizeof(int)*256);
//     //printf("(%d) nvshmem malloc target success\n",mype);
//     //fflush(stdout);
//     //simple_shift<<<1, 256>>>(target, mype, npes);
//     //CUDA_CHECK(cudaDeviceSynchronize());

// }

// void nv_init_wrapper(MPI_Comm mpi_comm1)
// {
//     int rank, nranks, ndevices;
//     MPI_Comm mpi_comm;
//     nvshmemx_init_attr_t attr;
//     int mype, npes, mype_node;

//     // MPI_CHECK(MPI_Init(c, &v));
//     //MPI_CHECK(MPI_Init_thread( c, &v, MPI_THREAD_MULTIPLE, omp_mpi_level));
//     MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
//     MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));


//     mpi_comm = MPI_COMM_WORLD;
//     attr.mpi_comm = &mpi_comm;
//     NVSHMEM_CHECK(nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
//     mype = nvshmem_my_pe();
//     npes = nvshmem_n_pes();
//     CUDA_CHECK(cudaGetDeviceCount(&ndevices));
//     CUDA_CHECK(cudaSetDevice(rank%ndevices));

//     char name[MPI_MAX_PROCESSOR_NAME];
//     int resultlength;
//     MPI_CHECK(MPI_Get_processor_name(name, &resultlength));
//     int get_cur_dev;
//     CUDA_CHECK(cudaGetDevice(&get_cur_dev));

//     cudaDeviceProp prop;
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, rank%ndevices));
//     //int status=nvshmemx_init_status();
//     printf("** MPI %d/%d, NVSHMEM %d/%d,device name: %s bus id: %d, "
//            "ndevices=%d,cur=%d, node=%s **\n",
//            rank,nranks,mype,npes,prop.name, prop.pciBusID,
//            ndevices,get_cur_dev,name);
//     fflush(stdout);


//     //int *target;
//     //target = (int *)nvshmem_malloc(sizeof(int)*256);
//     //printf("(%d) nvshmem malloc target success\n",mype);
//     //fflush(stdout);
//     //simple_shift<<<1, 256>>>(target, mype, npes);
//     //CUDA_CHECK(cudaDeviceSynchronize());

// }

#ifdef __cplusplus
}
#endif