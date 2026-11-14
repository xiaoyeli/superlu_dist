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

#define SLU_GPU_KEY_LEN (MPI_MAX_PROCESSOR_NAME + 16)
static int slu_gpu_key_cmp(const void *a, const void *b)
{
    return memcmp(a, b, SLU_GPU_KEY_LEN);
}

/*! \brief Count the processes of a communicator on each GPU.
 *
 * A GPU is identified by its node and its UUID, so the count does not depend
 * on how the devices are numbered in each process.  Collective on comm.
 *
 * \param[out] min_per_gpu  smallest number of processes sharing one GPU
 * \param[out] max_per_gpu  largest number of processes sharing one GPU
 * \return 1 if some GPU holds exactly one process while another holds several,
 *         0 otherwise.  NVSHMEM cannot be initialized on such a layout: the
 *         processes take different paths through its initialization and wait
 *         for each other forever (seen with NVSHMEM 3.2.5).
 */
int superlu_gpu_sharing_is_mixed(MPI_Comm comm, int *min_per_gpu, int *max_per_gpu)
{
    int nranks, dev, len, i, cnt, mn, mx;
    char key[SLU_GPU_KEY_LEN], *all;
    cudaDeviceProp prop;

    MPI_CHECK(MPI_Comm_size(comm, &nranks));
    memset(key, 0, SLU_GPU_KEY_LEN);
    MPI_CHECK(MPI_Get_processor_name(key, &len));
    memset(key + len, 0, SLU_GPU_KEY_LEN - len);
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    memcpy(key + MPI_MAX_PROCESSOR_NAME, &prop.uuid, 16);

    if ( !(all = (char *) malloc((size_t) nranks * SLU_GPU_KEY_LEN)) ) {
        fprintf(stderr, "Malloc fails for the GPU layout check.\n");
        exit(-1);
    }
    MPI_CHECK(MPI_Allgather(key, SLU_GPU_KEY_LEN, MPI_CHAR,
                            all, SLU_GPU_KEY_LEN, MPI_CHAR, comm));
    qsort(all, nranks, SLU_GPU_KEY_LEN, slu_gpu_key_cmp);
    mn = nranks; mx = 0; cnt = 1;
    for (i = 1; i <= nranks; ++i) {
        if (i < nranks && !memcmp(all + (size_t) (i-1) * SLU_GPU_KEY_LEN,
                                  all + (size_t) i * SLU_GPU_KEY_LEN, SLU_GPU_KEY_LEN)) {
            ++cnt;
        } else {
            if (cnt < mn) mn = cnt;
            if (cnt > mx) mx = cnt;
            cnt = 1;
        }
    }
    free(all);
    if (min_per_gpu) *min_per_gpu = mn;
    if (max_per_gpu) *max_per_gpu = mx;
    return mn == 1 && mx > 1;
}

void nv_init_wrapper( MPI_Comm mpi_comm)
{
#ifdef HAVE_NVSHMEM    
    int rank, nranks, ndevices;
    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;
    int min_per_gpu, max_per_gpu;

    // MPI_CHECK(MPI_Init(c, &v));
    //MPI_CHECK(MPI_Init_thread( c, &v, MPI_THREAD_MULTIPLE, omp_mpi_level));
    MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));
    MPI_CHECK(MPI_Comm_size(mpi_comm, &nranks));

    /* Stop here instead of hanging in the initialization of NVSHMEM. */
    if (superlu_gpu_sharing_is_mixed(mpi_comm, &min_per_gpu, &max_per_gpu)) {
        if (rank == 0) {
            fprintf(stderr,
                "SuperLU_DIST: the GPU triangular solve cannot run on this layout: the %d\n"
                "processes of the 2D process grid are spread unevenly over the GPUs, with one\n"
                "process on some GPU and up to %d on another, and NVSHMEM hangs in its\n"
                "initialization on such a layout.  Give every GPU one process, or every GPU\n"
                "at least two: use a number of processes per 2D grid that is at most the\n"
                "number of GPUs or a multiple of it, or set SUPERLU_MPI_PROCESS_PER_GPU\n"
                "(with SUPERLU_BIND_MPI_GPU) so that no GPU is left with a single process.\n",
                nranks, max_per_gpu);
            fflush(stderr);
        }
        MPI_Barrier(mpi_comm);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }


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