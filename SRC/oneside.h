#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
extern int *flag_bc_q;
extern int *flag_rd_q;
extern int *my_flag_bc;
extern int *my_flag_rd;
extern double *ready_x;
extern double *ready_lsum;
extern int *mystatus;
extern int *d_launch_flag;
extern int *d_nfrecv;
extern int *d_status;
#define RDMA_FLAG_SIZE 4
#define NVSHMEM_SIZES 10

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

#define MPI_CHECK(stmt)                                 \
do {                                                    \
    int result = (stmt);                                \
    if (MPI_SUCCESS != result) {                        \
        fprintf(stderr, "[%s:%d] MPI failed with error %d \n",\
         __FILE__, __LINE__, result);                   \
        exit(-1);                                       \
    }                                                   \
} while (0)

__device__ int clockrate;