#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern int mysendmsg_num;
extern int mysendmsg_num_u;
extern int mysendmsg_num_rd;
extern int mysendmsg_num_urd;
extern int *mystatus;
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