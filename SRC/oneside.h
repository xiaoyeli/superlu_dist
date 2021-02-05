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
extern int *mystatusmod;
extern int *d_launch_flag;

/* ************************************************* */
/*  for block column broadcast                       */
/*  *_nfrecv: totol number of received msg           */
/*  d_*: device memory                               */
/*  h_*: host memory                                 */
/*  d_mynum: number of msg I expected // each thread */
/*  d_mymaskstart: start index point at d_column     */
/* ************************************************  */
extern int *d_nfrecv;
extern int *h_nfrecv;
extern int *d_status;
extern int *d_colnum;
extern int *d_mynum;
extern int *d_mymaskstart;
extern int *d_mymasklength;
extern int *d_rownum;
extern int *d_rowstart;
extern int *senddone;
extern int *sumdone;
extern int *d_nfrecvmod;
extern int *h_nfrecvmod;
extern int *d_statusmod;
extern int *d_colnummod;
extern int *d_mynummod;
extern int *d_mymaskstartmod;
extern int *d_mymasklengthmod;
extern int *d_recv_cnt;
extern int *d_msgnum;
extern __device__ int clockrate;
#define RDMA_FLAG_SIZE 4



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