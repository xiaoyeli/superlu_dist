#ifdef HAVE_NVSHMEM  
#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#endif


/* ************************************************* */
/*  reuse L and U                                    */
/*  *_nfrecv: totol number of received msg           */
/*  d_*: device memory                               */
/*  h_*: host memory                                 */
/*  d_mynum: number of msg I expected // each thread */
/*  d_mymaskstart: start index point at d_column     */
/* ************************************************  */

extern uint64_t *flag_bc_q;
extern uint64_t *flag_rd_q;
extern int *my_flag_bc;
extern int *my_flag_rd;
extern int *mystatus;
extern int *mystatusmod;
extern int *mystatus_u;
extern int *mystatusmod_u;
extern int *d_msgnum;
extern int *d_status;
extern int *d_statusmod;

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
extern int *d_colnum;
extern int *d_mynum;
extern int *d_mymaskstart;
extern int *d_mymasklength;
extern int *d_rownum;
extern int *d_rowstart;


/* ************************************************* */
/*  for block row reduction                        */
/*  *_nfrecv: totol number of received msg           */
/*  d_*: device memory                               */
/*  h_*: host memory                                 */
/*  d_mynum: number of msg I expected // each thread */
/*  d_mymaskstart: start index point at d_column     */
/* ************************************************  */

extern int *d_nfrecvmod;
extern int *h_nfrecvmod;
extern int *d_statusmod;
extern int *d_colnummod;
extern int *d_mynummod;
extern int *d_mymaskstartmod;
extern int *d_mymasklengthmod;
extern int *d_recv_cnt;
extern int *d_flag_mod;


/* ************************************************* */
/*  U solve                                          */
/* ************************************************  */
extern int *d_nfrecv_u;
extern int *h_nfrecv_u;
extern int *d_colnum_u;
extern int *d_mynum_u;
extern int *d_mymaskstart_u;
extern int *d_mymasklength_u;
extern int *d_rownum_u;
extern int *d_rowstart_u;
extern int *d_nfrecvmod_u;
extern int *h_nfrecvmod_u;
extern int *d_colnummod_u;
extern int *d_mynummod_u;
extern int *d_mymaskstartmod_u;
extern int *d_mymasklengthmod_u;
extern int *d_recv_cnt_u;
extern int *d_msgnum_u;
extern int *d_flag_mod_u;


#define RDMA_FLAG_SIZE 2


#ifdef HAVE_NVSHMEM 
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
#endif



