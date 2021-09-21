/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Solves a system of distributed linear equations A*X = B with a
 * general N-by-N matrix A using the LU factors computed previously.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

#ifndef MAXSUPER
#define MAXSUPER 1024  
#endif

#include <stdio.h>
#include "mpi.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>
#include <sched.h>
#include <nvml.h>
#include <omp.h>
#include <cooperative_groups.h>

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


 
 

#ifdef __cplusplus
extern "C" {
#endif


// #define USESHARE1RHS 1


/***************************************************************************//**
	 Does sum reduction of n-element array x, leaving total in x[0].
	 Contents of x are destroyed in the process.
	 With k threads, can reduce array up to 2*k in size.
	 Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
	 Having n as template parameter allows compiler to evaluate some conditions at compile time.
	 Calls __syncthreads before & after reduction.
	 @ingroup magma_kernel
 *******************************************************************************/
__device__ void
magma_sum_reduce( int n, int i, double* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce



/******************************************************************************/
static __device__ void
gemv_device_dlsum_fmod(
        int_t m, int_t n, double alpha,
        const double * __restrict__ A, int_t lda,
        const double * __restrict__ x, int_t incx, double beta,
        double       * __restrict__ y, int_t incy)
{
    if (m <= 0 || n <= 0) return;

    int_t num_threads = DIM_X * DIM_Y;
    int_t thread_id = threadIdx_x + threadIdx_y * blockDim_x;

    // threads are all configurated locally
    int_t tx = thread_id % DIM_X;
    int_t ty = thread_id / DIM_X;

    int_t ind = tx;

    __shared__ double sdata[DIM_X * DIM_Y];


    int_t st = 0;

    int_t ed = min(st+m, CEILING(m,DIM_X)*DIM_X);

    int_t iters = CEILING(ed-st,DIM_X) ;

    double zero = 0.0;

    for (int_t i=0; i < iters; i++)
    {
        if (ind < m ) A += ind;

        double res = zero;

        if (ind < m )
        {
            for (int_t col=ty; col < n; col += DIM_Y)
            {
                res += A[col*lda] * x[col*incx];
            }
        }

        if (DIM_X >= num_threads) // indicated 1D threads configuration. Shared memory is not needed, reduction is done naturally
        {
            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*res + beta*y[ind*incy];
            }
        }
        else
        {
            sdata[ty + tx * DIM_Y] = res;

            __syncthreads();

            if ( DIM_Y > 16)
            {
                magma_sum_reduce(DIM_Y, ty, sdata + tx * DIM_Y);
            }
            else
            {
                if (ty == 0 && ind < m)
                {
                    for (int_t i=1; i < DIM_Y; i++)
                    {
                        sdata[tx * DIM_Y] += sdata[i + tx * DIM_Y];
                    }
                }
            }

            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*sdata[tx * DIM_Y] + beta*y[ind*incy];
            }

            __syncthreads();
        }

        if ( ind < m) A -= ind;

        ind += DIM_X;
    }
}





/******************************************************************************/
static __device__
void gemm_device_dlsum_fmod(
        int_t M, int_t N, int_t K,
        int_t blx, int_t bly,
        const double* __restrict__ A, int_t LDA,
        const double* __restrict__ B, int_t LDB,
        double rC[THR_N][THR_M],
        double alpha, double beta)
{
    // #if (__CUDA_ARCH__ >= 200)
    int_t idx = threadIdx_x;  // thread's m dimension
    int_t idy = threadIdx_y;  // thread's n dimension

    int_t idt = DIM_X * idy + idx;    // thread's global number

    int_t idxA = idt % DIM_XA;    // idx within A
    int_t idyA = idt / DIM_XA;    // idy within A

    int_t idxB = idt % DIM_XB;    // idx within B
    int_t idyB = idt / DIM_XB;    // idy within B

    // int_t blx = blockIdx_x;   // block's m dimension
    // int_t bly = blockIdx_y;   // block's n dimension

    __shared__ double sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ double sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    double rA[THR_M];
    double rB[THR_N];

    double ra[BLK_K/DIM_YA+1][BLK_M/DIM_XA];
    double rb[BLK_N/DIM_YB][BLK_K/DIM_XB+1];

    const double *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    const double *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    int_t boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
    int_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int_t m, n, k, kk;
    double zero = 0.0;

    // Zero C
#pragma unroll
    for (n = 0; n < THR_N; n++)
#pragma unroll
            for (m = 0; m < THR_M; m++)
                rC[n][m] = zero;

#pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
#pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
                sA[n+idyA][m+idxA] = fetch(A, m, n, boundA);

#pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
#pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;

        offs_dB += BLK_K;
        boundB  -= BLK_K;

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
#pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

#pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
#pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

        // Multiply
#pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
#pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
#pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
#pragma unroll
            for (n = 0; n < THR_N; n++) {
#pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(rA[m], rB[n], rC[n][m]);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
#pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

#pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
#pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
#pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
#pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
#pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
#pragma unroll
        for (n = 0; n < THR_N; n++) {
#pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(rA[m], rB[n], rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    // if( beta == make_FloatingPoint_t(0.0,0.0) ) {
    // #pragma unroll
    // for (n = 0; n < THR_N; n++) {
    // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
    // #pragma unroll
    // for (m = 0; m < THR_M; m++) {
    // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
    // if (coord_dCm < M && coord_dCn < N) {
    // int_t offsC = coord_dCn*LDC + coord_dCm;

    // double &regC = rC[n][m];
    // double &memC = C[offsC];

    // // memC = mul(alpha, regC);
    // }
    // }
    // }
    // } else {
    // #pragma unroll
    // for (n = 0; n < THR_N; n++) {
    // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
    // #pragma unroll
    // for (m = 0; m < THR_M; m++) {
    // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
    // if (coord_dCm < M && coord_dCn < N) {
    // int_t offsC = coord_dCn*LDC + coord_dCm;

    // double &regC = rC[n][m];
    // double &memC = C[offsC];

    // // memC = add(mul(alpha, regC), mul(beta, memC));
    // }
    // }
    // }
    // }
    // #endif /* (__CUDA_ARCH__ >= 200) */
}
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError();                           \
    if(e!=cudaSuccess) {                       \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));                           \
        exit(EXIT_FAILURE);                   \
    }                       \
}

void checkP2Paccess(int numGPUs)
{
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaCheckError();

        for (int j = 0; j < numGPUs; j++) {
            int access;
            if (i != j) {
                cudaDeviceCanAccessPeer(&access, i, j);
                cudaCheckError();
                printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" : "CANNOT", j);
            }
        }
    }
    printf("\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n");
}


void nv_init_wrapper(int* c, char *v[], int* omp_mpi_level)
{
    int *target;
    int rank, nranks, ndevices;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;
    //MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Init_thread( c, &v, MPI_THREAD_MULTIPLE, omp_mpi_level));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));


    //CUDA_CHECK(cudaSetDevice(rank%ndevices));
    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    NVSHMEM_CHECK(nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_CHECK(MPI_Get_processor_name(name, &resultlength));
    int get_cur_dev;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    //CUDA_CHECK(cudaSetDevice(rank%ndevices));
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaGetDevice(&get_cur_dev));

    cudaDeviceProp prop;
    //CUDA_CHECK(cudaGetDeviceProperties(&prop, rank%ndevices));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node));
    printf("** MPI %d/%d, NVSHMEM %d/%d, mype_node=%d, device name: %s bus id: %d, "
           "ndevices=%d,cur=%d, node=%s **\n",
           rank,nranks,mype,npes,mype_node, prop.name, prop.pciBusID,
           ndevices,get_cur_dev,name);
    fflush(stdout);

}


__device__ void C_BcTree_forwardMessageSimple_Device(C_Tree* tree,  int* flag_bc_q,  int* my_flag_bc, int mype, int tid,double* ready_x){
    //int BCsendoffset;
    int sig = 1;
    for( int idxRecv = 0; idxRecv < tree->destCnt_; ++idxRecv ) {
        int iProc = tree->myDests_[idxRecv];
        //BCsendoffset = my_flag_bc[2];
        //double sum=0;
        //if (tid==0) {
        //    for(int i=0;i<my_flag_bc[3];i++){
        //        //printf("(%d), data, %d,%lf\n",mype,i,ready_x[i]);
        //        sum+=ready_x[my_flag_bc[2]+i];
        //    }
        //    printf("Start (%d), forwardDevice, send to %d, signal offset=%d, msgsz=%d,sum=%lf\n",mype,iProc,my_flag_bc[0],my_flag_bc[3],sum);
        //}
        //__syncthreads();
        //if(tid==0) printf("Start (%d), forwardDevice, send to %d, signal offset=%d, data offset=%d, msgsz=%d\n",mype,iProc,my_flag_bc[1],BCsendoffset,my_flag_bc[3]);
        //__syncthreads();
        //nvshmemx_double_put_block(&ready_x[BCsendoffset],ready_x,my_flag_bc[3],iProc);
        nvshmemx_double_put_nbi_block(&ready_x[my_flag_bc[2]], &ready_x[my_flag_bc[2]], my_flag_bc[3], iProc);
        //nvshmem_double_put_nbi(ready_x, &ready_x[0], my_flag_bc[3], iProc);
        //nvshmem_double_put(&ready_x[BCsendoffset],ready_x,my_flag_bc[3],iProc);
        //nvshmem_quiet();
        nvshmem_fence();
        //__syncthreads();
        if (tid == 0) {
            nvshmemx_int_signal((int*)(flag_bc_q + my_flag_bc[0]), sig, iProc);
            //nvshmem_quiet();
            //printf("Done (%d), forwardDevice, send to %d, signal offset=%d, data offset=%d, msgsz=%d\n", mype, iProc,
            //       my_flag_bc[0], my_flag_bc[2], my_flag_bc[3]);

        }
    }
}

__device__ void C_RdTree_forwardMessageSimple_Device(C_Tree* Tree, int* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, double* ready_lsum, int maxrecvsz){
    int data_ofset,sig_ofset;
    if(Tree->myIdx %2 ==0){
        sig_ofset = my_flag_rd[0]*2;
        data_ofset = my_flag_rd[0]*maxrecvsz*2;
    }else{
        sig_ofset = my_flag_rd[0]*2+1;
        data_ofset = my_flag_rd[0]*maxrecvsz*2+maxrecvsz;
    }
    if(Tree->myRank_!=Tree->myRoot_){
        ////forward to my root if I have received everything
        int iProc = Tree->myRoot_;
        //// YL: Use NVSHMEM to send to my parent
        //printf("forwardMessage-(%d,%d,%d), rd--1, send to %d,data_ofset=%d,maxrecvsz=%d,k=%d,sigoffset=%d\n",mype,bid, tid,iProc,data_ofset,maxrecvsz,my_flag_rd[0],sig_ofset);
        //__device__ void nvshmem_TYPENAME_put(TYPE *dest, const TYPE *source, size_t nelems, int pe)
        //double sum=0;
        //for(int i=my_flag_rd[0]*maxrecvsz*2;i<my_flag_rd[0]*maxrecvsz*2+my_flag_rd[1];i++){
        //    //printf("(%d), data, %d\n",mype,i);
        //    //printf("(%d), data, %d,%lf\n",mype,i,ready_lsum[i]);
        //    sum+=ready_lsum[i];
        //}
        //printf("forwardMessage- (%d), forwardDevice, send to %d, lib=%d,sum=%lf\n",mype,iProc,my_flag_rd[0],sum);

        nvshmem_double_put_nbi(&ready_lsum[data_ofset],&ready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],iProc);
        nvshmem_fence();
        int sig=1;
        nvshmemx_int_signal((int*)flag_rd_q+sig_ofset, sig, iProc);
        //printf("forwardMessage-(%d,%d,%d), rd--2, send to %d,data_ofset=%d,sig_ofset=%d\n",mype,bid, tid,iProc,data_ofset,sig_ofset);
    }
}


__global__ void ping_pong(volatile double *data_d, volatile int *flag_d, volatile int *flag_d_local,
                          int len, int pe, int iter, int skip, int peer) {
    long long int start, stop;
    double usec, time,bw;
    int i, tid,bid;

    tid = threadIdx.x;
    bid = blockIdx.x;

    //nvshmem_barrier_all();
    //if (tid == 0) printf("(%d) iter=%d, skip=%d,peer=%d\n", pe, iter, skip, peer);
    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();
        if (pe == peer) {
            if (tid==0) printf("(%d,%d) mypeer=%d\n",pe,bid,peer);
            if (tid == 0) nvshmem_int_wait_until((int *) flag_d + bid, NVSHMEM_CMP_EQ, bid + (i + 1));
            if (tid==0) printf("(%d,%d) mypeer=%d,msg arrived\n",pe,bid,peer);

            if (tid==0) nvshmem_double_put_nbi((double *)data_d+bid, (double *)data_d+bid, len, peer);
            //nvshmemx_double_put_nbi_block((double *) data_d + bid, (double *) data_d + bid, len, 0);
            //__syncthreads();
            if (tid==0) printf("(%d,%d) mypeer=%d,send data done\n",pe,bid,peer);

            if (tid==0) nvshmem_fence();
            if (tid == 0) nvshmemx_int_signal((int *) flag_d + bid, bid + (i + 1), 0);
            if (tid==0) printf("(%d,%d) mypeer=%d, signal done\n",pe,bid,peer);
        } else if (pe == 0) {
            //nvshmem_double_put_nbi((double *) data_d + bid, (double *) data_d + bid, len, peer);
            if (tid==0) nvshmemx_double_put_nbi_block((double *) data_d + bid, (double *) data_d + bid, len, peer);
            if (tid==0) printf("(%d,%d) mypeer=%d,send data done\n",pe,bid,peer);

            if (tid==0) nvshmem_fence();
            if (tid == 0) {
                nvshmemx_int_signal((int *) flag_d + bid, bid + (i + 1), peer);
                printf("(%d,%d) mypeer=%d, signal done\n",pe,bid,peer);
                do{
                    printf("(%d,%d) checking flag=%d\n",pe,bid,flag_d[bid]);
                }while(flag_d[bid]!=1);
                nvshmem_int_wait_until((int *) flag_d + bid, NVSHMEM_CMP_EQ, bid + (i + 1));
                if (tid==0) printf("(%d,%d) mypeer=%d,msg arrived\n",pe,bid,peer);
            }
        }
    }
    //nvshmem_barrier_all();
    stop = clock64();
    nvshmem_quiet();

    if ((pe == 0) && !tid) {
        time = (stop - start) / iter;
        usec = time * 1000 / clockrate;
        bw = ((float) iter * (float) len * sizeof(double) * 2 * clockrate) / ((time / 1000) * 1024 * 1024 * 1024);
        printf("Block (%d) %7lu, %8.2f, %4.6f \n", bid, len * sizeof(int), usec, bw);
    }
}

__global__ void schedule
(
int nrhs,
C_Tree  *LRtree_ptr,
int_t maxrecvsz,
int mype,
int* flag_bc_q,
int* flag_rd_q,
double* ready_x,
double* ready_lsum,
int* my_flag_bc,
int* my_flag_rd,
int* d_nfrecv,
int* d_status,
int* d_launch_flag,
int* d_colnum,
int* d_mynum,
int* d_mymaskstart,
int* d_mymasklength,
int* d_nfrecvmod,
int* d_statusmod,
int* d_colnummod,
int* d_mynummod,
int* d_mymaskstartmod,
int* d_mymasklengthmod,
int* d_recv_cnt,
int* d_msgnum,
double *lsum,    /* Sum of local modifications.                        */
int_t *fmod,     /* Modification count for L-solve.                    */
gridinfo_t *grid,
int_t *xsup,
int_t *ilsum,
int nbrow_loc,
int_t  nsupers
) {

    int bid = blockIdx.x;
    //int global_id= blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    d_launch_flag[0] = 1;
    int WAIT_NUM_THREADS = d_nfrecv[1]*d_nfrecv[2];
    //if (tid==0) printf("(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecv[0]);

   if (bid == 0) { // for BC
        if (WAIT_NUM_THREADS >= d_nfrecv[0]) {
            if (tid < d_nfrecv[0]) {
                //printf("WAIT1 (%d,%d) wait for col %d,flag=%d\n", mype, tid, d_colnum[tid],flag_bc_q[d_colnum[tid]]);
                nvshmem_int_wait_until((int *) flag_bc_q + d_colnum[tid], NVSHMEM_CMP_EQ, 1);
                d_status[d_colnum[tid]] = 1;
                //printf("WAIT1 (%d,%d) msg arrived in col %d\n", mype, tid, d_colnum[tid]);
            }
        } else {
            int delta = d_nfrecv[0] % WAIT_NUM_THREADS;
            if (tid < delta) {
                d_mynum[tid] = d_nfrecv[0] / WAIT_NUM_THREADS + 1;
            } else {
                d_mynum[tid] = d_nfrecv[0] / WAIT_NUM_THREADS;
            }
            __syncthreads();
            d_mymaskstart[tid] = 0;
            for (int i = 0; i < tid; i++) {
                d_mymaskstart[tid] += d_mynum[i];
            }
            d_mymasklength[tid] = d_colnum[d_mymaskstart[tid] + d_mynum[tid] - 1] - d_colnum[d_mymaskstart[tid]] + 1;
            __syncthreads();
            //printf("WAIT2 (%d,%d) mynum=%d, start=%d,%d length=%d\n",mype,tid,d_mynum[tid],d_mymaskstart[tid],d_colnum[d_mymaskstart[tid]],d_mymasklength[tid]);

            for (int i = 0; i < d_mynum[tid]; i++) {
                int wm_val = nvshmem_int_wait_until_any(flag_bc_q + d_colnum[d_mymaskstart[tid]], d_mymasklength[tid],
                                                        d_status + d_colnum[d_mymaskstart[tid]], NVSHMEM_CMP_EQ, 1);
                d_status[d_colnum[d_mymaskstart[tid]] + wm_val] = 1;
                //printf("WAIT2 (%d,%d) msg arrived in col %d, i=%d\n",mype,tid,d_colnum[d_mymaskstart[tid]] + wm_val, i);
            }
        }
   }
#if 0
    if (bid == 1) { // for RD
        //if (tid==0) printf("RD---(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecvmod[1]);
        int j, iam, lib, mycol, myrow, k, knsupc, il, cnt;
        int_t fmod_tmp, aln_i;
        aln_i = 1;
        double temp;
        if (WAIT_NUM_THREADS >= d_nfrecvmod[1]) { // one thread wait for one col
            if (tid < d_nfrecvmod[1]) {
                //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]]);
                for (int i = 0; i < d_recv_cnt[d_colnummod[tid]]; i++) {
                    //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d,i=%d,wait_off=%d,%d,status=%d,%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]],i,d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
                    int wm_val = nvshmem_int_wait_until_any(flag_rd_q + d_colnummod[tid] * 2, 2,
                                                            d_statusmod + d_colnummod[tid] * 2, NVSHMEM_CMP_EQ, 1);
                    d_statusmod[d_colnummod[tid] * 2 + wm_val] = 1;

                    lib = (d_colnummod[tid] * 2 + wm_val) / 2;

                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);

                    k = myrow + lib * grid->nprow; // global block row
                    knsupc = SuperSize(k);
                    il = LSUM_BLK(lib);
                    cnt = LRtree_ptr[lib].destCnt_;
                    //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d\n", mype, bid, tid,
                    //       d_colnummod[tid] * 2 + wm_val, lib, cnt);
                    if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                        double tmp_sum = 0;
                        int ii = 0;
                        if (cnt == 2) {
                            for (ii = 0; ii < cnt; ++ii) {
                                tmp_sum = 0;
                                RHS_ITERATE(j) {
                                    for (int aab = 0; aab < knsupc; ++aab) {
                                        //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                        temp = atomicAdd(&lsum[il + aab + j * knsupc],
                                                         ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                    j * knsupc]);
                                        tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                        //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid,
                                        //       lib, k, ii,
                                        //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                        //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                    }

                                    // atomic return old val
                                    fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                                    //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,
                                    //       tmp_sum,fmod_tmp);
                                }
                            }
                        }
                        if (cnt == 1) {
                            if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; ++aab) {
                                    //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                    temp = atomicAdd(&lsum[il + aab + j * knsupc],
                                                     ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                    tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                    //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
                                    //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                    //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                }

                            }
                            // atomic return old val
                            fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                            //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                        }

                        if (fmod_tmp == 1) {// forward RD
                            //senddone[lk]=1;
                            if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                                //cnt=LRtree_ptr[lib].msgSize_;
                                my_flag_rd[k * RDMA_FLAG_SIZE] = lib;
                                my_flag_rd[k * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                                RHS_ITERATE(j) {
                                    for (int aab = 0; aab < knsupc; aab++) {
                                        ready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                        //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                        //       k * maxrecvsz * 2 + i +j * knsupc,
                                        //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                    }
                                }
                                //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                                //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                                C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int *) flag_rd_q,
                                                                     &my_flag_rd[RDMA_FLAG_SIZE * k], mype, bid, tid,
                                                                     &ready_lsum[0], maxrecvsz);
                            }
                        }
                    }
                }//for
            }
        } else {
            int delta = d_nfrecvmod[1] % WAIT_NUM_THREADS;

            if (tid < delta) {
                d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS+1;
            } else {
                d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS;
            }
            __syncthreads();

            d_mymaskstartmod[tid] = 0;
            d_mymasklengthmod[tid] = 0;
            d_msgnum[tid] = 0;

            //d_mymaskstartmod: start offset of d_colnummod
            for (int i = 0; i < tid; i++) {
                d_mymaskstartmod[tid] += d_mynummod[i];
                //printf("(%d,%d,%d),i=%d,d_mynummod=%d,d_mymaskstartmod=%d\n",
                //       mype,bid,tid,i,
                //       d_mynummod[i],d_mymaskstartmod[tid]);
            }
            __syncthreads();

            for (int i = d_mymaskstartmod[tid]; i < d_mymaskstartmod[tid] + d_mynummod[tid]; i++) {
                d_msgnum[tid] += d_recv_cnt[d_colnummod[i]];
                //printf("(%d,%d,%d),i=%d,d_recv_cnt=%d\n",mype,bid,tid,i,d_recv_cnt[d_colnummod[i]]);
            }
            d_mymasklengthmod[tid] = (d_colnummod[d_mymaskstartmod[tid] + d_mynummod[tid] - 1]
                                      - d_colnummod[d_mymaskstartmod[tid]]+1)*2;
            //printf("(%d,%d,%d) waitcol=%d,msgnum=%d,masklength=%d,start=%d\n",mype,bid,tid,
            //                   d_mynummod[tid],d_msgnum[tid],
            //                   d_mymasklengthmod[tid],d_mymaskstartmod[tid]);

            for (int i = 0; i < d_msgnum[tid]; i++) {
                //printf("(%d,%d,%d)--before wait any,i=%d/%d\n",mype,bid,tid,i,d_msgnum[tid]);
                int wm_val = nvshmem_int_wait_until_any(&flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                        d_mymasklengthmod[tid],
                                                        &d_statusmod[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                        NVSHMEM_CMP_EQ, 1);
                d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val] = 1;
                lib = (d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val) / 2;
                //printf("(%d,%d,%d)--recv a msg, offset=%d,base=%d, lib=%d, flag=%d,status=%d\n",mype,bid,tid,wm_val,d_colnummod[d_mymaskstartmod[tid]]*2,
                //                  lib,flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2+wm_val],d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val]);
                iam = grid->iam;
                mycol = MYCOL(iam, grid);
                myrow = MYROW(iam, grid);

                k = myrow + lib * grid->nprow; // global block row
                knsupc = SuperSize(k);
                il = LSUM_BLK(lib);
                cnt = LRtree_ptr[lib].destCnt_;
                //printf("HERE2-(%d,%d,%d),lib=%d,k=%d,wm_val=%d,cnt=%d,%d, mycnt=%d\n", mype, bid, tid, lib, k,
                //       wm_val,cnt,d_recv_cnt[lib],d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1]);

                if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                    double tmp_sum = 0;
                    int ii = 0;
                    if (cnt == 2) {
                        for (ii = 0; ii < cnt; ++ii) {
                            tmp_sum = 0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; aab++) {
                                    //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                    temp = atomicAdd(&lsum[il + aab + j * knsupc],
                                                     ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                j * knsupc]);
                                    tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                    //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid,
                                    //       lib, k, ii,
                                    //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                    //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                }

                                // atomic return old val
                                fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                                //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum,fmod_tmp);
                            }
                        }
                    }
                    if (cnt == 1) {
                        if (flag_rd_q[k * 2 + 1] == 1) ii = 1;
                        RHS_ITERATE(j) {
                            for (int aab = 0; aab < knsupc; ++aab) {
                                temp = atomicAdd(&lsum[il + aab + j * knsupc],
                                                 ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
                                //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                            }

                        }
                        // atomic return old val
                        fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                        //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                    }

                    if (fmod_tmp == 1) {// forward RD
                        //printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,LRtree_ptr[lib].myRoot_);
                        if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                            //cnt=LRtree_ptr[lib].msgSize_;
                            my_flag_rd[k * RDMA_FLAG_SIZE] = lib;
                            my_flag_rd[k * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; aab++) {
                                    ready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                    //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                    //       k * maxrecvsz * 2 + i +j * knsupc,
                                    //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                }
                            }
                            //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                            //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                            C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int *) flag_rd_q,
                                                                 &my_flag_rd[RDMA_FLAG_SIZE * k], mype, bid, tid,
                                                                 &ready_lsum[0], maxrecvsz);
                        }
                    }
                }
            }//for
        } // else WAIT_NUM_THREAD<recv
    }
#endif
}



// /************************************************************************/
// /*! \brief
// *
// * <pre>
// * Purpose
// * =======
// *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
// * </pre>
// */
// __global__ void dlsum_fmod_inv_gpu_1rhs
// /************************************************************************/
// (
// double *lsum,    /* Sum of local modifications.                        */
// double *x,       /* X array (local)                                    */
// double *rtemp,   /* Result of full matrix-vector multiply.             */
// int   nrhs,      /* Number of right-hand sides.                        */
// int   maxsup,      /* Max supernode size.                        */
// int_t   nsupers,      /* Number of total supernodes.                        */
// int_t *fmod,     /* Modification count for L-solve.                    */
// int_t *xsup,
// gridinfo_t *grid,
// LocalLU_t *Llu
// )
// {
// double alpha = 1.0, beta = 0.0,malpha=-1.0;
// double *lusup, *lusup1;
// double *dest;
// double *Linv;/* Inverse of diagonal block */
// int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r,m;
// int_t  k,i, l,ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready;
// int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
// int_t  luptr_tmp,luptr_tmp1,lptr1_tmp,maxrecvsz, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
// int thread_id1;
// flops_t ops_loc=0.0;
// MPI_Status status;
// int test_flag;
// yes_no_t done;
// C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
// C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
// int_t* idx_lsum,idx_lsum1;
// const int Nbk=1;
// // __shared__ double rtemp_loc[128];
// double temp,temp1;
// int_t ldalsum;
// int_t nleaf_send_tmp;
// int_t lptr;      /* Starting position in lsub[*].                      */
// int_t luptr;     /* Starting position in lusup[*].                     */
// int_t iword = sizeof(int_t);
// int_t dword = sizeof (double);
// int_t aln_d,aln_i;
// aln_d = 1;//ceil(CACHELINE/(double)dword);
// aln_i = 1;//ceil(CACHELINE/(double)iword);
// int   knsupc;    /* Size of supernode k.                               */
// int_t nlb;       /* Number of L blocks.                                */
// int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */

// int_t bid;
// int_t tmp;
// int_t tid = threadIdx_x + threadIdx_y * blockDim_x;
// int_t ready = 0;
// // int_t lock = 0;
// const int block_size = blockDim_x*blockDim_y; /* number of threads per block*/
// double zero = 0.0;


// double rC[THR_N][THR_M];

// gpuError_t error;

// bid= blockIdx_x;
// int_t idx = threadIdx_x;  // thread's m dimension
// int_t idy = threadIdx_y;  // thread's n dimension
// int_t ni,mi;


// int_t wrp;
// int_t lne = threadIdx_x & 0x1f ;
// // int_t ready = 0;
// // int_t lock = 0;
// const int warp_size = 32; /* number of threads per warp*/
// wrp= threadIdx_x + blockIdx_x * blockDim_x;
// wrp/=warp_size;



// // printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


// // rtemp_loc = (double*)malloc(maxsup*nrhs*Nbk*sizeof(double));

// if(wrp>=CEILING(nsupers, grid->npcol)){
// return;
// }else if(!Llu->Lrowind_bc_ptr[wrp]){
// return;
// }



// lk=wrp;
// iam = grid->iam;
// mycol = MYCOL( iam, grid );
// myrow = MYROW( iam, grid );
// k = mycol+lk*grid->npcol;
// knsupc = SuperSize( k );
// lsub = Llu->Lrowind_bc_ptr[lk];
// iam = grid->iam;
// krow = PROW( k, grid );
// lusup = Llu->Lnzval_bc_ptr[lk];
// lloc = Llu->Lindval_loc_bc_ptr[lk];
// nsupr = lsub[1];

// if(myrow==krow){
// nlb = lsub[0] - 1;
// idx_n = 1;
// idx_i = nlb+2;
// idx_v = 2*nlb+3;
// luptr_tmp = lloc[idx_v];
// m = nsupr-knsupc;
// }else{
// nlb = lsub[0];
// idx_n = 0;
// idx_i = nlb;
// idx_v = 2*nlb;
// luptr_tmp = lloc[idx_v];
// m = nsupr;
// }

// // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

// if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

// if(lne==0){  /*only the first thread in a warp handles the lock */

// // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
// // for (i=0 ; i<maxsup ; i++){
// // rtemp_loc[i]=0.0;
// // }

// lib = LBi( k, grid ); /* Local block number, row-wise. */
// do{
// tmp=fmod[lib*aln_i];
// __threadfence();
// }while(tmp>0);

// }
// __syncwarp();


// lib = LBi( k, grid ); /* Local block number, row-wise. */
// il = LSUM_BLK( lib );
// ii = X_BLK( lib );

// for (i = lne; i < knsupc; i+=warp_size)
// x[i + ii ] += lsum[i + il ];
// // __syncwarp();


// if(Llu->inv == 1){
// Linv = Llu->Linv_bc_ptr[lk];
// for (i = lne; i < knsupc; i+=warp_size){
// temp1=zero;
// for (l=0 ; l<knsupc ; l++){
// temp1+=  Linv[l*knsupc+i]*x[ii+l];
// }
// lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
// }
// // __syncwarp();

// for (i = lne; i < knsupc; i+=warp_size){
// x[i + ii] = lsum[il+i];
// // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
// }
// // __syncwarp();

// }
// // __syncwarp();
// }else{   /* off-diagonal block forward the message*/
// /* waiting for the x subvector and forward*/
// }


// if(nlb>0){
// if(nrhs==1){
// lib = LBi( k, grid ); /* Local block number, row-wise. */
// ii = X_BLK( lib );

// luptr_tmp1 = lloc[idx_v];
// lb = 0;
// nbrow=0;
// lptr1_tmp = lloc[lb+idx_i];
// lptr= lptr1_tmp+2;
// nbrow1 = lsub[lptr1_tmp+1];
// ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
// rel = xsup[ik]; /* Global row index of block ik. */
// lk = LBi( ik, grid ); /* Local block number, row-wise. */
// iknsupc = SuperSize( ik );
// il = LSUM_BLK( lk );
// for (i = lne; i < m; i+=warp_size){
// while(nbrow+lsub[lptr1_tmp+1]<=i){
// lb++;
// nbrow +=lsub[lptr1_tmp+1];
// lptr1_tmp = lloc[lb+idx_i];
// lptr= lptr1_tmp+2;
// ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
// rel = xsup[ik]; /* Global row index of block ik. */
// lk = LBi( ik, grid ); /* Local block number, row-wise. */
// iknsupc = SuperSize( ik );
// il = LSUM_BLK( lk );
// }

// irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
// RHS_ITERATE(j){
// temp1=zero;
// for (l=0 ; l<knsupc ; l++){
// temp1+= lusup[luptr_tmp1+l*nsupr+i]*x[ii+j*knsupc+l];
// }

// temp=atomicAdd(&lsum[il+irow + j*iknsupc],-temp1);
// }
// if(i==nbrow+lsub[lptr1_tmp+1]-1){
// fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
// // __threadfence();
// }
// }
// }
// // __syncwarp();

// // if(tid==0){
// // for (lb = tid; lb < nlb; lb+=warp_size){
// // lptr1_tmp = lloc[lb+idx_i];
// // ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
// // lk = LBi( ik, grid ); /* Local block number, row-wise. */
// // fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
// // // printf("k: %5d r: %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i]);
// // }
// // }
// // __syncwarp();
// // } /*if tid<Nchunk*/
// } /* if nlb>0*/


// } /* dlsum_fmod_inv_gpu_1rhs */

__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        //val += __shfl_down_sync(0xffffffff,val, offset,warpSize);
        val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
    //__shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    return val;
}

__inline__ __device__
int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff,val, mask,warpSize);
    return val;
}

__inline__ __device__
int blockReduceSum(int val, int bid, int tid, int mype) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    double sz=32.0;
    int lane = tid % warpSize;
    int wid = tid>>(int)log2(sz);
    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory
    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (tid < (blockDim.x * blockDim.y) / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}


__inline__ __device__ int warpReduceMin(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        int tmpVal = __shfl_down_sync(0xffffffff,val, offset, warpSize);
        if (tmpVal < val)  val = tmpVal;
    }
    return val;
}

__inline__ __device__  int blockReduceMin(int val,int bid, int tid, int mype)
{

    static __shared__ int shared[32]; // Shared mem for 32 partial mins
    double sz=32.0;
    int lane = tid % warpSize;
    int wid = tid>>(int)log2(sz);

    warpReduceMin(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (tid < (blockDim.x * blockDim.y) / warpSize) ? shared[lane] : INT_MAX;

    if (wid == 0)  warpReduceMin(val); //Final reduce within first warp
    return val;
}

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
__global__ void dlsum_fmod_inv_gpu_mrhs_nvshmem
/************************************************************************/
        (
                int_t nbcol_loc,
                double *lsum,    /* Sum of local modifications.                        */
                double *x,       /* X array (local)                                    */
                int   nrhs,      /* Number of right-hand sides.                        */
                int   maxsup,      /* Max supernode size.                        */
                int_t   nsupers,      /* Number of total supernodes.                        */
                int_t *fmod,     /* Modification count for L-solve.                    */
                C_Tree  *LBtree_ptr,
                C_Tree  *LRtree_ptr,
                int_t *ilsum,
                int_t *Lrowind_bc_dat,
                long int *Lrowind_bc_offset,
                double *Lnzval_bc_dat,
                long int *Lnzval_bc_offset,
                double *Linv_bc_dat,
                long int *Linv_bc_offset,
                int_t *Lindval_loc_bc_dat,
                long int *Lindval_loc_bc_offset,
                int_t *xsup,
                gridinfo_t *grid,
                int_t maxrecvsz,
                int mype,
                volatile int* flag_bc_q,
                volatile int* flag_rd_q,
                double* ready_x,
                double* ready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_launch_flag,
                int* d_nfrecv,
                volatile int* d_status,
                volatile int* d_statusmod,
                int_t nblock_ex
        )
{
    double alpha = 1.0, beta = 0.0,malpha=-1.0;
    double *lusup, *lusup1;
    double *dest;
    double *Linv;/* Inverse of diagonal block */
    int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r,m;
    int_t  k,i, l,ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready;
    int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
    int_t  luptr_tmp,luptr_tmp1,lptr1_tmp, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
    int thread_id1;
    flops_t ops_loc=0.0;
    MPI_Status status;
    int test_flag;
    yes_no_t done;
    int_t* idx_lsum,idx_lsum1;
    const int Nbk=1;
    __shared__ double rtemp_loc[128];
    double temp,temp1;
    int_t ldalsum;
    int_t nleaf_send_tmp;
    int_t lptr;      /* Starting position in lsub[*].                      */
    int_t luptr;     /* Starting position in lusup[*].                     */
    int_t iword = sizeof(int_t);
    int_t dword = sizeof (double);
    int_t aln_d,aln_i;
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int   knsupc;    /* Size of supernode k.                               */
    int_t nlb;       /* Number of L blocks.                                */

    int_t bid=blockIdx_x;
    int_t tmp;
    int_t tid = threadIdx_x + threadIdx_y * blockDim_x;
    int_t ready = 0;
    // int_t lock = 0;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
    double zero = 0.0;
    double rC[THR_N][THR_M];
    gpuError_t error;
    int_t idx = threadIdx_x;  // thread's m dimension
    int_t idy = threadIdx_y;  // thread's n dimension
    int_t ni,mi;
    int cnt;
    yes_no_t test;
    // rtemp_loc = (double*)malloc(maxsup*nrhs*Nbk*sizeof(double));
    //int delta=(nblock_ex < nbcol_loc ? nblock_ex : nbcol_loc);
    //int_t bid1;
    //bool flag=false;
    //if (bid<(delta*2)){
    //    bid1=bid/2;
    //    flag= bid%2==0 ? 1:0;
    //}else{
    //    bid1=bid-delta;
    //    flag= delta==nblock_ex ? 1:0;
    //}
    //if(tid==0) printf("(%d) iam bid=%d,bid1=%d,flag=%d\n",mype,bid,bid1,flag);
    //if(tid==0) printf("(%d) iam bid=%d,enter solve 0\n",mype,bid);

    //if(bid<nbcol_loc){
    if (Lrowind_bc_offset[bid] == -1) {
       return;
    }
    //if(tid==0) printf("(%d) iam bid=%d,enter solve--1\n",mype,bid);
    int get_offset, get_msgsize, get_rank, gc, gr, tmp_id, recv_offset = 0;

    lk = bid;
    iam = grid->iam;
    mycol = MYCOL(iam, grid);
    myrow = MYROW(iam, grid);
    gc = mycol + lk * grid->npcol;
    if (gc >= nsupers) return;
    k = gc; //mycol + lk * grid->npcol;

    knsupc = SuperSize(k);
    lsub = &Lrowind_bc_dat[Lrowind_bc_offset[lk]];
    iam = grid->iam;
    krow = PROW(k, grid);
    lusup = &Lnzval_bc_dat[Lnzval_bc_offset[lk]];
    lloc = &Lindval_loc_bc_dat[Lindval_loc_bc_offset[lk]];
    nsupr = lsub[1];

    if (myrow == krow) {
        nlb = lsub[0] - 1;
        idx_n = 1;
        idx_i = nlb + 2;
        idx_v = 2 * nlb + 3;
        luptr_tmp = lloc[idx_v];
        m = nsupr - knsupc;
    } else {
        nlb = lsub[0];
        idx_n = 0;
        idx_i = nlb;
        idx_v = 2 * nlb;
        luptr_tmp = lloc[idx_v];
        m = nsupr;
    }


    // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);
    if (myrow == krow) {   /* diagonal block performs trsm and forward the message*/

        if (tid == 0) {  /*only the first thread in a block handles the lock */
            //printf("(%d) iam bid=%d,enter solve--2, wait lock,gc=%d\n",mype,bid,gc);
            //printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            // for (i=0 ; i<maxsup ; i++){
            // rtemp_loc[i]=0.0;
            // }

            lib = LBi(k, grid); /* Local block number, row-wise. */
            do {
                tmp = fmod[lib * aln_i];
                __threadfence();
            } while (tmp > 0);
        }
        __syncthreads();
        //if(tid==0) printf("(%d) iam bid=%d,enter solve--2, unlock,gc=%d\n",mype,bid,gc);

        lib = LBi(k, grid); /* Local block number, row-wise. */
        il = LSUM_BLK(lib);
        ii = X_BLK(lib);

        RHS_ITERATE(j)for (i = tid; i < knsupc; i += block_size) {
                //atomicAdd(&ready_x[0],lsum[i + il + j * knsupc]);
                x[i + ii + j * knsupc] += lsum[i + il + j * knsupc];
            }
        __syncthreads();
        //if(tid==0) printf("(%d,%d,%d),CHECKING k=%d,gc=%d,checksum=%lf\n",mype,bid,tid,k,gc,ready_x[0]);

        //  if(Llu->inv == 1){

        Linv = &Linv_bc_dat[Linv_bc_offset[lk]];

        if (nrhs == 1) {

            for (i = tid; i < knsupc; i += block_size) {
                temp1 = zero;
                for (l = 0; l < knsupc; l++) {
                    temp1 += Linv[l * knsupc + i] * x[ii + l];
                }
                lsum[il + i] = temp1; //reuse lsum as temporary output as it's no longer accessed
            }
            __syncthreads();

            for (i = tid; i < knsupc; i += block_size) {
                x[i + ii] = lsum[il + i];
                // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
            }
            __syncthreads();

           // RHS_ITERATE(j){

           // for (i = tid; i < knsupc; i+=block_size)
           // rtemp_loc[i]=zero;
           // __syncthreads();


           // gemv_device_dlsum_fmod(
           // knsupc, knsupc, alpha,
           // Linv, knsupc,
           // &x[ii+j*knsupc], 1, beta,
           // rtemp_loc, 1);

           // __syncthreads();
           // // printf("tid %5d knsupc %5d block_size %5d\n",tid,knsupc,block_size);
           // for (i = tid; i < knsupc; i+=block_size){
           // x[i + ii + j*knsupc] = rtemp_loc[i];
           // // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
           // }
           // }
           // __syncthreads();

        } else {
           __syncthreads();
           for (int_t blx = 0; blx * BLK_M < knsupc; blx++) {
               for (int_t bly = 0; bly * BLK_N < nrhs; bly++) {
                   gemm_device_dlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
                                          Linv, knsupc, &x[ii], knsupc, rC,
                                          alpha, beta);
#pragma unroll
                   for (ni = 0; ni < THR_N; ni++) {
                       int_t coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                       for (mi = 0; mi < THR_M; mi++) {
                           int_t coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                           if (coord_dCm < knsupc && coord_dCn < nrhs) {
                               double &regC = rC[ni][mi];
                               lsum[coord_dCm + il + coord_dCn *
                                                     knsupc] = regC;  //reuse lsum as temporary output as it's no longer accessed
                           }//if (coord_dCm < knsupc && coord_dCn < nrhs)
                       }
                   }
               }
           }
           __syncthreads();

           RHS_ITERATE(j)for (i = tid; i < knsupc; i += block_size)
                   x[i + ii + j * knsupc] = lsum[i + il + j * knsupc];
           __syncthreads();
        }//if(nrhs==1)

        RHS_ITERATE(j)for (i = tid; i < knsupc; i += block_size)
                ready_x[i + maxrecvsz * lk + j * knsupc] = x[i + ii + j * knsupc];

        __syncthreads();
    } else {   /* off-diagonal block forward the message*/
            /* waiting for the x subvector and forward*/
            //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into ready_x[maxrecvsz*lk]
        volatile int msg_recv = 0;
        if (tid == 0) {
            //printf("in solve WAIT1 (%d,%d) wait for col %d,flag=%d\n", mype, bid, gc,flag_bc_q[gc]);
            //nvshmem_int_wait_until((int *) flag_bc_q + gc, NVSHMEM_CMP_EQ, 1);
            do {
                msg_recv = flag_bc_q[lk];
                //msg_recv=d_status[gc];
                //msg_recv=flag_bc_q[gc];
                __threadfence();
            } while (msg_recv != 1);
            //printf("(%d,%d,%d,%d) in compute kernel, I have msg=%d,sz=%d.ofset=%d\n",mype,bid,tid,gc,msg_recv,LBtree_ptr[lk].msgSize_*nrhs+XK_H,maxrecvsz*lk);
            //double sum=0;
            //for (int myi=0;myi<LBtree_ptr[lk].msgSize_*nrhs+XK_H;myi++){
            //    sum+=ready_x[maxrecvsz*lk+myi];
            //}
            //printf("(%d,%d,%d), gc=%d,lk=%d, sum=%lf\n",mype,bid,tid,gc,lk,sum);
        }
        __syncthreads();
        //for(int i=0;i<LBtree_ptr[lk].msgSize_*nrhs+XK_H;i++){
        //    ready_x[maxrecvsz*lk+i]=ready_x[maxrecvsz*gc+i];
        //}
        //__syncthreads();
    }
    __syncthreads();

    //YL: only the first thread in a block forwards the x subvector using NVSHMEM
    cnt = LBtree_ptr[lk].destCnt_;
    if (cnt > 0) {
        //cnt=LBtree_ptr[lk].msgSize_;
        my_flag_bc[gc * RDMA_FLAG_SIZE] = lk;
        my_flag_bc[gc * RDMA_FLAG_SIZE + 1] = gc;
        my_flag_bc[gc * RDMA_FLAG_SIZE + 2] = maxrecvsz * lk;
        //my_flag_bc[gc*RDMA_FLAG_SIZE+2]=maxrecvsz*lk;
        my_flag_bc[gc * RDMA_FLAG_SIZE + 3] = LBtree_ptr[lk].msgSize_ * nrhs + XK_H;
        C_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk], (int *) flag_bc_q, &my_flag_bc[gc * RDMA_FLAG_SIZE],
                                             mype, tid, &ready_x[0]);
        //printf("(%d,%d,%d), lk=%d, gc=%d\n",mype,bid,tid,lk,gc);
        //C_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&ready_x[maxrecvsz*lk],cnt*nrhs+XK_H);
    }
    int keep_lk = lk;
    __syncthreads();

    if (nlb > 0) {

        lib = LBi(k, grid); /* Local block number, row-wise. */
        ii = X_BLK(lib);

        if (nrhs == 1) {
            luptr_tmp1 = lloc[idx_v];
            lb = 0;
            nbrow = 0;
            lptr1_tmp = lloc[lb + idx_i];
            lptr = lptr1_tmp + 2;
            nbrow1 = lsub[lptr1_tmp + 1];
            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            rel = xsup[ik]; /* Global row index of block ik. */
            lk = LBi(ik, grid); /* Local block number, row-wise. */
            iknsupc = SuperSize(ik);
            il = LSUM_BLK(lk);

            for (i = tid; i < m; i += block_size) {
                while (nbrow + lsub[lptr1_tmp + 1] <= i) {
                    lb++;
                    nbrow += lsub[lptr1_tmp + 1];
                    lptr1_tmp = lloc[lb + idx_i];
                    lptr = lptr1_tmp + 2;
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi(ik, grid); /* Local block number, row-wise. */
                    iknsupc = SuperSize(ik);
                    il = LSUM_BLK(lk);
                }

                irow = lsub[lptr + i - nbrow] - rel; /* Relative row. */
                RHS_ITERATE(j) {
                    temp1 = zero;
                    for (l = 0; l < knsupc; l++) {
                        temp1 += lusup[luptr_tmp1 + l * nsupr + i] * ready_x[l + maxrecvsz * keep_lk + j * knsupc];
                    }

                    temp = atomicAdd(&lsum[il + irow + j * iknsupc], -temp1);
                    //printf("(%d,%d,%d),lsum[%d]=%f\n",mype,bid,tid,il+irow + j*iknsupc,lsum[il+irow + j*iknsupc]);
                }

                //  irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                //  if(i==nbrow+lsub[lptr1_tmp+1]-1){
                // 	 fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
                // 	 // __threadfence();
                //  }


            }
            __syncthreads();

            luptr_tmp1 = lloc[idx_v];
            lb = 0;
            nbrow = 0;
            lptr1_tmp = lloc[lb + idx_i];
            lptr = lptr1_tmp + 2;
            nbrow1 = lsub[lptr1_tmp + 1];
            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            rel = xsup[ik]; /* Global row index of block ik. */
            lk = LBi(ik, grid); /* Local block number, row-wise. */
            iknsupc = SuperSize(ik);
            il = LSUM_BLK(lk);
            gr=myrow + lk * grid->nprow;
            //knsupc = SuperSize(gr);

            for (i = tid; i < m; i += block_size) {
                while (nbrow + lsub[lptr1_tmp + 1] <= i) {
                    lb++;
                    nbrow += lsub[lptr1_tmp + 1];
                    lptr1_tmp = lloc[lb + idx_i];
                    lptr = lptr1_tmp + 2;
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi(ik, grid); /* Local block number, row-wise. */
                    iknsupc = SuperSize(ik);
                    il = LSUM_BLK(lk);
                }
                //if (ik==15) printf("(%d) iam bid=%d,enter solve--3,fmod=%d\n",mype,bid,fmod_tmp);

                irow = lsub[lptr + i - nbrow] - rel; /* Relative row. */
                if (i == nbrow + lsub[lptr1_tmp + 1] - 1) {
                    // atomic return old val, omp return new val
                    fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);
                    //printf("(%d) iam bid=%d,tid=%d,enter solve--6,i=%d,r=%d,lk=%d,ik=%d,fmod_tmp=%d,fmod=%d\n",mype,bid,tid,i,nbrow + lsub[lptr1_tmp + 1] - 1, lk,ik,fmod_tmp,fmod[lk * aln_i]);
                    // __threadfence();
                    if(fmod_tmp==1) {// forward RD
                        //senddone[lk]=1;
                        if(LRtree_ptr[lk].myRoot_ != LRtree_ptr[lk].myRank_){
                            //cnt=LRtree_ptr[lib].msgSize_;
                            my_flag_rd[ik*RDMA_FLAG_SIZE]=lk;
                            my_flag_rd[ik*RDMA_FLAG_SIZE+1]=LRtree_ptr[lk].msgSize_;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < iknsupc; aab++) {
                                    ready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc] = lsum[il + aab +j * iknsupc];
                                    //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                    //       k * maxrecvsz * 2 + i +j * knsupc,
                                    //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                }
                            }
                            //printf("(%d,%d,%d) in solve,lib=%d,gr=%d,ik=%d,myflagrd=%d,%d\n",mype,bid,tid,lk,gr,ik,my_flag_rd[ik*RDMA_FLAG_SIZE],my_flag_rd[ik*RDMA_FLAG_SIZE+1]);
                            C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lk], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*ik], mype, bid, tid, &ready_lsum[0],maxrecvsz);
                        }
                    }
                }
            }
            //__syncthreads();

        } else {
            for (lb = 0; lb < nlb; lb++) {
                luptr_tmp1 = lloc[lb + idx_v];

                // nbrow=0;
                // lptr1_tmp = lloc[lb+idx_i];
                // nbrow += lsub[lptr1_tmp+1];


                lib = LBi(k, grid); /* Local block number, row-wise. */
                ii = X_BLK(lib);

                lptr1_tmp = lloc[lb + idx_i];
                lptr = lptr1_tmp + 2;
                nbrow1 = lsub[lptr1_tmp + 1];
                ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                rel = xsup[ik]; /* Global row index of block ik. */

                lk = LBi(ik, grid); /* Local block number, row-wise. */

                iknsupc = SuperSize(ik);
                il = LSUM_BLK(lk);


                // if(nrhs==1){

                // for (i = tid; i < nbrow1; i+=block_size)
                // rtemp_loc[i]=zero;
                // __syncthreads();


                // gemv_device_dlsum_fmod(
                // nbrow1, knsupc, alpha,
                // &lusup[luptr_tmp1], nsupr,
                // &x[ii], 1, beta,
                // rtemp_loc, 1);

                // __syncthreads();
                // for (i = tid; i < nbrow1; i+=block_size){
                // irow = lsub[lptr+i] - rel; /* Relative row. */
                // temp=atomicAdd(&lsum[il+irow],-rtemp_loc[i]);
                // }
                // }else{

                for (int_t blx = 0; blx * BLK_M < nbrow1; blx++) {
                    for (int_t bly = 0; bly * BLK_N < nrhs; bly++) {
                        gemm_device_dlsum_fmod(nbrow1, nrhs, knsupc, blx, bly,
                                               &lusup[luptr_tmp1], nsupr, &ready_x[maxrecvsz * keep_lk], knsupc, rC,
                                               alpha, beta);
#pragma unroll
                            for (ni = 0; ni < THR_N; ni++) {
                                int_t coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                                for (mi = 0; mi < THR_M; mi++) {
                                    int_t coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                                    if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                        irow = lsub[lptr + coord_dCm] - rel; /* Relative row. */
                                        double &regC = rC[ni][mi];
                                        temp = atomicAdd(&lsum[il + irow + coord_dCn * iknsupc], -regC);
                                    }
                                }
                            }
                        }
                    }
                    // }//if(nrhs==1)

                    if (tid == 0)fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);


                }

            }//if(nrhs==1)


            // if(tid==0){
            // for (lb = tid; lb < nlb; lb+=block_size){
            // lptr1_tmp = lloc[lb+idx_i];
            // ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            // lk = LBi( ik, grid ); /* Local block number, row-wise. */
            // fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
            // // printf("k: %5d r: %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i]);
            // }
            // }
            //__syncthreads();
            // } /*if tid<Nchunk*/
        } /* if nlb>0*/
} /* dlsum_fmod_inv_gpu_mrhs */

__global__ void test(int nbcol_loc, int nblock_ex,int mype){
    int_t tid = threadIdx_x + threadIdx_y * blockDim_x;
    int_t bid= blockIdx_x;
    if(bid<nbcol_loc){
        if(tid==0) printf("(%d) iam in test if bid %d\n",mype,bid);

    }else if (bid<nbcol_loc + nblock_ex){
        if(tid==0) printf("(%d) iam bid %d\n",mype,bid);
    }
}

void dlsum_fmod_inv_gpu_wrap
        (
                int_t nbcol_loc,    /*number of local supernode columns*/
                int_t nbrow_loc,    /*number of local supernode rows*/
                int_t nthread_x,     /*kernel launch parameter*/
                int_t nthread_y,     /*kernel launch parameter*/
                double *lsum,    /* Sum of local modifications.                        */
                double *x,       /* X array (local)                                    */
                int   nrhs,      /* Number of right-hand sides.                        */
                int   maxsup,      /* Max supernode size.                        */
                int_t   nsupers,      /* Number of total supernodes.                        */
                int_t *fmod,     /* Modification count for L-solve.                    */
                C_Tree  *LBtree_ptr,
                C_Tree  *LRtree_ptr,
                int_t *ilsum,
                int_t *Lrowind_bc_dat,
                long int *Lrowind_bc_offset,
                double *Lnzval_bc_dat,
                long int *Lnzval_bc_offset,
                double *Linv_bc_dat,
                long int *Linv_bc_offset,
                int_t *Lindval_loc_bc_dat,
                long int *Lindval_loc_bc_offset,
                int_t *xsup,
                gridinfo_t *grid,
                int_t maxrecvsz,
                int* flag_bc_q,
                int* flag_rd_q,
                double* ready_x,
                double* ready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_launch_flag,
                int* d_nfrecv,
                int* h_nfrecv,
                int* d_status,
                int* d_colnum,
                int* d_mynum,
                int* d_mymaskstart,
                int* d_mymasklength,
                int* d_nfrecvmod,
                int* d_statusmod,
                int* d_colnummod,
                int* d_mynummod,
                int* d_mymaskstartmod,
                int* d_mymasklengthmod,
                int* d_recv_cnt,
                int* d_msgnum,
                int* senddone
                //int* d_rownum,
                //int* d_rowstart,
                //int* d_validrows
        ) {

    gpuStream_t sid = 0;
    int gid = 0;
    int mycol;
    int_t lk, k, knsupc;
    //int_t nblock_ex=CEILING( nbrow_loc, nthread_x*nthread_y);
    int_t nblock_ex = CEILING(nbrow_loc, ((nthread_x * nthread_y) / 32)); //32 (warp) * 8 =256
    //int_t nblock_ex = nbrow_loc; //CEILING(nbrow_loc, ((nthread_x * nthread_y) / 32)); //32 (warp) * 8 =256
    //int_t nblock_ex = CEILING(nbrow_loc, ((nthread_x * nthread_y) / 64)); //32 (warp) * 8 =256

    int mype, npes, ndevices;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    //printf("(%d) nbcol_loc %d\n", mype, nbcol_loc);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));


    //printf("(%d), Enter,mynode=%d\n",mype,mype_node);
    //fflush(stdout);


    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        //cudaStreamCreate(&stream[i]);
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }
    //int minGridSize;
    //int myblockSize;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize,&myblockSize,(const void *) schedule ,0,0 );
    //h_nfrecv[1]=myblockSize;
    //gpuMemcpy(d_nfrecv, h_nfrecv, 3 * sizeof(int), gpuMemcpyHostToDevice);
    //printf("(%d) solve=%d,%d, wait=%d,%d\n",mype,nbcol_loc,nthread_x*nthread_y,h_nfrecv[2],h_nfrecv[1]);
    //fflush(stdout);

    dim3 dimGrid_nv(h_nfrecv[2]); //2
    dim3 dimBlock_nv(h_nfrecv[1]); //1024
    dim3 dimGrid(nbcol_loc);
    dim3 dimBlock(nthread_x, nthread_y);

    int launch_success = 0;

    void *args[] = {&nrhs, &LRtree_ptr, &maxrecvsz, &mype, &flag_bc_q, &flag_rd_q,
                    &ready_x, &ready_lsum, &my_flag_bc, &my_flag_rd, &d_nfrecv, &d_status, &d_launch_flag,
                    &d_colnum, &d_mynum, &d_mymaskstart, &d_mymasklength,
                    &d_nfrecvmod, &d_statusmod, &d_colnummod, &d_mynummod, &d_mymaskstartmod, &d_mymasklengthmod,
                    &d_recv_cnt, &d_msgnum,&lsum,&fmod,&grid,&xsup,&ilsum,&nbrow_loc,&nsupers};

    NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void *) schedule, dimGrid_nv, dimBlock_nv, args, 0, stream[0]));
    do {
        cudaMemcpyAsync(&launch_success, d_launch_flag, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream[1]);
    } while (launch_success == 0);


    if (launch_success == 1)
    dlsum_fmod_inv_gpu_mrhs_nvshmem<<< dimGrid, dimBlock, 0, stream[1] >>>(nbcol_loc,
                                                                           lsum, x,
                                                                           nrhs, maxsup, nsupers,
                                                                           fmod,
                                                                           LBtree_ptr, LRtree_ptr,
                                                                           ilsum,
                                                                           Lrowind_bc_dat,
                                                                           Lrowind_bc_offset,
                                                                           Lnzval_bc_dat,
                                                                           Lnzval_bc_offset,
                                                                           Linv_bc_dat,
                                                                           Linv_bc_offset,
                                                                           Lindval_loc_bc_dat,
                                                                           Lindval_loc_bc_offset,
                                                                           xsup,
                                                                           grid, maxrecvsz,
                                                                           mype, flag_bc_q,
                                                                           flag_rd_q,
                                                                           ready_x, ready_lsum,
                                                                           my_flag_bc, my_flag_rd,
                                                                           d_launch_flag,
                                                                           d_nfrecv, d_status,
                                                                           d_statusmod,nblock_ex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}


 /************************************************************************/
 /*! \brief
  *
  * <pre>
  * Purpose
  * =======
  *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
  * </pre>
  */
  __global__ void dlsum_bmod_inv_gpu_mrhs
  /************************************************************************/
  (
   int_t nbcol_loc,
   double *lsum,    /* Sum of local modifications.                        */
   double *x,       /* X array (local)                                    */
   int   nrhs,      /* Number of right-hand sides.                        */
   int_t   nsupers,      /* Number of total supernodes.                        */
   int_t *bmod,     /* Modification count for U-solve.                    */
   C_Tree  *UBtree_ptr,
   C_Tree  *URtree_ptr,
   int_t *ilsum,
  int_t *Urbs,
  int_t   *Ufstnz_br_dat,
  long int *Ufstnz_br_offset,
  double *Unzval_br_dat,  
  long int *Unzval_br_offset,
  int_t  *Ucb_valdat,    
  long int *Ucb_valoffset,  
  Ucb_indptr_t *Ucb_inddat,
  long int *Ucb_indoffset,
  double *Uinv_bc_dat,     
  long int *Uinv_bc_offset,   
  int_t *xsup,
  gridinfo_t *grid
  )
  {
	  double alpha = 1.0, beta = 0.0,malpha=-1.0;
	  double xtemp;
	  double *lusup, *lusup1;
	  double *dest;
	  double *Uinv;/* Inverse of diagonal block */
	  int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r,m;
	  int_t  k,i, l,ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready, ub;
	  int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
	  int_t  luptr_tmp,luptr_tmp1,lptr1_tmp, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
	  int thread_id1;
	  int_t gik,ikfrow,iklrow;
	  int_t  uptr;
	  int_t fnz,fnzmin;
	  flops_t ops_loc=0.0;
	  MPI_Status status;
	  int test_flag;
	  yes_no_t done;
	  int_t* idx_lsum,idx_lsum1;
	  const int Nbk=1;
	//   __shared__ double rtemp_loc[128]; 
	  double temp,temp1;
	  double temp2[MAXSUPER];
	//   int_t temp3[128];
	//   int_t temp4[128];
	  int_t ldalsum;
	  int_t nleaf_send_tmp;
	  int_t lptr;      /* Starting position in lsub[*].                      */
	  int_t luptr;     /* Starting position in lusup[*].                     */
	  int_t iword = sizeof(int_t);
	  int_t dword = sizeof (double);
	  int_t aln_d,aln_i;
	  aln_d = 1;//ceil(CACHELINE/(double)dword);
	  aln_i = 1;//ceil(CACHELINE/(double)iword);
	  int   knsupc;    /* Size of supernode k.                               */
	  int_t nub;       /* Number of L blocks.                                */
 
	  int_t bid;
	  int_t tmp;
	  int_t tid = threadIdx_x + threadIdx_y * blockDim_x; 
	  int_t ready = 0;
	  // int_t lock = 0;
	  const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
	  double zero = 0.0;
	  	
  
	  double rC[THR_N][THR_M];
	  
	  gpuError_t error;
	  
	  bid= nbcol_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
	  int_t idx = threadIdx_x;  // thread's m dimension
	  int_t idy = threadIdx_y;  // thread's n dimension
	  int_t ni,mi;
	  int cnt;
	  yes_no_t test;

	  int_t  *usub;
	  double *uval, *y;

	  
	  
	  // printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);
	  
	  
	  // rtemp_loc = (double*)malloc(maxsup*nrhs*Nbk*sizeof(double));
	  
	  
	  // the first nbcol_loc handles all computations and broadcast communication
	  if(bid<nbcol_loc){
		  if(Uinv_bc_offset[bid]==-1){
		  return;
		  }
		  
		  lk=bid;
		  iam = grid->iam;
		  mycol = MYCOL( iam, grid );
		  myrow = MYROW( iam, grid );
		  k = mycol+lk*grid->npcol;
		  knsupc = SuperSize( k );
		  krow = PROW( k, grid );	
		  nub = Urbs[lk];      /* Number of U blocks in block column lk */
		  
		//   printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);
		  
		  if(myrow==krow){   /* diagonal block performs trsm and forward the message*/
  
			  if(tid==0){  /*only the first thread in a block handles the lock */
  
			  
			  // for (i=0 ; i<maxsup ; i++){
				  // rtemp_loc[i]=0.0;
			  // }	
			  
				  lib = LBi( k, grid ); /* Local block number, row-wise. */
				//   printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
				  do{
					  tmp=bmod[lib*aln_i];
					  __threadfence();			
				  }while(tmp>0);
				  
			  }
			  __syncthreads();
			//   if(tid==0)
			//   printf("spin: %d %d \n",threadIdx_x, blockIdx_x);

				  
				  lib = LBi( k, grid ); /* Local block number, row-wise. */
				  il = LSUM_BLK( lib );
				  ii = X_BLK( lib );
				  
				  RHS_ITERATE(j)
					  for (i = tid; i < knsupc; i+=block_size)
						  x[i + ii + j*knsupc] += lsum[i + il + j*knsupc ];
				  __syncthreads();
				  
				  
				 //  if(Llu->inv == 1){
				  
					  Uinv = &Uinv_bc_dat[Uinv_bc_offset[lk]];
						  
					  if(nrhs==1){
						  for (i = tid; i < knsupc; i+=block_size){					
							  temp1=zero;
							  for (l=0 ; l<knsupc ; l++){
								  temp1+=  Uinv[l*knsupc+i]*x[ii+l];
							  }								
							  lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
						  }
						  __syncthreads();					
							  
						  for (i = tid; i < knsupc; i+=block_size){
							  x[i + ii] = lsum[il+i];
							  // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
							  }					
						  __syncthreads();		
					  }else{
						  __syncthreads(); 	
						  for (int_t blx = 0; blx*BLK_M < knsupc; blx++){
							  for (int_t bly = 0; bly*BLK_N < nrhs; bly++){
								  gemm_device_dlsum_fmod(knsupc, nrhs, knsupc, blx, bly, 
								  Uinv, knsupc, &x[ii], knsupc, rC,
								  alpha, beta);
									  #pragma unroll
								  for (ni = 0; ni < THR_N; ni++) {
									  int_t coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
									  #pragma unroll
									  for (mi = 0; mi < THR_M; mi++) {
										  int_t coord_dCm = blx*BLK_M + mi*DIM_X + idx;
										  if (coord_dCm < knsupc && coord_dCn < nrhs) {
											  double &regC = rC[ni][mi];
											  lsum[coord_dCm + il + coord_dCn*knsupc ]=regC;  //reuse lsum as temporary output as it's no longer accessed
										  }//if (coord_dCm < knsupc && coord_dCn < nrhs)
									  }
								  }						
							  }
						  }
						  __syncthreads(); 	
  
						  RHS_ITERATE(j)
						  for (i = tid; i < knsupc; i+=block_size)
							  x[i + ii + j*knsupc] = lsum[i + il + j*knsupc ];
						  __syncthreads(); 		
					  }//if(nrhs==1)
				 //  }
				  
				//   RHS_ITERATE(j)
				//   for (i = tid; i < knsupc; i+=block_size)
				// 	  recvbuf_BC_gpu[i + maxrecvsz*lk + j*knsupc ] = x[i + ii + j*knsupc];
					  
			  __syncthreads();	
		  }else{   /* off-diagonal block forward the message*/
			  /* waiting for the x subvector and forward*/ 
			  if(tid==0){  //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into recvbuf_BC_gpu[maxrecvsz*lk]
			  
			  }
		  }
		   
		  
		//   if(tid==0){  //YL: only the first thread in a block forwards the x subvector using NVSHMEM
		//   cnt=LBtree_ptr[lk].destCnt_;
		//  //  printf("good1 %5d%5d\n",lk,cnt);
		//   if(cnt>0){
		// 	 cnt=LBtree_ptr[lk].msgSize_;
		// 	  C_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&recvbuf_BC_gpu[maxrecvsz*lk],cnt*nrhs+XK_H);
		//   }
		//   }	
		  
		  if(nub>0){
		  
				  lib = LBi( k, grid ); /* Local block number, row-wise. */
				  ii = X_BLK( lib );	


					int ngroup=SUPERLU_MIN(nub,block_size);
					int block_size_loc = floor((double)block_size/ngroup);
					int remainder = nub % ngroup;
					int gid=tid/block_size_loc;

					for (ub = gid; ub < nub; ub+=ngroup) {
						ik = Ucb_inddat[Ucb_indoffset[lk]+ub].lbnum; /* Local block number, row-wise. */
						usub = &Ufstnz_br_dat[Ufstnz_br_offset[ik]];
						uval = &Unzval_br_dat[Unzval_br_offset[ik]];
						i = Ucb_inddat[Ucb_indoffset[lk]+ub].indpos; /* Start of the block in usub[]. */
						i += UB_DESCRIPTOR;
						il = LSUM_BLK( ik );
						gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
						iknsupc = SuperSize( gik );
						ikfrow = FstBlockC( gik );
						iklrow = FstBlockC( gik+1 );
			
						// printf("ub %d bmod: %d \n",ub, bmod[ik*aln_i]);

						if(tid % block_size_loc==0){ // parallelizing this supernode across knsupc or irow doesn't seem to have any benefit
						fnzmin=100000000;
						for (jj = 0; jj < knsupc; ++jj)
							fnzmin = min(fnzmin,usub[i + jj]);
						RHS_ITERATE(j) {
							dest = &lsum[il + j*iknsupc];
							uptr = Ucb_valdat[Ucb_valoffset[lk]+ub]; /* Start of the block in uval[]. */
							
							for (jj = 0; jj < iknsupc; ++jj) 
								temp2[jj]=0;
							for (jj = 0; jj < knsupc; ++jj) {
								fnz = usub[i + jj];
								if ( fnz < iklrow ) { /* Nonzero segment. */
									/* AXPY */
									xtemp=x[ii+j*knsupc+jj];
									for (irow = fnz; irow < iklrow; ++irow){
										temp2[irow - ikfrow]+=uval[uptr++] * xtemp; // YL: this is most expensive operation on GPU
									}
								}
							} /* for jj ... */

							for (irow = fnzmin; irow < iklrow; ++irow){
								temp=atomicAdd(&dest[irow - ikfrow],-temp2[irow - ikfrow]);
							}								
						}
						fmod_tmp=atomicSub(&bmod[ik*aln_i],1);
						}
						
					} /* for ub ... */
				// }
				__syncthreads();

			
				//   __syncthreads();
			  // } /*if tid<Nchunk*/
		  } /* if nlb>0*/		
  
		  // printf("nimbgood \n");
  
//   }else if(bid<nbcol_loc+nblock_ex){  //the next nblock_ex blocks handle all reduction communication
	  
  }
  
		  
	  
  } /* dlsum_bmod_inv_gpu_mrhs */
  
  
 


 

 





 void dlsum_fmod_inv_gpu_wrap
 (
  int_t nbcol_loc,    /*number of local supernode columns*/
  int_t nbrow_loc,    /*number of local supernode rows*/
  int_t nthread_x,     /*kernel launch parameter*/
  int_t nthread_y,     /*kernel launch parameter*/
  double *lsum,    /* Sum of local modifications.                        */
  double *x,       /* X array (local)                                    */
  int   nrhs,      /* Number of right-hand sides.                        */
  int   maxsup,      /* Max supernode size.                        */
  int_t   nsupers,      /* Number of total supernodes.                        */
  int_t *fmod,     /* Modification count for L-solve.                    */
  C_Tree  *LBtree_ptr,
  C_Tree  *LRtree_ptr,
  int_t *ilsum,
  int_t *Lrowind_bc_dat,   
  long int *Lrowind_bc_offset,      
  double *Lnzval_bc_dat,     
  long int *Lnzval_bc_offset,     
  double *Linv_bc_dat,     
  long int *Linv_bc_offset,     
  int_t *Lindval_loc_bc_dat,     
  long int *Lindval_loc_bc_offset,     
  int_t *xsup,
  gridinfo_t *grid,
  double * recvbuf_BC_gpu,
  double * recvbuf_RD_gpu,
  int_t maxrecvsz
 ){
 
 gpuStream_t sid=0;
 int gid=0;
 int mycol;
 int_t lk,k,knsupc;
 int_t nblock_ex=CEILING( nbrow_loc, nthread_x*nthread_y);;
 
	 
	 // printf("pinv %d\n",Llu->inv);
	 // fflush(stdout);

 
	 // if(nrhs>1){
		 dim3 dimBlock(nthread_x, nthread_y);
		 dlsum_fmod_inv_gpu_mrhs<<< nbcol_loc+nblock_ex, dimBlock >>>(nbcol_loc,nblock_ex,lsum,x,nrhs,maxsup,nsupers,fmod,LBtree_ptr,LRtree_ptr,ilsum,Lrowind_bc_dat,Lrowind_bc_offset,Lnzval_bc_dat,Lnzval_bc_offset,Linv_bc_dat,Linv_bc_offset,Lindval_loc_bc_dat,Lindval_loc_bc_offset, xsup,grid,recvbuf_BC_gpu,recvbuf_RD_gpu,maxrecvsz);
	 // }else{
		 // dim3 dimBlock(nthread_x*nthread_y, 1);
		 // dlsum_fmod_inv_gpu_1rhs<<< CEILING(nbcol_loc,NWARP), dimBlock >>>(lsum,x,rtemp,nrhs,maxsup,nsupers,fmod,xsup,grid,Llu);	
	 // }


 
	 gpuDeviceSynchronize();
 }




 void dlsum_bmod_inv_gpu_wrap
 (
  int_t nbcol_loc,    /*number of local supernode columns*/
  int_t nbrow_loc,    /*number of local supernode rows*/
  int_t nthread_x,     /*kernel launch parameter*/
  int_t nthread_y,     /*kernel launch parameter*/
  double *lsum,    /* Sum of local modifications.                        */
  double *x,       /* X array (local)                                    */
  int   nrhs,      /* Number of right-hand sides.                        */
  int   maxsup,      /* Max supernode size.                        */
  int_t   nsupers,      /* Number of total supernodes.                        */
  int_t *bmod,     /* Modification count for L-solve.                    */
  C_Tree  *UBtree_ptr,
  C_Tree  *URtree_ptr,
  int_t *ilsum,
  int_t *Urbs,
  int_t   *Ufstnz_br_dat,
  long int *Ufstnz_br_offset,
  double *Unzval_br_dat,  
  long int *Unzval_br_offset,
  int_t  *Ucb_valdat,    
  long int *Ucb_valoffset,  
  Ucb_indptr_t *Ucb_inddat,
  long int *Ucb_indoffset,
  double *Uinv_bc_dat,     
  long int *Uinv_bc_offset,   
  int_t *xsup,
  gridinfo_t *grid
 ){
 
 gpuStream_t sid=0;
 int gid=0;
 int mycol;
 int_t lk,k,knsupc;

	 
	 // printf("pinv %d\n",Llu->inv);
	 // fflush(stdout);
		int_t maxsuper = sp_ienv_dist(3);
		if(MAXSUPER<maxsuper){
			printf("increase MAXSUPER\n");
			exit(1);
		}
		dim3 dimBlock(nthread_x, nthread_y);
		dlsum_bmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Urbs,Ufstnz_br_dat,Ufstnz_br_offset,Unzval_br_dat,Unzval_br_offset,Ucb_valdat,Ucb_valoffset,Ucb_inddat,Ucb_indoffset,Uinv_bc_dat,Uinv_bc_offset,xsup,grid);



 
	 gpuDeviceSynchronize();
 }

 
 
#ifdef __cplusplus
}
#endif
