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
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */

 #include <math.h>
 #include "superlu_zdefs.h"

 #ifndef BLK_M
 #define BLK_M  DIM_X*2
 #define BLK_N  DIM_Y*2
 #define BLK_K 512/(BLK_M)
 #endif

 #ifndef CACHELINE
 #define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
 #endif

 #ifndef MAXSUPER
 #define MAXSUPER 256
 #endif

#ifndef RDMA_FLAG_SIZE
#define RDMA_FLAG_SIZE 2
#endif

#include <stdio.h>
#include "mpi.h"
#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <cooperative_groups.h>
#include <nvml.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


// #define USESHARE1RHS 1

//TODO: zgemv_device_dlsum_fmod is not used anymore and hence hasn't been created


__device__ doublecomplex z_atomicAdd(doublecomplex* a, doublecomplex b){
  //transform the addresses of real and imag. parts to double pointers
  doublecomplex out;
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  out.r=atomicAdd(x, b.r);
  out.i=atomicAdd(y, b.i);
  return out;
}




/******************************************************************************/
static __device__
void gemm_device_zlsum_fmod(
        int M, int N, int K,
        int blx, int bly,
        const doublecomplex* __restrict__ A, int LDA,
        const doublecomplex* __restrict__ B, int LDB,
        doublecomplex rC[THR_N][THR_M],
        doublecomplex alpha, doublecomplex beta)
{
    // #if (__CUDA_ARCH__ >= 200)
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    // int blx = blockIdx_x;   // block's m dimension
    // int bly = blockIdx_y;   // block's n dimension

    __shared__ doublecomplex sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ doublecomplex sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    doublecomplex rA[THR_M];
    doublecomplex rB[THR_N];

    doublecomplex ra[BLK_K/DIM_YA+1][BLK_M/DIM_XA];
    doublecomplex rb[BLK_N/DIM_YB][BLK_K/DIM_XB+1];

    const doublecomplex *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    const doublecomplex *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    int boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
    int boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;
    doublecomplex zero = {0.0, 0.0};

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


// #pragma unroll
//     for (n = 0; n < BLK_N; n += DIM_YB)
// #pragma unroll
//             for (m = 0; m < BLK_K; m += DIM_XB){
//                 int_t nn = min(n+idyB,N-1);
//                 int_t mm = min(m+idxB,K-1);
//                 // sB[n+idyB][m+idxB] = B[nn*LDB+mm+bly*BLK_N*LDB + idyB*LDB + idxB];
//                 sB[n+idyB][m+idxB] = B[bly*BLK_N*LDB + nn*LDB+mm];
//             }


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

// #pragma unroll
//         for (n = 0; n < BLK_N/DIM_YB; n++)
// #pragma unroll
//                 for (m = 0; m < BLK_K/DIM_XB; m++){
//                     int_t nn = min(n*DIM_YB+idyB,N-1);
//                     int_t mm = min(m*DIM_XB+idxB+kk+BLK_K,K-1);
//                     rb[n][m] = B[nn*LDB+mm+bly*BLK_N*LDB];
//                 }



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
                    doublecomplex ctemp;
	                zz_mult(&ctemp, &rA[m], &rB[n]);
	                z_add(&rC[n][m], &rC[n][m], &ctemp);
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
                doublecomplex ctemp;
                zz_mult(&ctemp, &rA[m], &rB[n]);
                z_add(&rC[n][m], &rC[n][m], &ctemp);

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

    // doublecomplex &regC = rC[n][m];
    // doublecomplex &memC = C[offsC];

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

    // doublecomplex &regC = rC[n][m];
    // doublecomplex &memC = C[offsC];

    // // memC = add(mul(alpha, regC), mul(beta, memC));
    // }
    // }
    // }
    // }
    // #endif /* (__CUDA_ARCH__ >= 200) */
}



/******************************************************************************/
static __device__
void gemm_device_zlsum_bmod_stridedB(
        int M, int N, int K,
        int blx, int bly,
        const doublecomplex* __restrict__ A, int LDA,
        const doublecomplex* __restrict__ B, int LDB,
        doublecomplex rC[THR_N][THR_M],
        doublecomplex alpha, doublecomplex beta,
        int_t lptr, int_t rel, int_t *usub)
{
    // #if (__CUDA_ARCH__ >= 200)
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    // int blx = blockIdx_x;   // block's m dimension
    // int bly = blockIdx_y;   // block's n dimension

    __shared__ doublecomplex sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ doublecomplex sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    doublecomplex rA[THR_M];
    doublecomplex rB[THR_N];

    doublecomplex ra[BLK_K/DIM_YA+1][BLK_M/DIM_XA];
    doublecomplex rb[BLK_N/DIM_YB][BLK_K/DIM_XB+1];

    const doublecomplex *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    // const doublecomplex *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    int boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
    // int boundB = (K*N) - ( bly*BLK_N*K + idyB*K + idxB ) -1;

    int m, n, k, kk;
    doublecomplex zero = {0.0, 0.0};

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
            for (m = 0; m < BLK_K; m += DIM_XB){
                int nn = min(n+idyB,N-1);
                int mm = min(m+idxB,K-1);
                int_t icol = usub[lptr+mm] - rel; /* Relative col. */
                sB[n+idyB][m+idxB] = B[bly*BLK_N*LDB + nn*LDB+icol];
            }

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;

        // offs_dB += BLK_K;
        // boundB  -= BLK_K;

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
#pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

#pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
#pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++){
                    int nn = min(n*DIM_YB+idyB,N-1);
                    int mm = min(m*DIM_XB+idxB+kk+BLK_K,K-1);
                    int_t icol = usub[lptr+mm] - rel; /* Relative col. */
                    rb[n][m] = B[nn*LDB+icol+bly*BLK_N*LDB];
                }

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
                    doublecomplex ctemp;
	                zz_mult(&ctemp, &rA[m], &rB[n]);
	                z_add(&rC[n][m], &rC[n][m], &ctemp);
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
                doublecomplex ctemp;
                zz_mult(&ctemp, &rA[m], &rB[n]);
                z_add(&rC[n][m], &rC[n][m], &ctemp);
            }
        }
    }
}


#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError();                           \
    if(e!=cudaSuccess) {                       \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));                           \
        exit(EXIT_FAILURE);                   \
    }                       \
}

#if 0
__global__ void simple_shift(int *target, int mype, int npes) {
    int tid = threadIdx.x;
    for (int i=tid; i<256;i=i+256){
        target[i]=i;
        printf("(%d,%d), val=%d\n",mype,tid,i);
    }
}
#endif

void zprepare_multiGPU_buffers(int flag_bc_size,int flag_rd_size,int ready_x_size,int ready_lsum_size,int my_flag_bc_size,int my_flag_rd_size){
#ifdef HAVE_NVSHMEM
    flag_bc_q = (uint64_t *)nvshmem_malloc( flag_bc_size * sizeof(uint64_t)); // for sender
    flag_rd_q = (uint64_t *)nvshmem_malloc( flag_rd_size * sizeof(uint64_t)); // for sender
    zready_x = (doublecomplex *)nvshmem_malloc( ready_x_size * sizeof(doublecomplex)); // for receiver
    zready_lsum = (doublecomplex *)nvshmem_malloc( ready_lsum_size * sizeof(doublecomplex)); // for receiver
    my_flag_bc = (int *) nvshmem_malloc ( my_flag_bc_size * sizeof(int)); // for sender
    my_flag_rd = (int *) nvshmem_malloc ( my_flag_rd_size * sizeof(int)); // for sender


    checkGPU(gpuMemset(my_flag_bc, 0, my_flag_bc_size * sizeof(int)));
    checkGPU(gpuMemset(my_flag_rd, 0, my_flag_rd_size * sizeof(int)));
    checkGPU(gpuMemset(zready_x, 0, ready_x_size * sizeof(doublecomplex)));
    checkGPU(gpuMemset(zready_lsum, 0, ready_lsum_size * sizeof(doublecomplex)));


	// int iam;
    // MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &iam));
    //printf("(%d) in zprepare_multiGPU_buffers:\n "
    //       "flag_bc_size=%d int, zready_x=%d doublecomplex, "
    //       "flag_rd_size=%d int, zready_lsum=%d doublecomplex, "
    //       "int=%d B, doublecomplex=%d B\n",
    //       iam,
    //       flag_bc_size, ready_x_size,
    //       flag_rd_size , ready_lsum_size,
    //       sizeof(int), sizeof(doublecomplex) );
    //fflush(stdout);
#endif
}

void zdelete_multiGPU_buffers(){
#ifdef HAVE_NVSHMEM
    nvshmem_free(my_flag_bc);
    nvshmem_free(my_flag_rd);
    nvshmem_free(zready_x);
    nvshmem_free(zready_lsum);
    nvshmem_finalize();
#endif
}

__device__ void zC_BcTree_forwardMessageSimple_Device(C_Tree* tree,  volatile uint64_t* flag_bc_q,  int* my_flag_bc, int mype, int tid,doublecomplex* zready_x, int maxrecvsz){
#ifdef HAVE_NVSHMEM
//int BCsendoffset;
    uint64_t sig = 1;
    int data_ofset=my_flag_bc[0]*maxrecvsz;
    for( int idxRecv = 0; idxRecv < tree->destCnt_; ++idxRecv ) {
        int iProc = tree->myDests_[idxRecv];
        nvshmemx_double_put_signal_nbi_block((double*)&zready_x[data_ofset], (double*)&zready_x[data_ofset], my_flag_bc[1]*2,(uint64_t*)(flag_bc_q + my_flag_bc[0]), sig, NVSHMEM_SIGNAL_SET,iProc);
    }
#endif
}
__device__ void zC_RdTree_forwardMessageSimple_Device(C_Tree* Tree, volatile uint64_t* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, doublecomplex* zready_lsum, int maxrecvsz, int myroot){
    #ifdef HAVE_NVSHMEM
        int data_ofset,sig_ofset;
        uint64_t sig = 1;
        if (Tree->myIdx % 2 == 0) {
            sig_ofset = my_flag_rd[0] * 2;
            data_ofset = my_flag_rd[0] * maxrecvsz * 2;
        } else {
            sig_ofset = my_flag_rd[0] * 2 + 1;
            data_ofset = my_flag_rd[0] * maxrecvsz * 2 + maxrecvsz;
        }
        nvshmem_double_put_signal_nbi((double*)&zready_lsum[data_ofset],(double*)&zready_lsum[data_ofset],my_flag_rd[1],(uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);


    // ////forward to my root if I have received everything
    // //double sum = 0;

    // //for (int i = my_flag_rd[0]*maxrecvsz*2; i < my_flag_rd[0]*maxrecvsz*2 + my_flag_rd[1]; i++) {
    // //    //printf("(%d), data, %d\n",mype,i);
    // //    printf("(%d), data, %d,%lf\n", mype, i, zready_lsum[i]);
    // //    sum += zready_lsum[i];
    // //}

    // //printf("---- Start RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
    // //       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
    // //       mype, bid,tid, myroot,
    // //       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
    //     nvshmem_double_put_nbi(&zready_lsum[data_ofset],&zready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],myroot);
    // //printf("---- END RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
    // //       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
    // //       mype, bid,tid, myroot,
    // //       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
    //     nvshmem_fence();
    //     int sig=1;
    //     nvshmemx_signal_op((uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);
    // //printf("Tsend:%d,%d,%d,%d,%d\n",
    // //       mype, my_flag_rd[0],data_ofset, sig_ofset,my_flag_rd[1]);
    #endif
    }




//__device__ void zC_RdTree_forwardMessageBlock_Device(C_Tree* Tree, volatile int* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, doublecomplex* zready_lsum, int maxrecvsz, int myroot){
//#ifdef HAVE_NVSHMEM
//    int data_ofset,sig_ofset;
//if (Tree->myIdx % 2 == 0) {
//    sig_ofset = my_flag_rd[0] * 2;
//    data_ofset = my_flag_rd[0] * maxrecvsz * 2;
//} else {
//    sig_ofset = my_flag_rd[0] * 2 + 1;
//    data_ofset = my_flag_rd[0] * maxrecvsz * 2 + maxrecvsz;
//}
//////forward to my root if I have received everything
////double sum = 0;
////if (tid==0){
////    for (int i = my_flag_rd[0]*maxrecvsz*2; i < my_flag_rd[0]*maxrecvsz*2 + my_flag_rd[1]; i++) {
////        //printf("(%d), data, %d\n",mype,i);
////        //printf("(%d), data, %d,%lf\n", mype, i, zready_lsum[i]);
////        sum += zready_lsum[i];
////    }
////    printf("---- Start RD (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
////       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
////       mype, bid,tid, myroot,
////       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
////}
//
//@precision DOUBLE
//
//    nvshmemx_double_put_nbi_block(&zready_lsum[data_ofset],&zready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],myroot);
////if (tid==0)
////    printf("---- END RD (%d), forwardMessage, forwardDevice, send to %d, "
////       "lib=%d,size=%d, sum=%lf, dataoffset=%d,maxrecvsz=%d\n",
////       mype, myroot,
////       my_flag_rd[0], my_flag_rd[1], sum, data_ofset,maxrecvsz);
//    nvshmem_fence();
//    int sig=1;
//    if (tid==0)  nvshmemx_signal_op((uint64_t*)(flag_rd_q+sig_ofset), sig, NVSHMEM_SIGNAL_SET,myroot);
//    //if (tid==0)  nvshmemx_int_signal((int*)flag_rd_q+sig_ofset, sig, myroot);
//@precision SCOMPLEX DCOMPLEX SINGLE
//        //TODO: nvshmemx_double_put_nbi_block and nvshmemx_int_signal not yet working for other precisions?
//@precision !
//
////if (tid==0)
////    printf("Bsend:%d,%d,%d,%d,%d\n", mype, my_flag_rd[0],data_ofset, sig_ofset,my_flag_rd[1]);
//#endif
//}
//
//
//__device__ void zC_RdTree_forwardMessageWarp_Device(C_Tree* Tree, volatile uint64_t* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, doublecomplex* zready_lsum, int maxrecvsz, int myroot){
//#ifdef HAVE_NVSHMEM
//    int data_ofset,sig_ofset;
//    if (Tree->myIdx % 2 == 0) {
//        sig_ofset = my_flag_rd[0] * 2;
//        data_ofset = my_flag_rd[0] * maxrecvsz * 2;
//    } else {
//        sig_ofset = my_flag_rd[0] * 2 + 1;
//        data_ofset = my_flag_rd[0] * maxrecvsz * 2 + maxrecvsz;
//    }
//
//////forward to my root if I have received everything
////double sum = 0;
////if (tid%32==0) {
////    for (int i = my_flag_rd[0]*maxrecvsz*2; i < my_flag_rd[0]*maxrecvsz*2 + my_flag_rd[1]; i++) {
////        //printf("(%d), data, %d\n",mype,i);
////        //printf("(%d), data, %d,%lf\n", mype, i, zready_lsum[i]);
////        sum += zready_lsum[i];
////    }
////    printf("---- Start RD Warp (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
////           "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
////           mype, bid, tid, myroot,
////           my_flag_rd[0], my_flag_rd[1], data_ofset, maxrecvsz);
////}
//
//@precision DOUBLE
//    nvshmemx_double_put_nbi_warp(&zready_lsum[data_ofset],&zready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],myroot);
////if (tid%32==0)
////    printf("---- END RD Warp (%d), forwardMessage, forwardDevice, send to %d, "
////       "lib=%d,size=%d, sum=%lf, dataoffset=%d,maxrecvsz=%d\n",
////       mype, myroot,
////       my_flag_rd[0], my_flag_rd[1], sum, data_ofset,maxrecvsz);
//    nvshmem_fence();
//    int sig=1;
//    if (tid%32==0)  nvshmemx_signal_op((uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);
////if (tid%32==0)
////    printf("Wsend:%d,%d,%d,%d,%d\n", mype, my_flag_rd[0],data_ofset, sig_ofset,my_flag_rd[1]);
//
//@precision SCOMPLEX DCOMPLEX SINGLE
//        //TODO: nvshmemx_double_put_nbi_warp and nvshmemx_signal_op not yet working for other precisions?
//@precision !
//
//
//#endif
//}
//
//
//
//
//__device__ void zC_RdTree_forwardMessageThread_Device(C_Tree* Tree, volatile uint64_t* flag_rd_q, int* my_flag_rd, int mype, int bid, int tid, doublecomplex* zready_lsum, int maxrecvsz, int myroot){
//#ifdef HAVE_NVSHMEM
//    int data_ofset,sig_ofset;
//    if (Tree->myIdx % 2 == 0) {
//        sig_ofset = my_flag_rd[0] * 2;
//        data_ofset = my_flag_rd[0] * maxrecvsz * 2;
//    } else {
//        sig_ofset = my_flag_rd[0] * 2 + 1;
//        data_ofset = my_flag_rd[0] * maxrecvsz * 2 + maxrecvsz;
//    }
//
//////forward to my root if I have received everything
////double sum = 0;
//
////for (int i = my_flag_rd[0]*maxrecvsz*2; i < my_flag_rd[0]*maxrecvsz*2 + my_flag_rd[1]; i++) {
////    //printf("(%d), data, %d\n",mype,i);
////    printf("(%d), data, %d,%lf\n", mype, i, zready_lsum[i]);
////    sum += zready_lsum[i];
////}
//
////printf("---- Start RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
////       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
////       mype, bid,tid, myroot,
////       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
//
//@precision DOUBLE
//    nvshmem_double_put_nbi(&zready_lsum[data_ofset],&zready_lsum[my_flag_rd[0]*maxrecvsz*2],my_flag_rd[1],myroot);
////printf("---- END RD Thread (%d,%d,%d), forwardMessage, forwardDevice, send to %d, "
////       "lib=%d,size=%d,  dataoffset=%d,maxrecvsz=%d\n",
////       mype, bid,tid, myroot,
////       my_flag_rd[0], my_flag_rd[1], data_ofset,maxrecvsz);
//    nvshmem_fence();
//    int sig=1;
//    nvshmemx_signal_op((uint64_t*)flag_rd_q+sig_ofset, sig, NVSHMEM_SIGNAL_SET, myroot);
////printf("Tsend:%d,%d,%d,%d,%d\n",
////       mype, my_flag_rd[0],data_ofset, sig_ofset,my_flag_rd[1]);
//
//
//@precision SCOMPLEX DCOMPLEX SINGLE
//        //TODO: nvshmem_double_put_nbi and nvshmemx_signal_op not yet working for other precisions?
//@precision !
//
//#endif
//}


// Yang/Nan. Note that NVSHMEM-based SpTRSV has only been tested on Perlmutter and Summit. Here is the status of the code on the two mahchines:
// On Perlmutter:
//      #ifdef _USE_SUMMIT: hangs for any Px*Py>1 configuration
//      #else: no hangs
// On Summit:
//      #ifdef _USE_SUMMIT: no hangs
//      #else: no hangs for Px=1 or Py=1, but hangs for some matrices for Px>1, Py>1
// Therefore, on Summit one should always define _USE_SUMMIT in C_FLAGS; on Perlmutter one should not define it.


__global__ void zwait_bcrd
        (
                int nrhs,
                C_Tree  *LRtree_ptr,
                int_t maxrecvsz,
                int mype,
                uint64_t* flag_bc_q,
                uint64_t* flag_rd_q,
                doublecomplex* zready_x,
                doublecomplex* zready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_nfrecv,
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
                int* d_flag_mod,
                doublecomplex *lsum,    /* Sum of local modifications.                        */
                int *fmod,     /* Modification count for L-solve.                    */
                gridinfo_t *grid,
                int_t *xsup,
                int_t *ilsum,
                int nbrow_loc,
                int_t  nsupers
        ) {
#ifdef HAVE_NVSHMEM
    int bid = blockIdx.x;
//int global_id= blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int WAIT_NUM_THREADS = d_nfrecv[1]; //*d_nfrecv[2];
//if (tid==0) printf("(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecv[0]);
#ifdef _USE_SUMMIT
     if (bid < 4) { // for BC recv
	tid=bid*WAIT_NUM_THREADS+tid;
	WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
#else
     if (bid == 0) { // for BC recv
#endif
   //if (tid < WAIT_NUM_THREADS) { // for BC recv
   //if (tid==0) printf("(%d) WAIT_NUM_THREADS=%d,bc_tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecv[0]);
       if (WAIT_NUM_THREADS >= d_nfrecv[0]) {
           if (tid < d_nfrecv[0]) {
               nvshmem_signal_wait_until((uint64_t *) (flag_bc_q + d_colnum[tid]), NVSHMEM_CMP_EQ, 1);
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
               int wm_val = nvshmem_uint64_wait_until_any(flag_bc_q + d_colnum[d_mymaskstart[tid]], d_mymasklength[tid],
                                                       d_status + d_colnum[d_mymaskstart[tid]], NVSHMEM_CMP_EQ, 1);
               d_status[d_colnum[d_mymaskstart[tid]] + wm_val] = 1;
               //printf("WAIT2 (%d,%d) msg arrived in col %d, i=%d\n",mype,tid,d_colnum[d_mymaskstart[tid]] + wm_val, i);
           }
       }
   }
//if (tid==0) printf("(%d,%d,%d) WAIT EXIT\n",mype,bid,tid);
#ifdef _USE_SUMMIT
    if (bid >=4) { // for RD recv
        tid=(bid-4)*WAIT_NUM_THREADS+tid;
        WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
#else
    if (bid == 1) { // for RD recv
#endif
        //if (tid==0) printf("RD---(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecvmod[1]);
       int j, iam, lib, myrow, k, knsupc, il, cnt;
       int fmod_tmp, aln_i;

       aln_i = 1;
    //   doublecomplex temp;
       if (WAIT_NUM_THREADS >= d_nfrecvmod[1]) { // one thread wait for one col
           if (tid < d_nfrecvmod[1]) {
               //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]]);
               for (int i = 0; i < d_recv_cnt[d_colnummod[tid]]; i++) {
                   //printf("(%d,%d,%d) I'm waiting for %d/%d msg in row %d.wait_off=%d,%d,status=%d,%d\n",mype, bid,tid, i,d_recv_cnt[d_colnummod[tid]],d_colnummod[tid],d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
           //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d,i=%d,wait_off=%d,%d,status=%d,%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]],i,d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
                   int wm_val = nvshmem_uint64_wait_until_any(flag_rd_q + d_colnummod[tid] * 2, 2,
                                                           d_statusmod + d_colnummod[tid] * 2, NVSHMEM_CMP_EQ, 1);
                   d_statusmod[d_colnummod[tid] * 2 + wm_val] = 1;
                   lib = (d_colnummod[tid] * 2 + wm_val) / 2;
                   iam = grid->iam;
               //    mycol = MYCOL(iam, grid);
                   myrow = MYROW(iam, grid);
                   k = myrow + lib * grid->nprow; // global block row
                   knsupc = SuperSize(k);
                   il = LSUM_BLK(lib);
                   cnt = LRtree_ptr[lib].destCnt_;
                   //printf("recv1,%d,%d,%d,%d\n",
                   //       mype,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
                   //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d,i=%d/%d\n", mype, bid, tid,
                   //       d_colnummod[tid] * 2 + wm_val, lib, cnt,i,d_recv_cnt[d_colnummod[tid]]);
                   if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                       //double tmp_sum = 0;
                       int ii = 0;
                       if (cnt == 2) {
                           for (ii = 0; ii < cnt; ++ii) {
                               //tmp_sum = 0;
                               RHS_ITERATE(j) {
                                   for (int aab = 0; aab < knsupc; ++aab) {
                                       //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                       z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                        zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                   j * knsupc]);
                                       //tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                       //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,zready_lsum[%d]=%f\n", mype, bid, tid,
                                       //       lib, k, ii, tmp_sum,
                                       //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                       //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                   }

                                   // atomic return old val
                                   fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d, tmp_sum=%lf\n", mype, bid, tid, lib, k,
                                   //       tmp_sum,fmod_tmp, tmp_sum);
                                   //printf("sum2-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                               }
                           }
                       }
                       if (cnt == 1) {
                           if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                           //tmp_sum = 0;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; ++aab) {
                                   //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                   z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                    zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                   //tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                   //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,zready_lsum[%d]=%lf\n", mype, bid, tid, lib, k, ii,
                                   //       tmp_sum,maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc,
                                   //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                               }

                           }
                           // atomic return old val
                           fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,fmod_tmp=%d\n", mype, bid, tid, lib, k,fmod_tmp);
                           //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                       }

                       if (fmod_tmp == 1) {// forward RD
                           //senddone[lk]=1;
                           if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                               //cnt=LRtree_ptr[lib].msgSize_;
                               my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                               my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                               //double tmp_sum=0;
                               RHS_ITERATE(j) {
                                   for (int aab = 0; aab < knsupc; aab++) {
                                       zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                       //tmp_sum += zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc];
                                       //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                       //       k * maxrecvsz * 2 + i +j * knsupc,
                                       //       zready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                   }
                               }
                               //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                               //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                               //int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                               //int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lib);
                               //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                               //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d, sum=%lf\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod, tmp_sum);
                               zC_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], flag_rd_q,
                                                                    &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                    &zready_lsum[0], maxrecvsz,LRtree_ptr[lib].myRoot_ );
                           }
                       }
                   }
               }//for
           }
       } else {
           int delta = d_nfrecvmod[1] % WAIT_NUM_THREADS;
           //int mynum = d_nfrecvmod[1] / WAIT_NUM_THREADS;
           //int mystart = tid*(mynum+1);
           //if (tid < delta) {
           //    mynum = mynum + 1;
           //    //d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS+1;
           //}else{
           //    mystart = (delta*(mynum+1))+(tid-delta)*mynum;
           //}
           //int mymasklength=(d_colnummod[mystart + mynum - 1] - d_colnummod[mystart]+1)*2;

           if (tid < delta){
               d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS + 1;
           }else {
               d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS;
           }
           __syncthreads();

           d_mymaskstartmod[tid] = 0;
           d_mymasklengthmod[tid] = 0;
           d_msgnum[tid] = 0;

           ////d_mymaskstartmod: start offset of d_colnummod
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

           //printf("(%d,%d), mynum=%d,%d,mystart=%d,%d, mylength=%d,%d\n",mype,tid,d_mynummod[tid], mynum,d_mymaskstartmod[tid], mystart, d_mymasklengthmod[tid],mymasklength);
           for (int i = 0; i < d_msgnum[tid]; i++) {
               //printf("(%d,%d,%d)--before wait any,i=%d/%d\n",mype,bid,tid,i,d_msgnum[tid]);
               int wm_val = nvshmem_uint64_wait_until_any(&flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                       d_mymasklengthmod[tid],
                                                       &d_statusmod[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                       NVSHMEM_CMP_EQ, 1);
               d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val] = 1;
               lib = (d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val) / 2;
                   //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d,i=%d/%d\n", mype, bid, tid,
                   //       d_colnummod[d_mymaskstartmod[tid]] * 2 + wm_val, lib, cnt,i,d_msgnum[tid]);
               //printf("recv,%d,%d,%d,%d,%d\n",
               //       mype,tid,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
               iam = grid->iam;
           //    mycol = MYCOL(iam, grid);
               myrow = MYROW(iam, grid);

               k = myrow + lib * grid->nprow; // global block row
               knsupc = SuperSize(k);
               il = LSUM_BLK(lib);
               cnt = LRtree_ptr[lib].destCnt_;
               //printf("HERE2-(%d,%d,%d),lib=%d,k=%d,wm_val=%d,cnt=%d,%d, mycnt=%d\n", mype, bid, tid, lib, k,
               //       wm_val,cnt,d_recv_cnt[lib],d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1]);

               if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                   //double tmp_sum = 0;
                   int ii = 0;
                   if (cnt == 2) {
                       for (ii = 0; ii < cnt; ++ii) {
                           //tmp_sum = 0;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; aab++) {
                                   //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                   z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                    zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                               j * knsupc]);
                                   //tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                   //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,zready_lsum[%d]=%f\n", mype, bid, tid,
                                   //       lib, k, ii,
                                   //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                   //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                               }

                               // atomic return old val
                               fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                               //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum,fmod_tmp);
                           }
                       }
                   }
                   if (cnt == 1) {
                       if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                       RHS_ITERATE(j) {
                           for (int aab = 0; aab < knsupc; ++aab) {
                               z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                               //tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                               //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
                               //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                               //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                           }

                       }
                       // atomic return old val
                       fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
                       //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
                   }

                   if (fmod_tmp == 1) {// forward RD
                       //printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,LRtree_ptr[lib].myRoot_);
                       if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
                           my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                           my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
                           RHS_ITERATE(j) {
                               for (int aab = 0; aab < knsupc; aab++) {
                                   zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                   //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                   //       k * maxrecvsz * 2 + i +j * knsupc,
                                   //       zready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                               }
                           }
                           //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                           //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                           //int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                           //int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lib);
                           //printf("iam=%d in wait2,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                           zC_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], flag_rd_q,
                                                                &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                &zready_lsum[0], maxrecvsz,LRtree_ptr[lib].myRoot_);
                       }
                   }
               }
           }//for
       } // else WAIT_NUM_THREAD<recv
   }
#if 0
    if (bid==2){
        int tot_threads=blockDim.x * blockDim.y;
        //if (tid==0){
        //    printf("iam=%d, len=%d, tot_threads=%d\n",mype,d_nfrecvmod[3],tot_threads);
        //}

        //if (d_nfrecvmod[3]==0) return;
        int lk=-1,k=-1,iam=-1,myroot=-1,myrank=-1;
        int mycol,myrow,lib;
        __shared__ int recv_num, finish_num;
        __shared__ int cur_send_num;
        recv_num = finish_num = 0;

        for (int i=1; i<d_nfrecvmod[3]+1;i=i+cur_send_num){

            if (tid==0){
                int tmp, tmp1;
                //printf("iam=%d,i=%d, count=%d\n",mype,i,d_flag_mod[0]);
                do {
                    tmp = d_flag_mod[0];
                    //tmp1 == d_flag_mod[tmp];
                    __threadfence();
                    //msg_recv=d_status[gc];
                    //msg_recv=flag_bc_q[gc];
                } while (tmp == finish_num);

                recv_num=tmp;
            }
            __syncthreads();
            cur_send_num=recv_num-finish_num;
            finish_num=recv_num;
            //if (cur_send_num==1) {
            //    lk=d_flag_mod[i];
            //    iam = grid->iam;
            //    mycol = MYCOL(iam, grid);
            //    myrow = MYROW(iam, grid);
            //    k = myrow + lk * grid->nprow; // global block row
            //    myroot=LRtree_ptr[lk].myRoot_;
            //    myrank=LRtree_ptr[lk].myRank_;
            //    //if (tid==0) printf("B,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype,tid,i, recv_num,cur_send_num,lk, myroot);
            //    //__syncthreads();
            //    zC_RdTree_forwardMessageBlock_Device(&LRtree_ptr[lk], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
            //    //__syncthreads();
            //    //if (tid==0) printf("B Done,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i, recv_num,cur_send_num);
            //}else if ((cur_send_num <= tot_threads/32) && (cur_send_num >1)){
            if ((cur_send_num <= tot_threads/32)){
                if (tid/32 < cur_send_num){
                    lk=d_flag_mod[i+tid/32];
                    //if (tid%32==0) printf("-- Warp, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //if (tid%32==0) printf("W, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, lk=%d, to %d\n",mype,tid,i, recv_num,cur_send_num, lk, myroot);
                    zC_RdTree_forwardMessageWarp_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //if (tid%32==0) printf("W Done, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, lk=%d\n",mype,tid,i, recv_num,cur_send_num,lk);
                }
            }else if ((cur_send_num > tot_threads/32) && (cur_send_num <= tot_threads)){
                __syncthreads();
                if (tid < cur_send_num){
                    lk=d_flag_mod[i+tid];
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //printf("-- Thread, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    zC_RdTree_forwardMessageThread_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //printf("T Done,(%d,%d) recv_num=%d,cur_send_num=%d\n",mype,tid, recv_num,cur_send_num);
                }
            }else if (cur_send_num > tot_threads){
                int delta=cur_send_num%tot_threads;
                int mynum=cur_send_num/tot_threads;
                int myoffset=0;
                if (tid < delta){
                    myoffset=tid*(mynum+1);
                }else{
                    myoffset=(delta*(mynum+1))+(tid-delta)*mynum;
                }

                for(int j=0;j<mynum;j++){
                    lk=d_flag_mod[i+myoffset+j];
                    iam = grid->iam;
                    mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=LRtree_ptr[lk].myRoot_;
                    myrank=LRtree_ptr[lk].myRank_;
                    //printf("-- Threadloop, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    zC_RdTree_forwardMessageThread_Device(&LRtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //printf("Ts Done,(%d,%d) loop=%d (%d,%d) recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype, tid,i, j, mynum,recv_num,cur_send_num,lk,myroot);
                }
            }

            __syncthreads();

            //if (tid==0) printf("iam=%d,tid=%d,i=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i,recv_num,cur_send_num);
        }
    }
#endif
#endif
}


__global__ void zwait_bcrd_u
        (
                int nrhs,
                C_Tree  *URtree_ptr,
                int_t maxrecvsz,
                int mype,
                uint64_t* flag_bc_q,
                uint64_t* flag_rd_q,
                doublecomplex* zready_x,
                doublecomplex* zready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_nfrecv,
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
                int* d_flag_mod_u,
                doublecomplex *lsum,    /* Sum of local modifications.                        */
                int *bmod,     /* Modification count for L-solve.                    */
                gridinfo_t *grid,
                int_t *xsup,
                int_t *ilsum,
                int nbrow_loc,
                int_t  nsupers
        ) {
#ifdef HAVE_NVSHMEM
    int bid = blockIdx.x;
//int global_id= blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int WAIT_NUM_THREADS = d_nfrecv[1]; //*d_nfrecv[2];
    #ifdef _USE_SUMMIT
    if (bid <4) { // for BC recv
          tid=bid*WAIT_NUM_THREADS+tid;
          WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
    #else
        if (bid == 0) { // for BC recv
    #endif
        if (WAIT_NUM_THREADS >= d_nfrecv[0]) {
            if (tid < d_nfrecv[0]) {
                nvshmem_signal_wait_until((uint64_t *) flag_bc_q + d_colnum[tid], NVSHMEM_CMP_EQ, 1);
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
                int wm_val = nvshmem_uint64_wait_until_any(flag_bc_q + d_colnum[d_mymaskstart[tid]], d_mymasklength[tid],
                                                        d_status + d_colnum[d_mymaskstart[tid]], NVSHMEM_CMP_EQ, 1);
                d_status[d_colnum[d_mymaskstart[tid]] + wm_val] = 1;
                //printf("WAIT2 (%d,%d) msg arrived in col %d, i=%d\n",mype,tid,d_colnum[d_mymaskstart[tid]] + wm_val, i);
            }
        }
    }
//if (tid==0) printf("(%d,%d,%d) WAIT EXIT\n",mype,bid,tid);
#ifdef _USE_SUMMIT
     if (bid >=4) { // for RD recv
        tid=(bid-4)*WAIT_NUM_THREADS+tid;
        WAIT_NUM_THREADS=WAIT_NUM_THREADS*4;
#else
     if (bid == 1) { // for RD recv
#endif
        //if (tid==0) printf("RD---(%d) WAIT_NUM_THREADS=%d,tot_wait_col=%d\n",mype,WAIT_NUM_THREADS,d_nfrecvmod[1]);
        int j, iam, lib, myrow, k, knsupc, il, cnt;
        int bmod_tmp, aln_i;

        aln_i = 1;
        // doublecomplex temp;
        if (WAIT_NUM_THREADS >= d_nfrecvmod[1]) { // one thread wait for one col
            if (tid < d_nfrecvmod[1]) {
                //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]]);
                for (int i = 0; i < d_recv_cnt[d_colnummod[tid]]; i++) {
                    //printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d,i=%d,wait_off=%d,%d,status=%d,%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]],i,d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
                    int wm_val = nvshmem_uint64_wait_until_any(flag_rd_q + d_colnummod[tid] * 2, 2,
                                                            d_statusmod + d_colnummod[tid] * 2, NVSHMEM_CMP_EQ, 1);
                    d_statusmod[d_colnummod[tid] * 2 + wm_val] = 1;
                    lib = (d_colnummod[tid] * 2 + wm_val) / 2;
                    iam = grid->iam;
                    // mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lib * grid->nprow; // global block row
                    knsupc = SuperSize(k);
                    il = LSUM_BLK(lib);
                    cnt = URtree_ptr[lib].destCnt_;
                    //printf("recv1,%d,%d,%d,%d\n",
                    //       mype,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
                    //printf("(%d,%d,%d),idx=%d,lib=%d,cnt=%d\n", mype, bid, tid,
                    //       d_colnummod[tid] * 2 + wm_val, lib, cnt);
                    if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                        //double tmp_sum = 0;
                        int ii = 0;
                        if (cnt == 2) {
                            for (ii = 0; ii < cnt; ++ii) {
                                // double tmp_sum = 0;
                                RHS_ITERATE(j) {
                                    for (int aab = 0; aab < knsupc; ++aab) {
                                        //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                        z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                         zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                    j * knsupc]);

                                        // tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                        //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,zready_lsum[%d]=%f\n", mype, bid, tid,
                                        //       lib, k, ii, tmp_sum,
                                        //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                        //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                    }

                                    // atomic return old val
                                    bmod_tmp = atomicSub(&bmod[lib * aln_i], 1);
                                    //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,bmod_tmp=%d, tmp_sum=%lf\n", mype, bid, tid, lib, k,
                                    //       tmp_sum,bmod_tmp, tmp_sum);
                                    //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%lf,bmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum, bmod_tmp);
                                }
                            }
                        }
                        if (cnt == 1) {
                            if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                            // double tmp_sum = 0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; ++aab) {
                                    //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                    z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                     zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                    // tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                    //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,sum=%lf,zready_lsum[%d]=%lf\n", mype, bid, tid, lib, k, ii,
                                    //       tmp_sum,maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc,
                                    //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                }

                            }
                            // atomic return old val
                            bmod_tmp = atomicSub(&bmod[lib * aln_i], 1);
                            //printf("u sum1-(%d,%d,%d),lib=%d,k=%d,sum=%lf,bmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum, bmod_tmp);
                            //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,bmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,bmod_tmp);
                        }

                        if (bmod_tmp == 1) {// forward RD
                            //senddone[lk]=1;
                            if (URtree_ptr[lib].myRoot_ != URtree_ptr[lib].myRank_) {
                                //cnt=URtree_ptr[lib].msgSize_;
                                my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                                my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = URtree_ptr[lib].msgSize_;
                                // double tmp_sum=0;
                                RHS_ITERATE(j) {
                                    for (int aab = 0; aab < knsupc; aab++) {
                                        zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                        // tmp_sum += zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc];
                                        //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, aab,
                                        //       lib * maxrecvsz * 2 + aab + j * knsupc,
                                        //       zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc]);

                                    }
                                }
                                //printf("(%d,%d,%d),in u wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                                //       my_flag_rd[lib * RDMA_FLAG_SIZE], my_flag_rd[lib * RDMA_FLAG_SIZE + 1]);
                                // int temp_mysendcout=atomicAdd(&d_flag_mod_u[0], 1);
                                // int temp_flag_mod=atomicExch(&d_flag_mod_u[temp_mysendcout+1],lib);
                                //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod_u[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                                //printf("iam=%d in wait,lib=%d,%d,%d, pos=%d, temp %d,%d, sum=%lf\n",mype,lib,k, d_flag_mod_u[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod, tmp_sum);
                                zC_RdTree_forwardMessageSimple_Device(&URtree_ptr[lib], flag_rd_q,
                                                                     &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                     &zready_lsum[0], maxrecvsz, URtree_ptr[lib].myRoot_);
                            }
                        }
                    }
                }//for
            }
        } else {
            int delta = d_nfrecvmod[1] % WAIT_NUM_THREADS;
            //int mynum = d_nfrecvmod[1] / WAIT_NUM_THREADS;
            //int mystart = tid*(mynum+1);
            //if (tid < delta) {
            //    mynum = mynum + 1;
            //    //d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS+1;
            //}else{
            //    mystart = (delta*(mynum+1))+(tid-delta)*mynum;
            //}
            //int mymasklength=(d_colnummod[mystart + mynum - 1] - d_colnummod[mystart]+1)*2;

            if (tid < delta){
                d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS + 1;
            }else {
                d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS;
            }
            __syncthreads();

            d_mymaskstartmod[tid] = 0;
            d_mymasklengthmod[tid] = 0;
            d_msgnum[tid] = 0;

            ////d_mymaskstartmod: start offset of d_colnummod
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
                int wm_val = nvshmem_uint64_wait_until_any(&flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                        d_mymasklengthmod[tid],
                                                        &d_statusmod[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                                        NVSHMEM_CMP_EQ, 1);
                d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val] = 1;
                lib = (d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val) / 2;
                //printf("recv,%d,%d,%d,%d,%d\n",
                //       mype,tid,d_colnummod[d_mymaskstartmod[tid]]*2,wm_val,lib);
                iam = grid->iam;
                // mycol = MYCOL(iam, grid);
                myrow = MYROW(iam, grid);

                k = myrow + lib * grid->nprow; // global block row
                knsupc = SuperSize(k);
                il = LSUM_BLK(lib);
                cnt = URtree_ptr[lib].destCnt_;
                //printf("HERE2-(%d,%d,%d),lib=%d,k=%d,wm_val=%d,cnt=%d,%d, mycnt=%d\n", mype, bid, tid, lib, k,
                //       wm_val,cnt,d_recv_cnt[lib],d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1]);

                if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
                    // double tmp_sum = 0;
                    int ii = 0;
                    if (cnt == 2) {
                        for (ii = 0; ii < cnt; ++ii) {
                            // tmp_sum = 0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; aab++) {
                                    //temp=z_atomicAdd(&lsum[il+i + j*knsupc], zready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
                                    z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                     zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab +
                                                                j * knsupc]);
                                    // tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                    //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,zready_lsum[%d]=%f\n", mype, bid, tid,
                                    //       lib, k, ii,
                                    //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                    //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                                }

                                // atomic return old val
                                bmod_tmp = atomicSub(&bmod[lib * aln_i], 1);
                                //printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,bmod_tmp=%d\n", mype, bid, tid, lib, k,tmp_sum,bmod_tmp);
                            }
                        }
                    }
                    if (cnt == 1) {
                        if (flag_rd_q[lib * 2 + 1] == 1) ii = 1;
                        // tmp_sum = 0;
                        RHS_ITERATE(j) {
                            for (int aab = 0; aab < knsupc; ++aab) {
                                z_atomicAdd(&lsum[il + aab + j * knsupc],
                                                 zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc]);
                                // tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
                                //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
                                //       zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
                            }

                        }
                        // atomic return old val
                        bmod_tmp = atomicSub(&bmod[lib * aln_i], 1);
                        //printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,bmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,bmod_tmp);
                    }

                    if (bmod_tmp == 1) {// forward RD
                        //printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,URtree_ptr[lib].myRoot_);
                        if (URtree_ptr[lib].myRoot_ != URtree_ptr[lib].myRank_) {
                            my_flag_rd[lib * RDMA_FLAG_SIZE] = lib;
                            my_flag_rd[lib * RDMA_FLAG_SIZE + 1] = URtree_ptr[lib].msgSize_;
                            // tmp_sum=0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < knsupc; aab++) {
                                    zready_lsum[lib * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
                                    // tmp_sum += zready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
                                    //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,zready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
                                    //       k * maxrecvsz * 2 + i +j * knsupc,
                                    //       zready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

                                }
                            }
                            //printf("sumforward-(%d,%d,%d),lib=%d,k=%d,sum=%f,bmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,bmod_tmp);
                            //printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
                            //       my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
                            // int temp_mysendcout=atomicAdd(&d_flag_mod_u[0], 1);
                            // int temp_flag_mod=atomicExch(&d_flag_mod_u[temp_mysendcout+1],lib);
                            //printf("iam=%d in wait2,lib=%d,%d,%d, pos=%d, temp %d,%d\n",mype,lib,k, d_flag_mod_u[temp_mysendcout+1], temp_mysendcout+1, temp_mysendcout,temp_flag_mod);
                            zC_RdTree_forwardMessageSimple_Device(&URtree_ptr[lib], flag_rd_q,
                                                                 &my_flag_rd[RDMA_FLAG_SIZE * lib], mype, bid, tid,
                                                                 &zready_lsum[0], maxrecvsz,URtree_ptr[lib].myRoot_);
                        }
                    }
                }
            }//for
        } // else WAIT_NUM_THREAD<recv
    }
#if 0
    if (bid==2){
        int tot_threads=blockDim.x * blockDim.y;
        //if (tid==0){
        //    printf("iam=%d, len=%d, tot_threads=%d\n",mype,d_nfrecvmod[3],tot_threads);
        //}

        //if (d_nfrecvmod[3]==0) return;
        int lk=-1,k=-1,iam=-1,myroot=-1,myrank=-1;
        int myrow,lib;
        __shared__ int recv_num, finish_num;
        __shared__ int cur_send_num;
        recv_num = finish_num = 0;

        for (int i=1; i<d_nfrecvmod[3]+1;i=i+cur_send_num){

            if (tid==0){
                int tmp, tmp1;
                //printf("iam=%d,i=%d, count=%d\n",mype,i,d_flag_mod_u[0]);
                do {
                    tmp = d_flag_mod_u[0];
                    //tmp1 == d_flag_mod_u[tmp];
                    __threadfence();
                    //msg_recv=d_status[gc];
                    //msg_recv=flag_bc_q[gc];
                } while (tmp == finish_num);

                recv_num=tmp;
            }
            __syncthreads();
            cur_send_num=recv_num-finish_num;
            finish_num=recv_num;
            //if (cur_send_num==1) {
            //    lk=d_flag_mod_u[i];
            //    iam = grid->iam;
            //    mycol = MYCOL(iam, grid);
            //    myrow = MYROW(iam, grid);
            //    k = myrow + lk * grid->nprow; // global block row
            //    myroot=URtree_ptr[lk].myRoot_;
            //    myrank=URtree_ptr[lk].myRank_;
            //    //if (tid==0) printf("B,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype,tid,i, recv_num,cur_send_num,lk, myroot);
            //    //__syncthreads();
            //    zC_RdTree_forwardMessageBlock_Device(&URtree_ptr[lk], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
            //    //__syncthreads();
            //    //if (tid==0) printf("B Done,(%d,%d) loop=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i, recv_num,cur_send_num);
            //}else if ((cur_send_num <= tot_threads/32) && (cur_send_num >1)){
            if ((cur_send_num <= tot_threads/32)){
                if (tid/32 < cur_send_num){
                    lk=d_flag_mod_u[i+tid/32];
                    //if (tid%32==0) printf("-- Warp, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    iam = grid->iam;
                    // mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=URtree_ptr[lk].myRoot_;
                    myrank=URtree_ptr[lk].myRank_;
                    //if (tid%32==0) printf("in U W, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype,tid,i, recv_num,cur_send_num, lk, myroot);
                    zC_RdTree_forwardMessageWarp_Device(&URtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //if (tid%32==0) printf("W Done, (%d,%d) loop=%d, recv_num=%d,cur_send_num=%d, lk=%d\n",mype,tid,i, recv_num,cur_send_num,lk);
                }
            }else if ((cur_send_num > tot_threads/32) && (cur_send_num <= tot_threads)){
                __syncthreads();
                if (tid < cur_send_num){
                    lk=d_flag_mod_u[i+tid];
                    iam = grid->iam;
                    // mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=URtree_ptr[lk].myRoot_;
                    myrank=URtree_ptr[lk].myRank_;
                    //printf("-- Thread, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    zC_RdTree_forwardMessageThread_Device(&URtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //printf("T Done,(%d,%d) recv_num=%d,cur_send_num=%d\n",mype,tid, recv_num,cur_send_num);
                }
            }else if (cur_send_num > tot_threads){
                int delta=cur_send_num%tot_threads;
                int mynum=cur_send_num/tot_threads;
                int myoffset=0;
                if (tid < delta){
                    myoffset=tid*(mynum+1);
                }else{
                    myoffset=(delta*(mynum+1))+(tid-delta)*mynum;
                }

                for(int j=0;j<mynum;j++){
                    lk=d_flag_mod_u[i+myoffset+j];
                    iam = grid->iam;
                    // mycol = MYCOL(iam, grid);
                    myrow = MYROW(iam, grid);
                    k = myrow + lk * grid->nprow; // global block row
                    myroot=URtree_ptr[lk].myRoot_;
                    myrank=URtree_ptr[lk].myRank_;
                    //printf("-- Threadloop, (%d,%d) i=%d, recv_num=%d,cur_send_num=%d, lk=%d, size=%d\n",mype,tid,i, recv_num,cur_send_num,lk,my_flag_rd[RDMA_FLAG_SIZE*k+1]);
                    zC_RdTree_forwardMessageThread_Device(&URtree_ptr[lk], (uint64_t*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &zready_lsum[0],maxrecvsz,myroot);
                    //printf("Ts Done,(%d,%d) loop=%d (%d,%d) recv_num=%d,cur_send_num=%d, k=%d, to %d\n",mype, tid,i, j, mynum,recv_num,cur_send_num,lk,myroot);
                }
            }

            __syncthreads();

            //if (tid==0) printf("iam=%d,tid=%d,i=%d, recv_num=%d,cur_send_num=%d\n",mype,tid,i,recv_num,cur_send_num);
        }
    }
#endif
#endif
}


#ifdef HAVE_CUDA
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
#endif

/************************************************************************/
/*! \brief
*
* <pre>
* Purpose
* =======
*   Perform local block modifications: lsum[i] -= L_i,k * X[k] on multi-GPU.
* </pre>
*/
__global__ void zlsum_fmod_inv_gpu_mrhs_nvshmem
/************************************************************************/
        (
                int nbcol_loc,
                int nblock_ex,
                doublecomplex *lsum,    /* Sum of local modifications.                        */
                doublecomplex *x,       /* X array (local)                                    */
                int   nrhs,      /* Number of right-hand sides.                        */
                int   maxsup,      /* Max supernode size.                        */
                int_t   nsupers,      /* Number of total supernodes.                        */
                int *fmod,     /* Modification count for L-solve.                    */
                C_Tree  *LBtree_ptr,
                C_Tree  *LRtree_ptr,
                int_t *ilsum,
                int_t *Lrowind_bc_dat,
                long int *Lrowind_bc_offset,
                doublecomplex *Lnzval_bc_dat,
                long int *Lnzval_bc_offset,
                doublecomplex *Linv_bc_dat,
                long int *Linv_bc_offset,
                int_t *Lindval_loc_bc_dat,
                long int *Lindval_loc_bc_offset,
                int_t *xsup,
                gridinfo_t *grid,
                int_t maxrecvsz,
                int mype,
                volatile uint64_t* flag_bc_q,
                volatile uint64_t* flag_rd_q,
                doublecomplex* zready_x,
                doublecomplex* zready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
                int* d_nfrecv,
                volatile int* d_status,
                volatile int* d_statusmod,
                int* d_flag_mod
        )
{
    doublecomplex zero = {0.0, 0.0};
    doublecomplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    doublecomplex *lusup;
    doublecomplex *Linv;/* Inverse of diagonal block */
    int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nsupr, m;
    int_t  k,i, l,ii, ik, il, irow, j, lb, lk, rel, lib;
    int_t  *lsub, *lloc;
    int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v, fmod_tmp;
    //__shared__ doublecomplex rtemp_loc[128];
    doublecomplex temp1;
    int_t lptr;      /* Starting position in lsub[*].                      */
    int aln_i;
    // aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int   knsupc;    /* Size of supernode k.                               */
    int_t nlb;       /* Number of L blocks.                                */

    int bid=blockIdx_x;
    int_t tmp;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
// int_t lock = 0;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
    doublecomplex rC[THR_N][THR_M];
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension
    int_t ni,mi;
    int cnt;

    if (Lrowind_bc_offset[bid] == -1) {
        return;
    }

    int gc;

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
	// idx_n = 1;
        idx_i = nlb + 2;
        idx_v = 2 * nlb + 3;
        // luptr_tmp = lloc[idx_v];
        m = nsupr - knsupc;
    } else {
        nlb = lsub[0];
        // idx_n = 0;
        idx_i = nlb;
        idx_v = 2 * nlb;
        // luptr_tmp = lloc[idx_v];
        m = nsupr;
    }

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

        RHS_ITERATE(j)
            for (i = tid; i < knsupc; i += block_size) {
                //z_atomicAdd(&zready_x[0],lsum[i + il + j * knsupc]);
                z_add(&x[i + ii + j*knsupc],
                    &x[i + ii + j*knsupc],
                    &lsum[i + il + j*knsupc]);

            }
        __syncthreads();
        //if(tid==0) printf("(%d,%d,%d),CHECKING k=%d,gc=%d,checksum=%lf\n",mype,bid,tid,k,gc,zready_x[0]);
        //if(tid==0) printf("(%d) iam bid=%d,enter solve--3,gc=%d\n",mype,bid,gc);

        //  if(Llu->inv == 1){

        Linv = &Linv_bc_dat[Linv_bc_offset[lk]];

        if (nrhs == 1) {

            for (i = tid; i < knsupc; i += block_size) {
                temp1 = zero;
                for (l = 0; l < knsupc; l++) {
                    doublecomplex ctemp;
                    zz_mult(&ctemp, &Linv[l * knsupc + i], &x[ii + l]);
                    z_add(&temp1, &temp1, &ctemp);

                }
                lsum[il + i] = temp1; //reuse lsum as temporary output as it's no longer accessed
            }
            __syncthreads();

            for (i = tid; i < knsupc; i += block_size) {
                x[i + ii] = lsum[il + i];
                //printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
            }
            __syncthreads();
        } else {
            __syncthreads();
            for (int blx = 0; blx * BLK_M < knsupc; blx++) {
                for (int bly = 0; bly * BLK_N < nrhs; bly++) {
                    gemm_device_zlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
                                           Linv, knsupc, &x[ii], knsupc, rC,
                                           alpha, beta);
#pragma unroll
                    for (ni = 0; ni < THR_N; ni++) {
                        int coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                        for (mi = 0; mi < THR_M; mi++) {
                            int coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                            if (coord_dCm < knsupc && coord_dCn < nrhs) {
                                doublecomplex &regC = rC[ni][mi];
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
                zready_x[i + maxrecvsz * lk + j * knsupc] = x[i + ii + j * knsupc];

        __syncthreads();
    } else {   /* off-diagonal block forward the message*/
        /* waiting for the x subvector and forward*/
        //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into zready_x[maxrecvsz*lk]
        volatile uint64_t msg_recv = 0;
        if (tid == 0) {
            //printf("in solve WAIT1 (%d,%d) wait for col %d,flag=%d\n", mype, bid, gc,flag_bc_q[gc]);
            //nvshmem_signal_wait_until((int *) flag_bc_q + gc, NVSHMEM_CMP_EQ, 1);
            do {
                msg_recv = flag_bc_q[lk];
                //msg_recv=d_status[gc];
                //msg_recv=flag_bc_q[gc];
                __threadfence();
            } while (msg_recv != 1);
            //printf("(%d,%d,%d,%d) in compute kernel, I have msg=%d,sz=%d,ofset=%d\n",mype,bid,tid,gc,msg_recv,LBtree_ptr[lk].msgSize_*nrhs+XK_H,maxrecvsz*lk);
            //double sum=0;
            //for (int myi=0;myi<LBtree_ptr[lk].msgSize_*nrhs+XK_H;myi++){
            //    sum+=zready_x[maxrecvsz*lk+myi];
            //}
            //printf("(%d,%d,%d), gc=%d,lk=%d, sum=%lf\n",mype,bid,tid,gc,lk,sum);
        }
        __syncthreads();
    }
    __syncthreads();

//YL: only the first thread in a block forwards the x subvector using NVSHMEM
    cnt = LBtree_ptr[lk].destCnt_;
    if (cnt > 0) {
        //cnt=LBtree_ptr[lk].msgSize_;
        my_flag_bc[gc * RDMA_FLAG_SIZE] = lk;
        my_flag_bc[gc * RDMA_FLAG_SIZE + 1] = LBtree_ptr[lk].msgSize_ * nrhs + XK_H;
        zC_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk], flag_bc_q, &my_flag_bc[gc * RDMA_FLAG_SIZE],
                                             mype, tid, &zready_x[0], maxrecvsz);
        //printf("(%d,%d,%d), lk=%d, gc=%d\n",mype,bid,tid,lk,gc);
        //zC_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&zready_x[maxrecvsz*lk],cnt*nrhs+XK_H);
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
                        doublecomplex ctemp;
	                    zz_mult(&ctemp, &lusup[luptr_tmp1 + l * nsupr + i], &zready_x[l + maxrecvsz * keep_lk + j * knsupc]);
	                    z_add(&temp1, &temp1, &ctemp);
                        //temp1+= lusup[luptr_tmp1+l*nsupr+i]*x[ii+j*knsupc+l];
                    }
                    doublecomplex tempm;
                    tempm.r=-temp1.r;
                    tempm.i=-temp1.i;
                    z_atomicAdd(&lsum[il + irow + j * iknsupc], tempm);
                    //printf("(%d,%d,%d),lsum[%d]=%f\n",mype,bid,tid,il+irow + j*iknsupc,lsum[il+irow + j*iknsupc]);
                }

                //  irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                //  if(i==nbrow+lsub[lptr1_tmp+1]-1){
                //   fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
                //   // __threadfence();
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
            // gr=myrow + lk * grid->nprow;
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
                    // __threadfence();
                    if(fmod_tmp==1) {// forward RD
                        //senddone[lk]=1;
                        if(LRtree_ptr[lk].myRoot_ != LRtree_ptr[lk].myRank_){
                            //cnt=LRtree_ptr[lib].msgSize_;

                            my_flag_rd[ik*RDMA_FLAG_SIZE]=lk;
                            my_flag_rd[ik*RDMA_FLAG_SIZE+1]=LRtree_ptr[lk].msgSize_;
                            //double tmp_sum=0;
                            RHS_ITERATE(j) {
                                for (int aab = 0; aab < iknsupc; aab++) {
                                    zready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc] = lsum[il + aab +j * iknsupc];
                                    //tmp_sum += zready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc];
                                    //printf("data3-(%d,%d,%d),lib=%d,k=%d,%d,i=%d,sum=%lf,zready_lsum[%d]=%lf, size=%d\n", mype, bid, tid, lk, gr,ik, i, tmp_sum,
                                    //       lk * maxrecvsz * 2 + aab +j * iknsupc,
                                    //       zready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc],my_flag_rd[ik*RDMA_FLAG_SIZE+1]);

                                }
                            }
                            // int temp_mysendcout=atomicAdd(&d_flag_mod[0], 1);
                            // int temp_flag_mod=atomicExch(&d_flag_mod[temp_mysendcout+1],lk);
                            //printf("iam=%d in solve,lib=%d,%d,%d, "
                            //       "pos=%d, temp %d,%d, "
                            //       "maxrecvsz=%d\n",mype,lk,k, d_flag_mod[temp_mysendcout+1],
                            //       temp_mysendcout+1,
                            //       temp_mysendcout,temp_flag_mod,
                            //       maxrecvsz);
                            //printf("(%d,%d,%d) in solve,lib=%d,gr=%d,ik=%d,myflagrd=%d,%d\n",mype,bid,tid,lk,gr,ik,my_flag_rd[ik*RDMA_FLAG_SIZE],my_flag_rd[ik*RDMA_FLAG_SIZE+1]);
                            zC_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lk], flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*ik], mype, bid, tid, &zready_lsum[0],maxrecvsz,LRtree_ptr[lk].myRoot_);
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
                for (int blx = 0; blx * BLK_M < nbrow1; blx++) {
                    for (int bly = 0; bly * BLK_N < nrhs; bly++) {
                        gemm_device_zlsum_fmod(nbrow1, nrhs, knsupc, blx, bly,
                                               &lusup[luptr_tmp1], nsupr, &zready_x[maxrecvsz * keep_lk], knsupc, rC,
                                               alpha, beta);
#pragma unroll
                        for (ni = 0; ni < THR_N; ni++) {
                            int coord_dCn = bly * BLK_N + ni * DIM_Y + idy;
#pragma unroll
                            for (mi = 0; mi < THR_M; mi++) {
                                int coord_dCm = blx * BLK_M + mi * DIM_X + idx;
                                if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                    irow = lsub[lptr + coord_dCm] - rel; /* Relative row. */
                                    doublecomplex &regC = rC[ni][mi];
                                    doublecomplex tempm;
                                    tempm.r=-regC.r;
                                    tempm.i=-regC.i;
                                    z_atomicAdd(&lsum[il + irow + coord_dCn * iknsupc], tempm);


                                }
                            }
                        }
                    }
                }
                if (tid == 0) fmod_tmp = atomicSub(&fmod[lk * aln_i], 1);


            }

        }//if(nrhs==1)
    } /* if nlb>0*/

} /* zlsum_fmod_inv_gpu_mrhs_nvshmem */

__global__ void zlsum_fmod_inv_gpu_mrhs
/************************************************************************/
(
 int nbcol_loc,
 int nblock_ex,
 doublecomplex *lsum,    /* Sum of local modifications.                        */
 doublecomplex *x,       /* X array (local)                                    */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   maxsup,      /* Max supernode size.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int *fmod,     /* Modification count for L-solve.                    */
 C_Tree  *LBtree_ptr,
 C_Tree  *LRtree_ptr,
 int_t *ilsum,
 int_t *Lrowind_bc_dat,
 long int *Lrowind_bc_offset,
 doublecomplex *Lnzval_bc_dat,
 long int *Lnzval_bc_offset,
 doublecomplex *Linv_bc_dat,
 long int *Linv_bc_offset,
 int_t *Lindval_loc_bc_dat,
 long int *Lindval_loc_bc_offset,
 int_t *xsup,
 int *bcols_masked,
 gridinfo_t *grid
)
{
    doublecomplex zero = {0.0, 0.0};
    doublecomplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    doublecomplex *lusup;
    doublecomplex *Linv;/* Inverse of diagonal block */
    int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nsupr,m;
    int_t  k,i, l,ii,ik, il, irow, j, lb, lk, rel, lib;
    int_t  *lsub, *lloc;
    int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;
    // int fmod_tmp;
   //  MPI_Status status;
   //  const int Nbk=1;
   //  __shared__ doublecomplex rtemp_loc[128];
    doublecomplex temp1;
    int_t lptr;      /* Starting position in lsub[*].                      */
   //  int iword = sizeof(int_t);
   //  int dword = sizeof (doublecomplex);
    int aln_i;
   //  aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int   knsupc;    /* Size of supernode k.                               */
    int nlb;       /* Number of L blocks.                                */

    int bid;
    int_t tmp;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
   //  int ready = 0;
    // int lock = 0;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/


    doublecomplex rC[THR_N][THR_M];

    bid= blockIdx_x;
    int idx = threadIdx_x;  // thread's m dimension
    int idy = threadIdx_y;  // thread's n dimension
    int ni,mi;


    // printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


    // rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


    // the first nbcol_loc handles all computations and broadcast communication
    if(bid<nbcol_loc){

        lk=bcols_masked[bid];

        if(Lrowind_bc_offset[lk]==-1){
        return;
        }

        iam = grid->iam;
        mycol = MYCOL( iam, grid );
        myrow = MYROW( iam, grid );
        k = mycol+lk*grid->npcol;
        knsupc = SuperSize( k );
        lsub = &Lrowind_bc_dat[Lrowind_bc_offset[lk]];
        iam = grid->iam;
        krow = PROW( k, grid );
        lusup = &Lnzval_bc_dat[Lnzval_bc_offset[lk]];
        lloc = &Lindval_loc_bc_dat[Lindval_loc_bc_offset[lk]];
        nsupr = lsub[1];

        if(myrow==krow){
            nlb = lsub[0] - 1;
           //  idx_n = 1;
            idx_i = nlb+2;
            idx_v = 2*nlb+3;
           //  luptr_tmp = lloc[idx_v];
            m = nsupr-knsupc;
        }else{
            nlb = lsub[0];
           //  idx_n = 0;
            idx_i = nlb;
            idx_v = 2*nlb;
           //  luptr_tmp = lloc[idx_v];
            m = nsupr;
        }

        // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

        if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

            if(tid==0){  /*only the first thread in a block handles the lock */

            // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            // for (i=0 ; i<maxsup ; i++){
                // rtemp_loc[i]=0.0;
            // }

                lib = LBi( k, grid ); /* Local block number, row-wise. */
                do{
                    tmp=fmod[lib*aln_i];
                    __threadfence();
                }while(tmp>0);

            }
            __syncthreads();


                lib = LBi( k, grid ); /* Local block number, row-wise. */
                il = LSUM_BLK( lib );
                ii = X_BLK( lib );

                RHS_ITERATE(j)
                    for (i = tid; i < knsupc; i+=block_size)
                        z_add(&x[i + ii + j*knsupc],
                            &x[i + ii + j*knsupc],
                            &lsum[i + il + j*knsupc]);
                __syncthreads();


               //  if(Llu->inv == 1){

                    Linv = &Linv_bc_dat[Linv_bc_offset[lk]];

                    if(nrhs==1){

                        for (i = tid; i < knsupc; i+=block_size){
                            temp1=zero;
                            for (l=0 ; l<knsupc ; l++){
                                doublecomplex ctemp;
                                zz_mult(&ctemp, &Linv[l*knsupc+i], &x[ii+l]);
                                z_add(&temp1, &temp1, &ctemp);

                            }
                            lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
                        }
                        __syncthreads();

                        for (i = tid; i < knsupc; i+=block_size){
                            x[i + ii] = lsum[il+i];
                            // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
                            }
                        __syncthreads();



                        // RHS_ITERATE(j){

                        // for (i = tid; i < knsupc; i+=block_size)
                            // rtemp_loc[i]=zero;
                        // __syncthreads();


                        // zgemv_device_dlsum_fmod(
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

                    }else{
                        __syncthreads();
                        for (int blx = 0; blx*BLK_M < knsupc; blx++){
                            for (int bly = 0; bly*BLK_N < nrhs; bly++){
                                gemm_device_zlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
                                Linv, knsupc, &x[ii], knsupc, rC,
                                alpha, beta);
                                    #pragma unroll
                                for (ni = 0; ni < THR_N; ni++) {
                                    int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
                                    #pragma unroll
                                    for (mi = 0; mi < THR_M; mi++) {
                                        int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                                        if (coord_dCm < knsupc && coord_dCn < nrhs) {
                                            doublecomplex &regC = rC[ni][mi];
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
            __syncthreads();
        }else{   /* off-diagonal block forward the message*/
            /* waiting for the x subvector and forward*/
            if(tid==0){  //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into recvbuf_BC_gpu[maxrecvsz*lk]

            }
        }

        if(nlb>0){

                lib = LBi( k, grid ); /* Local block number, row-wise. */
                ii = X_BLK( lib );

                if(nrhs==1){
                    luptr_tmp1 = lloc[idx_v];
                    lb = 0;
                    nbrow=0;
                    lptr1_tmp = lloc[lb+idx_i];
                    lptr= lptr1_tmp+2;
                    nbrow1 = lsub[lptr1_tmp+1];
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi( ik, grid ); /* Local block number, row-wise. */
                    iknsupc = SuperSize( ik );
                    il = LSUM_BLK( lk );

                    for (i = tid; i < m; i+=block_size){
                        while(nbrow+lsub[lptr1_tmp+1]<=i){
                            lb++;
                            nbrow +=lsub[lptr1_tmp+1];
                            lptr1_tmp = lloc[lb+idx_i];
                            lptr= lptr1_tmp+2;
                            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                            rel = xsup[ik]; /* Global row index of block ik. */
                            lk = LBi( ik, grid ); /* Local block number, row-wise. */
                            iknsupc = SuperSize( ik );
                            il = LSUM_BLK( lk );
                        }

                        irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                        RHS_ITERATE(j){
                        temp1=zero;
                        for (l=0 ; l<knsupc ; l++){
                            doublecomplex ctemp;
                            zz_mult(&ctemp, &lusup[luptr_tmp1+l*nsupr+i], &x[ii+j*knsupc+l]);
                            z_add(&temp1, &temp1, &ctemp);
                        }
                    doublecomplex tempm;
                    tempm.r=-temp1.r;
                    tempm.i=-temp1.i;
                    z_atomicAdd(&lsum[il + irow + j * iknsupc], tempm);

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
                    nbrow=0;
                    lptr1_tmp = lloc[lb+idx_i];
                    lptr= lptr1_tmp+2;
                    nbrow1 = lsub[lptr1_tmp+1];
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi( ik, grid ); /* Local block number, row-wise. */
                    iknsupc = SuperSize( ik );
                    il = LSUM_BLK( lk );

                    for (i = tid; i < m; i+=block_size){
                       while(nbrow+lsub[lptr1_tmp+1]<=i){
                           lb++;
                           nbrow +=lsub[lptr1_tmp+1];
                           lptr1_tmp = lloc[lb+idx_i];
                           lptr= lptr1_tmp+2;
                           ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                           rel = xsup[ik]; /* Global row index of block ik. */
                           lk = LBi( ik, grid ); /* Local block number, row-wise. */
                           iknsupc = SuperSize( ik );
                           il = LSUM_BLK( lk );
                       }

                       irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                       if(i==nbrow+lsub[lptr1_tmp+1]-1){
                           atomicSub(&fmod[lk*aln_i],1);
                           // __threadfence();
                       }
                   }
                   __syncthreads();


                }else {
                    for (lb = 0; lb < nlb; lb++){
                        luptr_tmp1 = lloc[lb+idx_v];

                        // nbrow=0;
                        // lptr1_tmp = lloc[lb+idx_i];
                        // nbrow += lsub[lptr1_tmp+1];


                        lib = LBi( k, grid ); /* Local block number, row-wise. */
                        ii = X_BLK( lib );

                        lptr1_tmp = lloc[lb+idx_i];
                        lptr= lptr1_tmp+2;
                        nbrow1 = lsub[lptr1_tmp+1];
                        ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        rel = xsup[ik]; /* Global row index of block ik. */

                        lk = LBi( ik, grid ); /* Local block number, row-wise. */

                        iknsupc = SuperSize( ik );
                        il = LSUM_BLK( lk );


                        // if(nrhs==1){

                            // for (i = tid; i < nbrow1; i+=block_size)
                                // rtemp_loc[i]=zero;
                            // __syncthreads();


                            // zgemv_device_dlsum_fmod(
                                // nbrow1, knsupc, alpha,
                                // &lusup[luptr_tmp1], nsupr,
                                // &x[ii], 1, beta,
                                // rtemp_loc, 1);

                            // __syncthreads();
                            // for (i = tid; i < nbrow1; i+=block_size){
                                // irow = lsub[lptr+i] - rel; /* Relative row. */
                                // doublecomplex tempm;
                                // tempm.r=-rtemp_loc[i].r;
                                // tempm.i=-rtemp_loc[i].i;
                                // temp = z_atomicAdd(&lsum[il+irow], tempm);

                                // }
                        // }else{

                            for (int blx = 0; blx*BLK_M < nbrow1; blx++){
                                for (int bly = 0; bly*BLK_N < nrhs; bly++){
                                    gemm_device_zlsum_fmod(nbrow1, nrhs, knsupc, blx, bly,
                                    &lusup[luptr_tmp1], nsupr, &x[ii], knsupc, rC,
                                    alpha, beta);
                                        #pragma unroll
                                    for (ni = 0; ni < THR_N; ni++) {
                                        int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
                                        #pragma unroll
                                        for (mi = 0; mi < THR_M; mi++) {
                                            int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                                            if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                                irow = lsub[lptr+coord_dCm] - rel; /* Relative row. */
                                                doublecomplex &regC = rC[ni][mi];
                                                doublecomplex tempm;
                                                tempm.r=-regC.r;
                                                tempm.i=-regC.i;
                                                z_atomicAdd(&lsum[il + irow + coord_dCn * iknsupc], tempm);
                                            }
                                        }
                                    }
                                }
                            }
                        // }//if(nrhs==1)

                        if(tid==0)atomicSub(&fmod[lk*aln_i],1);



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
                __syncthreads();
            // } /*if tid<Nchunk*/
        } /* if nlb>0*/

        // printf("nimbgood \n");

}

} /* zlsum_fmod_inv_gpu_mrhs */


__global__ void zlsum_fmod_inv_gpu_1rhs_warp
/************************************************************************/
(
 int nbcol_loc,
 int nblock_ex,
 doublecomplex *lsum,    /* Sum of local modifications.                        */
 doublecomplex *x,       /* X array (local)                                    */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   maxsup,      /* Max supernode size.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int *fmod,     /* Modification count for L-solve.                    */
 C_Tree  *LBtree_ptr,
 C_Tree  *LRtree_ptr,
 int_t *ilsum,
 int_t *Lrowind_bc_dat,
 long int *Lrowind_bc_offset,
 doublecomplex *Lnzval_bc_dat,
 long int *Lnzval_bc_offset,
 doublecomplex *Linv_bc_dat,
 long int *Linv_bc_offset,
 int_t *Lindval_loc_bc_dat,
 long int *Lindval_loc_bc_offset,
 int_t *xsup,
 int *bcols_masked,
 gridinfo_t *grid
)
{
    doublecomplex zero = {0.0, 0.0};
    doublecomplex *lusup;
    doublecomplex *Linv;/* Inverse of diagonal block */
    int    iam, iknsupc, myrow, mycol, krow, nbrow, nsupr,m;
    int_t  k,i, l,ii,ik, il, irow, j, lb, lk, rel, lib;
    int_t  *lsub, *lloc;
    int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;
    // int fmod_tmp;
   //  MPI_Status status;
   //  const int Nbk=1;
   //  __shared__ doublecomplex rtemp_loc[128];
    doublecomplex temp1;
    int_t lptr;      /* Starting position in lsub[*].                      */
   //  int iword = sizeof(int_t);
   //  int dword = sizeof (doublecomplex);
    int aln_i;
   //  aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int   knsupc;    /* Size of supernode k.                               */
    int nlb;       /* Number of L blocks.                                */

    // int bid;
    int_t tmp;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
   //  int ready = 0;
    // int lock = 0;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/


    int wrp;
    int lne;
    wrp= tid + blockIdx_x*block_size;
    lne=wrp%WARP_SIZE;
	// printf("  Entering kernel:   %i %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,wrp,wrp/WARP_SIZE,tid);
	wrp/=WARP_SIZE;


    // int wrp;
    // int lne = threadIdx_x & 0x1f ;
    // // int ready = 0;
    // // int lock = 0;
    // wrp= threadIdx_x + blockIdx_x * blockDim_x;
    // wrp/=WARP_SIZE;

    // bid= blockIdx_x;

    // printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


    // rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


    // the first nbcol_loc handles all computations and broadcast communication
    if(wrp<nbcol_loc){

        lk=bcols_masked[wrp];

        if(Lrowind_bc_offset[lk]==-1){
        return;
        }

        iam = grid->iam;
        mycol = MYCOL( iam, grid );
        myrow = MYROW( iam, grid );
        k = mycol+lk*grid->npcol;
        knsupc = SuperSize( k );
        lsub = &Lrowind_bc_dat[Lrowind_bc_offset[lk]];
        iam = grid->iam;
        krow = PROW( k, grid );
        lusup = &Lnzval_bc_dat[Lnzval_bc_offset[lk]];
        lloc = &Lindval_loc_bc_dat[Lindval_loc_bc_offset[lk]];
        nsupr = lsub[1];

        if(myrow==krow){
            nlb = lsub[0] - 1;
           //  idx_n = 1;
            idx_i = nlb+2;
            idx_v = 2*nlb+3;
           //  luptr_tmp = lloc[idx_v];
            m = nsupr-knsupc;
        }else{
            nlb = lsub[0];
           //  idx_n = 0;
            idx_i = nlb;
            idx_v = 2*nlb;
           //  luptr_tmp = lloc[idx_v];
            m = nsupr;
        }

        // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

        if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

            if(lne==0){  /*only the first thread in a warp handles the lock */

            // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            // for (i=0 ; i<maxsup ; i++){
                // rtemp_loc[i]=0.0;
            // }

                lib = LBi( k, grid ); /* Local block number, row-wise. */
                do{
                    tmp=fmod[lib*aln_i];
                    __threadfence();
                }while(tmp>0);

            }
            __syncwarp();


                lib = LBi( k, grid ); /* Local block number, row-wise. */
                il = LSUM_BLK( lib );
                ii = X_BLK( lib );

                for (i = lne; i < knsupc; i+=WARP_SIZE)
                    z_add(&x[i + ii ],
                    &x[i + ii ],
                    &lsum[i + il ]);

                // __syncwarp();


               //  if(Llu->inv == 1){

                    Linv = &Linv_bc_dat[Linv_bc_offset[lk]];


                        for (i = lne; i < knsupc; i+=WARP_SIZE){
                            temp1=zero;
                            for (l=0 ; l<knsupc ; l++){
                                doublecomplex ctemp;
                                zz_mult(&ctemp, &Linv[l*knsupc+i], &x[ii+l]);
                                z_add(&temp1, &temp1, &ctemp);
                            }
                            lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
                        }
                        // __syncwarp();

                        for (i = lne; i < knsupc; i+=WARP_SIZE){
                            x[i + ii] = lsum[il+i];
                            // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
                            }
                        // __syncwarp();


               //  }
            // __syncwarp();
        }else{   /* off-diagonal block forward the message*/
        }

        if(nlb>0){

                lib = LBi( k, grid ); /* Local block number, row-wise. */
                ii = X_BLK( lib );

                if(nrhs==1){
                    luptr_tmp1 = lloc[idx_v];
                    lb = 0;
                    nbrow=0;
                    lptr1_tmp = lloc[lb+idx_i];
                    lptr= lptr1_tmp+2;
                    // nbrow1 = lsub[lptr1_tmp+1];
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi( ik, grid ); /* Local block number, row-wise. */
                    iknsupc = SuperSize( ik );
                    il = LSUM_BLK( lk );

                    for (i = lne; i < m; i+=WARP_SIZE){
                        while(nbrow+lsub[lptr1_tmp+1]<=i){
                            lb++;
                            nbrow +=lsub[lptr1_tmp+1];
                            lptr1_tmp = lloc[lb+idx_i];
                            lptr= lptr1_tmp+2;
                            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                            rel = xsup[ik]; /* Global row index of block ik. */
                            lk = LBi( ik, grid ); /* Local block number, row-wise. */
                            iknsupc = SuperSize( ik );
                            il = LSUM_BLK( lk );
                        }

                        irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                        RHS_ITERATE(j){
                        temp1=zero;
                        for (l=0 ; l<knsupc ; l++){
                            doublecomplex ctemp;
                            zz_mult(&ctemp, &lusup[luptr_tmp1+l*nsupr+i], &x[ii+j*knsupc+l]);
                            z_add(&temp1, &temp1, &ctemp);
                        }
                        doublecomplex tempm;
                        tempm.r=-temp1.r;
                        tempm.i=-temp1.i;
                        z_atomicAdd(&lsum[il+irow + j*iknsupc], tempm);
                        }

                        if(i==nbrow+lsub[lptr1_tmp+1]-1){
                       	 atomicSub(&fmod[lk*aln_i],1);
                       	 // __threadfence();
                        }



}
                    // __syncwarp();

                //     luptr_tmp1 = lloc[idx_v];
                //     lb = 0;
                //     nbrow=0;
                //     lptr1_tmp = lloc[lb+idx_i];
                //     lptr= lptr1_tmp+2;
                //     nbrow1 = lsub[lptr1_tmp+1];
                //     ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                //     rel = xsup[ik]; /* Global row index of block ik. */
                //     lk = LBi( ik, grid ); /* Local block number, row-wise. */
                //     iknsupc = SuperSize( ik );
                //     il = LSUM_BLK( lk );

                //     for (i = lne; i < m; i+=WARP_SIZE){
                //        while(nbrow+lsub[lptr1_tmp+1]<=i){
                //            lb++;
                //            nbrow +=lsub[lptr1_tmp+1];
                //            lptr1_tmp = lloc[lb+idx_i];
                //            lptr= lptr1_tmp+2;
                //            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                //            rel = xsup[ik]; /* Global row index of block ik. */
                //            lk = LBi( ik, grid ); /* Local block number, row-wise. */
                //            iknsupc = SuperSize( ik );
                //            il = LSUM_BLK( lk );
                //        }

                //        irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                //        if(i==nbrow+lsub[lptr1_tmp+1]-1){
                //            fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
                //            // __threadfence();
                //        }
                //    }
                //    __syncwarp();


                }//if(nrhs==1)


                // __syncwarp();
            // } /*if tid<Nchunk*/
        } /* if nlb>0*/

        // printf("nimbgood \n");

}

} /* zlsum_fmod_inv_gpu_1rhs_warp */



void zlsum_fmod_inv_gpu_wrap
        (
                int nbcol_loc,    /*number of local supernode columns*/
                int nbrow_loc,    /*number of local supernode rows*/
                int nthread_x,     /*kernel launch parameter*/
                int nthread_y,     /*kernel launch parameter*/
                doublecomplex *lsum,    /* Sum of local modifications.                        */
                doublecomplex *x,       /* X array (local)                                    */
                int   nrhs,      /* Number of right-hand sides.                        */
                int   maxsup,      /* Max supernode size.                        */
                int_t   nsupers,      /* Number of total supernodes.                        */
                int *fmod,     /* Modification count for L-solve.                    */
                C_Tree  *LBtree_ptr,
                C_Tree  *LRtree_ptr,
                int_t *ilsum,
                int_t *Lrowind_bc_dat,
                long int *Lrowind_bc_offset,
                doublecomplex *Lnzval_bc_dat,
                long int *Lnzval_bc_offset,
                doublecomplex *Linv_bc_dat,
                long int *Linv_bc_offset,
                int_t *Lindval_loc_bc_dat,
                long int *Lindval_loc_bc_offset,
                int_t *xsup,
                int * bcols_masked,
                gridinfo_t *grid,
                int_t maxrecvsz,
                uint64_t* flag_bc_q,
                uint64_t* flag_rd_q,
                doublecomplex* zready_x,
                doublecomplex* zready_lsum,
                int* my_flag_bc,
                int* my_flag_rd,
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
                int* d_flag_mod,
                int procs
        ) {

    int nblock_ex = CEILING(nbrow_loc, ((nthread_x * nthread_y) / 32)); //32 (warp) * 8 =256

    int mype;

    if(procs==1){
        nblock_ex=0;
    #ifdef SINGLE_RHS_OPT
        if(nrhs>1){
    #else
        if(1){
    #endif
            dim3 dimBlock(nthread_x, nthread_y);
            zlsum_fmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc,nblock_ex,lsum,x,nrhs,maxsup,nsupers,fmod,LBtree_ptr,LRtree_ptr,ilsum,Lrowind_bc_dat,Lrowind_bc_offset,Lnzval_bc_dat,Lnzval_bc_offset,Linv_bc_dat,Linv_bc_offset,Lindval_loc_bc_dat,Lindval_loc_bc_offset, xsup,bcols_masked, grid);
        }else{
            dim3 dimBlock(nthread_x, nthread_y,1);
            zlsum_fmod_inv_gpu_1rhs_warp<<< CEILING(nbcol_loc,NWARP), dimBlock >>>(nbcol_loc,nblock_ex,lsum,x,nrhs,maxsup,nsupers,fmod,LBtree_ptr,LRtree_ptr,ilsum,Lrowind_bc_dat,Lrowind_bc_offset,Lnzval_bc_dat,Lnzval_bc_offset,Linv_bc_dat,Linv_bc_offset,Lindval_loc_bc_dat,Lindval_loc_bc_offset, xsup,bcols_masked, grid);
        }
        checkGPU(gpuGetLastError());
     }else{

#ifdef HAVE_NVSHMEM
        mype = nvshmem_my_pe();
        // npes = nvshmem_n_pes();


        cudaStream_t stream[2];
        for (int i = 0; i < 2; ++i) {
            //cudaStreamCreate(&stream[i]);
            cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        }

        cudaFuncAttributes cuattr;
        cudaFuncGetAttributes(&cuattr, zlsum_fmod_inv_gpu_mrhs_nvshmem);
        cudaDeviceSetLimit(cudaLimitStackSize, cuattr.localSizeBytes);

        int minGridSize, myblockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize,&myblockSize,(const void *) zwait_bcrd ,0,0 );
        if (myblockSize < h_nfrecv[1]) {
            h_nfrecv[1] = myblockSize;
            gpuMemcpy(d_nfrecv, h_nfrecv, 3 * sizeof(int), gpuMemcpyHostToDevice);
        }
        //printf("(%d) solve=%d,%d, minGridSize=%d,myblockSize%d, nvshmem_kernel=%d,%d\n",
        //       mype,nbcol_loc,nthread_x*nthread_y,
        //       minGridSize,myblockSize,h_nfrecv[2],h_nfrecv[1]);
        //fflush(stdout);


        dim3 dimGrid_bc(h_nfrecv[2]); //3
        dim3 dimBlock_bc(h_nfrecv[1]); //256 by default
        dim3 dimGrid(nbcol_loc);
        dim3 dimBlock(nthread_x, nthread_y);

    //if (npes==1){
    //    zlsum_fmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc,nblock_ex,lsum,x,nrhs,maxsup,nsupers,fmod,LBtree_ptr,LRtree_ptr,ilsum,Lrowind_bc_dat,Lrowind_bc_offset,Lnzval_bc_dat,Lnzval_bc_offset,Linv_bc_dat,Linv_bc_offset,Lindval_loc_bc_dat,Lindval_loc_bc_offset, xsup,grid,maxrecvsz);
    //}else{

        void *args[] = {&nrhs, &LRtree_ptr, &maxrecvsz, &mype, &flag_bc_q, &flag_rd_q,
                        &zready_x, &zready_lsum, &my_flag_bc, &my_flag_rd, &d_nfrecv, &d_status,
                        &d_colnum, &d_mynum, &d_mymaskstart, &d_mymasklength,
                        &d_nfrecvmod, &d_statusmod, &d_colnummod, &d_mynummod, &d_mymaskstartmod, &d_mymasklengthmod,
                        &d_recv_cnt, &d_msgnum, &d_flag_mod, &lsum,&fmod,&grid,&xsup,&ilsum,&nbrow_loc,&nsupers};

        int status=1;
        status = nvshmemx_collective_launch((const void *) zwait_bcrd, dimGrid_bc, dimBlock_bc, args, 0, stream[0]);
        //status1 = nvshmemx_collective_launch((const void *) send_rd, dimGrid_rd, dimBlock_bc, args, 0, stream[1]);
        //printf("(%d), status=%d\n",mype, status);
        //fflush(stdout);

        if ((status != NVSHMEMX_SUCCESS)) {
            fprintf(stderr, "shmemx_collective_launch failed %d\n", status);
            exit(-1);
        }else{
            zlsum_fmod_inv_gpu_mrhs_nvshmem<<< dimGrid, dimBlock, 0, stream[1] >>>(nbcol_loc,nblock_ex,
                                                                                lsum,x,nrhs,maxsup,nsupers,fmod,
                                                                                LBtree_ptr,LRtree_ptr,ilsum,
                                                                                Lrowind_bc_dat,Lrowind_bc_offset,
                                                                                Lnzval_bc_dat,Lnzval_bc_offset,
                                                                                Linv_bc_dat,Linv_bc_offset,
                                                                                Lindval_loc_bc_dat,Lindval_loc_bc_offset,
                                                                                xsup,grid,maxrecvsz,
                                                                                mype, flag_bc_q,
                                                                                flag_rd_q,
                                                                                zready_x, zready_lsum,
                                                                                my_flag_bc, my_flag_rd,
                                                                                d_nfrecv, d_status,
                                                                                d_statusmod,d_flag_mod);
            CUDA_CHECK(cudaGetLastError());
        } // if status
    //} // if npes==1
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
#else
    printf("NVSHMEM is needed for multi-GPU solve\n");
    exit(1);
#endif
}

//printf("(%d) Done L solve\n",mype);
//fflush(stdout);
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

__global__ void zlsum_bmod_inv_gpu_mrhs
/************************************************************************/
(
 int nbcol_loc,
 doublecomplex *lsum,    /* Sum of local modifications.                        */
 doublecomplex *x,       /* X array (local)                                    */
 int   nrhs,      /* Number of right-hand sides.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int *bmod,     /* Modification count for U-solve.                    */
 C_Tree  *UBtree_ptr,
 C_Tree  *URtree_ptr,
 int_t *ilsum,
 int_t *Ucolind_bc_dat,
 long int *Ucolind_bc_offset,
 doublecomplex *Unzval_bc_dat,
 long int *Unzval_bc_offset,
doublecomplex *Uinv_bc_dat,
long int *Uinv_bc_offset,
int_t *Uindval_loc_bc_dat,
long int *Uindval_loc_bc_offset,
int_t *xsup,
gridinfo_t *grid
)
{
    doublecomplex zero = {0.0, 0.0};
    doublecomplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
	doublecomplex *Uinv;/* Inverse of diagonal block */
	int    iam, iknsupc, myrow, mycol, krow;
	int_t  k,i, l,ii, ik, il, j, lk, lib, ub;
	int_t gik, rel, lptr, ncol, icol;
	doublecomplex temp1;
     __shared__ doublecomplex temp2[MAXSUPER];
	int aln_i;
	aln_i = 1;//ceil(CACHELINE/(double)iword);
	int   knsupc;    /* Size of supernode k.                               */
	int nub;       /* Number of L blocks.                                */

	int bid;
	int_t tmp;
	// int bmod_tmp;
	int tid = threadIdx_x + threadIdx_y * blockDim_x;
	const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
	doublecomplex rC[THR_N][THR_M];
	// __shared__ doublecomplex x_share[DIM_X*DIM_Y];

	bid= nbcol_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
	int idx = threadIdx_x;  // thread's m dimension
	int idy = threadIdx_y;  // thread's n dimension
	int ni,mi;
	int_t  *usub, *lloc;
	doublecomplex *lusup;
	int_t nrow, nnz_offset, offset;
	int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;



	// printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


	// rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


	// the first nbcol_loc handles all computations and broadcast communication
	if(bid<nbcol_loc){
		if(Uinv_bc_offset[bid]==-1 && Ucolind_bc_offset[bid]==-1){
		return;
		}

		lk=bid;
		iam = grid->iam;
		mycol = MYCOL( iam, grid );
		myrow = MYROW( iam, grid );
		k = mycol+lk*grid->npcol;
		knsupc = SuperSize( k );
		krow = PROW( k, grid );
		usub = &Ucolind_bc_dat[Ucolind_bc_offset[lk]];
		lusup = &Unzval_bc_dat[Unzval_bc_offset[lk]];
		lloc = &Uindval_loc_bc_dat[Uindval_loc_bc_offset[lk]];
		rel = xsup[k]; /* Global column index of block ik. */

	    // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

		if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

			if(tid==0){  /*only the first thread in a block handles the lock */


			// for (i=0 ; i<maxsup ; i++){
				// rtemp_loc[i]=0.0;
			// }

				lib = LBi( k, grid ); /* Local block number, row-wise. */
			    // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
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
					for (i = tid; i < knsupc; i+=block_size){
			            z_add(&x[i + ii + j*knsupc],
			            &x[i + ii + j*knsupc],
			            &lsum[i + il + j*knsupc]);
						// if(lib==1){
						// printf("lib %5d %5d %5d %lf\n",lib,i, il, lsum[i + il + j*knsupc ]);
						// // printf("lib %5d %5d %lf\n",lib,i, x[i + ii + j*knsupc]);
						// }
					}
				__syncthreads();



			   //  if(Llu->inv == 1){

					Uinv = &Uinv_bc_dat[Uinv_bc_offset[lk]];

					if(nrhs==1){
						for (i = tid; i < knsupc; i+=block_size){
							temp1=zero;
							for (l=0 ; l<knsupc ; l++){
                                doublecomplex ctemp;
                                zz_mult(&ctemp, &Uinv[l*knsupc+i], &x[ii+l]);
                                z_add(&temp1, &temp1, &ctemp);
							}
							lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
						}
						__syncthreads();

						for (i = tid; i < knsupc; i+=block_size){
							x[i + ii] = lsum[il+i];
							// // if(lk==69)
							// printf("lk %5d %5d %lf\n",lk,i, x[i + ii]);
							}
						__syncthreads();
					}else{
						__syncthreads();
						for (int blx = 0; blx*BLK_M < knsupc; blx++){
							for (int bly = 0; bly*BLK_N < nrhs; bly++){
								gemm_device_zlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
								Uinv, knsupc, &x[ii], knsupc, rC,
								alpha, beta);
									#pragma unroll
								for (ni = 0; ni < THR_N; ni++) {
									int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
									#pragma unroll
									for (mi = 0; mi < THR_M; mi++) {
										int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
										if (coord_dCm < knsupc && coord_dCn < nrhs) {
											doublecomplex &regC = rC[ni][mi];
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
	  // 	  zC_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&recvbuf_BC_gpu[maxrecvsz*lk],cnt*nrhs+XK_H);
	  //   }
	  //   }

		if(Ucolind_bc_offset[bid]!=-1){
			nub = usub[0];      /* Number of U blocks in block column lk */
		}else{
			nub = 0;
		}
		if(nub>0){
				nrow = usub[1];  // total number of nonzero rows
				nnz_offset = usub[2]; // total number of nonzero column segments

				lib = LBi( k, grid ); /* Local block number, row-wise. */
				ii = X_BLK( lib );

				if(nrhs==1){
				// if(0){
					for (i=tid;i<knsupc;i+=block_size)
						temp2[i]=x[ii+i];
					__syncthreads();
                    for (i = tid; i < nrow; i+=block_size){
                        // printf("good1 bid nub i nrow %5d %5d %5d %5d\n",bid, nub, i, nrow);
                        ub = usub[nnz_offset+i*2];
                        offset = usub[nnz_offset+i*2+1];
                        ik = lloc[ub];
                        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                        iknsupc = SuperSize( gik );
                        // // if(lk==2 && ik==1)
                        // // printf("ub offset %5d %5d %5d %5d\n",ub, i, offset,SuperSize( gik ));

                        idx_v=2*nub+ub;
                        idx_i=nub+ub;
                        luptr_tmp1 = lloc[idx_v];
                        lptr1_tmp = lloc[idx_i];
                        lptr= lptr1_tmp+2;
                        ncol = usub[lptr1_tmp+1];
                        il = LSUM_BLK( ik );

                        temp1=zero;
                        for (l=0 ; l<ncol ; l++){
                            icol = usub[lptr+l] - rel; /* Relative col. */
                            doublecomplex ctemp;
                            zz_mult(&ctemp, &lusup[luptr_tmp1+l*iknsupc+offset], &temp2[icol]);
                            z_add(&temp1, &temp1, &ctemp);

                            // // if(offset==159 && ik==1)
                            // if(lk==2 && ik==1)
                            // printf("lsum %5d %5d %5d %10f %10f %5d %5d %5d\n",l, icol, offset, x[ii+j*knsupc+icol], lusup[luptr_tmp1+l*iknsupc+offset], luptr_tmp1, ncol, iknsupc);


                            // printf("lsum %5d %5d %5d %10f %10f %10f\n",uptr-1, jj, irow - ikfrow, uval[uptr-1], xtemp, temp2[irow - ikfrow]);

                        }
                        doublecomplex tempm;
                        tempm.r=-temp1.r;
                        tempm.i=-temp1.i;
                        z_atomicAdd(&lsum[il+offset], tempm);

                    }
                    __syncthreads();

                    for (ub = tid; ub < nub; ub+=block_size){
                        ik = lloc[ub];
                        atomicSub(&bmod[ik*aln_i],1);
                        // printf("ik %5d bmod[ik*aln_i] %5d\n",ik,bmod[ik*aln_i]);
                    }
                }else{
                    for (ub = 0; ub < nub; ub++){
                        ik = lloc[ub];
                        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                        iknsupc = SuperSize( gik );
                        idx_v=2*nub+ub;
                        idx_i=nub+ub;
                        luptr_tmp1 = lloc[idx_v];
                        lptr1_tmp = lloc[idx_i];
                        lptr= lptr1_tmp+2;
                        ncol = usub[lptr1_tmp+1];
                        il = LSUM_BLK( ik );


                        for (int blx = 0; blx*BLK_M < iknsupc; blx++){
                            for (int bly = 0; bly*BLK_N < nrhs; bly++){

                                gemm_device_zlsum_bmod_stridedB(iknsupc, nrhs, ncol, blx, bly,
                                &lusup[luptr_tmp1], iknsupc, &x[ii], knsupc, rC,
                                alpha, beta, lptr, rel, usub);

                                #pragma unroll
                                for (ni = 0; ni < THR_N; ni++) {
                                    int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
                                    #pragma unroll
                                    for (mi = 0; mi < THR_M; mi++) {
                                        int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                                        if (coord_dCm < iknsupc && coord_dCn < nrhs) {
                                            doublecomplex &regC = rC[ni][mi];
                                            doublecomplex tempm;
                                            tempm.r=-regC.r;
                                            tempm.i=-regC.i;
                                            z_atomicAdd(&lsum[il+coord_dCm + coord_dCn*iknsupc], tempm);
                                        }
                                    }
                                }
                            }
                        }
                        if(tid==0)atomicSub(&bmod[ik*aln_i],1);
                    }

				}//if(nrhs==1)
                __syncthreads();
			// } /*if tid<Nchunk*/
		} /* if nlb>0*/

		// printf("nimbgood \n");

//   }else if(bid<nbcol_loc+nblock_ex){  //the next nblock_ex blocks handle all reduction communication

}



} /* zlsum_bmod_inv_gpu_mrhs */


 /************************************************************************/
 /*! \brief
  *
  * <pre>
  * Purpose
  * =======
  *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
  * </pre>
  */
  __global__ void zlsum_bmod_inv_gpu_1rhs_new
  /************************************************************************/
  (
   int nbrow_loc,
   doublecomplex *lsum,    /* Sum of local modifications.                        */
   doublecomplex *x,       /* X array (local)                                    */
   int   nrhs,      /* Number of right-hand sides.                        */
   int_t   nsupers,      /* Number of total supernodes.                        */
   int *bmod,     /* Modification count for U-solve.                    */
   C_Tree  *UBtree_ptr,
   C_Tree  *URtree_ptr,
   int_t *ilsum,
   int_t *Ucolind_bc_dat,
   long int *Ucolind_bc_offset,
   int_t *Uind_br_dat,
   long int *Uind_br_offset,
   doublecomplex *Unzval_bc_dat,
   long int *Unzval_bc_offset,
  doublecomplex *Uinv_bc_dat,
  long int *Uinv_bc_offset,
  int_t *Uindval_loc_bc_dat,
  long int *Uindval_loc_bc_offset,
  int_t *xsup,
  gridinfo_t *grid
  )
  {
    //   doublecomplex xtemp;
    //   doublecomplex *dest;
    //  doublecomplex *Uinv;/* Inverse of diagonal block */
      int    iam, iknsupc, myrow, mycol, kcol;
      int_t  k,i,i1, bb, l,ii, lk, jk, lib, ljb, ub;
      int_t gik, rel, lptr, ncol, icol;
      doublecomplex temp1;
      __shared__ doublecomplex s_lsum[MAXSUPER];
      // volatile __shared__ doublecomplex temp2[MAXSUPER];
      volatile __shared__ int s_bmod;
      int aln_i;
      aln_i = 1;//ceil(CACHELINE/(double)iword);
      int nub;       /* Number of U blocks.                                */

      int bid;
      int_t tmp;
      // int bmod_tmp;
      int tid = threadIdx_x + threadIdx_y * blockDim_x;
      const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
      doublecomplex zero = {0.0, 0.0};
    //   doublecomplex rC[THR_N][THR_M];
      // __shared__ doublecomplex x_share[DIM_X*DIM_Y];

      bid= nbrow_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
      int_t  *usub, *lloc, *uind_br;;
      doublecomplex *lusup;
      int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;

      int wrp;
      int lne;
      wrp= tid;
      lne=wrp%WARP_SIZE;
      wrp/=WARP_SIZE;
// printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,wrp,tid);



    //   printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


      // rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


      // the first nbcol_loc handles all computations and broadcast communication
      if(bid<nbrow_loc){

          lk=bid;
          iam = grid->iam;
          mycol = MYCOL( iam, grid );
          myrow = MYROW( iam, grid );
          gik = myrow+lk*grid->nprow;
          if(gik<nsupers){
          iknsupc = SuperSize( gik );
        //   il = LSUM_BLK( lk );
          kcol = PCOL( gik, grid );
          jk = LBj( gik, grid ); /* Local block number, column-wise. */

          if(Uinv_bc_offset[jk]==-1 && Uind_br_offset[lk]==-1){
          return;
          }


          // initialize the shared memory data, which requires __syncthreads
          if(tid==0)s_bmod = bmod[lk*aln_i];
          for (i = tid; i < MAXSUPER; i+=block_size){s_lsum[i]=zero;}
          __syncthreads();


          uind_br = &Uind_br_dat[Uind_br_offset[lk]];
        //   if(lne==0)
        //      printf("  Entering kernel:   %i %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i],NWARP);
          for (bb = wrp; bb < uind_br[0]; bb+=NWARP){ // loop through the nonzero block columns in this block row
            i1 = uind_br[0] - bb ;
            ljb = uind_br[i1*2-1];
            ub = uind_br[i1*2];
            k = mycol+ljb*grid->npcol;
            lib = LBi( k, grid ); /* Local block number, row-wise. */


            // if(lne==0)printf("  afa kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,k,uind_br[0],bmod[lib*aln_i]);


            if(lne==0){  /*only the first thread in a warp handles the lock */
                // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
                do{
                    tmp=bmod[lib*aln_i];
                    __threadfence();
                }while(tmp>-1);
            }
            __syncwarp();


            ii = X_BLK( ljb );
            usub = &Ucolind_bc_dat[Ucolind_bc_offset[ljb]];
            nub = usub[0];
            lusup = &Unzval_bc_dat[Unzval_bc_offset[ljb]];
            lloc = &Uindval_loc_bc_dat[Uindval_loc_bc_offset[ljb]];
            rel = xsup[k]; /* Global column index of block k. */

            for (i = lne; i < iknsupc; i+=WARP_SIZE){
                idx_v=2*nub+ub;
                idx_i=nub+ub;
                luptr_tmp1 = lloc[idx_v];
                lptr1_tmp = lloc[idx_i];
                lptr= lptr1_tmp+2;
                ncol = usub[lptr1_tmp+1];


                temp1=zero;
                for (l=0 ; l<ncol ; l++){
                    icol = usub[lptr+l] - rel; /* Relative col. */
                    doublecomplex ctemp;
                    zz_mult(&ctemp, &lusup[luptr_tmp1+l*iknsupc+i], &x[icol+ii]);
                    z_add(&temp1, &temp1, &ctemp);

                    // // if(offset==159 && ik==1)
                    // if(lk==8 )
                    // printf("lsum %5d %5d %5d %10f %10f %5d %5d %5d\n",l, icol, ii, x[ii+icol], lusup[luptr_tmp1+l*iknsupc+i], luptr_tmp1, ncol, iknsupc);


                    // printf("lsum %5d %5d %5d %10f %10f %10f\n",uptr-1, jj, irow - ikfrow, uval[uptr-1], xtemp, temp2[irow - ikfrow]);

                }
                // temp=z_atomicSub(&lsum[il+i],temp1);
                doublecomplex tempm;
                tempm.r=-temp1.r;
                tempm.i=-temp1.i;
                z_atomicAdd((doublecomplex *)&s_lsum[i], tempm);
            }
            __syncwarp();

            /*only the first thread in a warp modify bmod */
            if(lne==0)atomicSub((int *)&s_bmod,1);
            // if(bid==4 && lne==0)printf("  Row 4 kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

          }

        //   if(lne==0)printf("  Done kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

        #if 1
          if(wrp==0 && mycol==kcol){

                if(lne==0){  /*only the first thread in a warp handles the lock */
                    // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
                    do{
                        // tmp=s_bmod;
                        // __threadfence();
                    }while(s_bmod>0);
                }
                __syncwarp();

                // if(lne==0)printf("  jibaba kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);



                ii = X_BLK( lk );
                for (i = lne; i < iknsupc; i+=WARP_SIZE){
                    z_add(&x[i + ii ],
                    &x[i + ii ],
                    &s_lsum[i  ]);

                }

                __syncwarp();

                // Uinv = &Uinv_bc_dat[Uinv_bc_offset[jk]];

                if(nrhs==1){
                    for (i = lne; i < iknsupc; i+=WARP_SIZE){
                        temp1=zero;
                        for (l=0 ; l<iknsupc ; l++){
                            z_add(&temp1,
                            &temp1,
                            &s_lsum[i]);

                        }
                        s_lsum[i]=temp1; //reuse lsum as temporary output as it's no longer accessed
                    }
                    __syncwarp();

                    for (i = lne; i < iknsupc; i+=WARP_SIZE){
                        x[i + ii] = s_lsum[i];
                        // // if(lk==69)
                        // printf("lk %5d %5d %lf\n",lk,i, x[i + ii]);
                        }
                    __syncwarp();
                    // if(lne==0)bmod_tmp=atomicSub(&bmod[lk*aln_i],1); // set bmod[lk*aln_i] to -1
                    if(lne==0)bmod[lk*aln_i]=-1; // set bmod[lk*aln_i] to -1
                }
            }

        #else
        // __syncthreads();
        if(mycol==kcol){

            // if(tid==0){  /*only the first thread in the block handles the lock */
            //     // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
            //     do{
            //         tmp=s_bmod;
            //         __threadfence();
            //     }while(tmp>0);
            // }
            __syncthreads();

            // if(lne==0)printf("  jibaba kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

            ii = X_BLK( lk );
            for (i = tid; i < iknsupc; i+=block_size){
                z_add(&s_lsum[i],
                &s_lsum[i],
                &x[i + ii ]);
            }
            __syncthreads();

            Uinv = &Uinv_bc_dat[Uinv_bc_offset[jk]];

            if(nrhs==1){
                for (i = tid; i < iknsupc; i+=block_size){
                    temp1=zero;
                    for (l=0 ; l<iknsupc ; l++){
                        doublecomplex ctemp;
                        zz_mult(&ctemp, &Uinv[l*iknsupc+i], &s_lsum[l]);
                        z_add(&temp1, &temp1, &ctemp);
                    }
                    x[i + ii]=temp1; //reuse lsum as temporary output as it's no longer accessed
                }
                __syncthreads();

                if(tid==0)bmod[lk*aln_i]=-1; // set bmod[lk*aln_i] to -1
            }
        }
        #endif





        }
        }
} /* zlsum_bmod_inv_gpu_1rhs_new */



__global__ void zlsum_bmod_inv_gpu_1rhs_new_rowdata
/************************************************************************/
(
 int nbrow_loc,
 doublecomplex *lsum,    /* Sum of local modifications.                        */
 doublecomplex *x,       /* X array (local)                                    */
 int   nrhs,      /* Number of right-hand sides.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int *bmod,     /* Modification count for U-solve.                    */
 C_Tree  *UBtree_ptr,
 C_Tree  *URtree_ptr,
 int_t *ilsum,
 int_t *Ucolind_br_dat,
 long int *Ucolind_br_offset,
 doublecomplex *Unzval_br_new_dat,
 long int *Unzval_br_new_offset,
doublecomplex *Uinv_bc_dat,
long int *Uinv_bc_offset,
int_t *xsup,
gridinfo_t *grid
)
{
  //   doublecomplex xtemp;
  //   doublecomplex *dest;
    doublecomplex *Uinv;/* Inverse of diagonal block */
    int    iam, iknsupc, myrow, mycol, kcol;
    int_t  k,i, bb, l,ii, lk, jk, lib, ljb;
    int_t gik, rel, ncol, icol;
    doublecomplex temp1;
    __shared__ doublecomplex s_lsum[MAXSUPER];
    // volatile __shared__ doublecomplex temp2[MAXSUPER];
    volatile __shared__ int s_bmod;
    int aln_i;
    aln_i = 1;//ceil(CACHELINE/(double)iword);

    int bid;
    int_t tmp;
    int tid = threadIdx_x + threadIdx_y * blockDim_x;
    const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
    doublecomplex zero = {0.0, 0.0};
  //   doublecomplex rC[THR_N][THR_M];
    // __shared__ doublecomplex x_share[DIM_X*DIM_Y];

    bid= nbrow_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
    int_t  *usub;
    doublecomplex *lusup;
    int  LDA;

    int wrp;
    int lne;
    wrp= tid;
    lne=wrp%WARP_SIZE;
    wrp/=WARP_SIZE;
// printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,wrp,tid);



  //   printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


    // rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


    // the first nbcol_loc handles all computations and broadcast communication
    if(bid<nbrow_loc){

        lk=bid;
        iam = grid->iam;
        mycol = MYCOL( iam, grid );
        myrow = MYROW( iam, grid );
        gik = myrow+lk*grid->nprow;
        if(gik<nsupers){
        iknsupc = SuperSize( gik );
      //   il = LSUM_BLK( lk );
        kcol = PCOL( gik, grid );
        jk = LBj( gik, grid ); /* Local block number, column-wise. */

        if(Uinv_bc_offset[jk]==-1 && Ucolind_br_offset[lk]==-1){
        return;
        }


        // initialize the shared memory data, which requires __syncthreads
        if(tid==0)s_bmod = bmod[lk*aln_i];
        for (i = tid; i < MAXSUPER; i+=block_size){s_lsum[i]=zero;}
        __syncthreads();

        usub = &Ucolind_br_dat[Ucolind_br_offset[lk]];
        lusup = &Unzval_br_new_dat[Unzval_br_new_offset[lk]];
        int_t nubr = usub[0];
      //   if(lne==0)
      //      printf("  Entering kernel:   %i %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i],NWARP);
        for (bb = wrp; bb < nubr; bb+=NWARP){ // loop through the nonzero block columns in this block row
          k=usub[UB_DESCRIPTOR_NEWUCPP+bb];
          int_t idx_s = usub[UB_DESCRIPTOR_NEWUCPP+nubr+bb];
          ncol = usub[UB_DESCRIPTOR_NEWUCPP+nubr+bb+1] - idx_s;
          LDA = usub[2];
          ljb = LBj( k, grid ); /* Local block number, column-wise. */
          lib = LBi( k, grid ); /* Local block number, row-wise. */

          // if(lne==0)printf("  afa kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,k,uind_br[0],bmod[lib*aln_i]);

          if(lne==0){  /*only the first thread in a warp handles the lock */
              // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
              do{
                  tmp=bmod[lib*aln_i];
                  __threadfence();
              }while(tmp>-1);
          }
          __syncwarp();


          ii = X_BLK( ljb );
          rel = xsup[k]; /* Global column index of block k. */

          for (i = lne; i < LDA; i+=WARP_SIZE){
              temp1=zero;

              for (l=0 ; l<ncol ; l++){
                  icol = usub[UB_DESCRIPTOR_NEWUCPP+2*nubr+1+idx_s+l] - rel; /* Relative col. */
                doublecomplex ctemp;
                zz_mult(&ctemp, &lusup[idx_s*LDA+l*LDA+i], &x[icol+ii]);
                z_add(&temp1, &temp1, &ctemp);
                  // // if(offset==159 && ik==1)
                  // if(lk==8 )
                  // printf("lsum %5d %5d %5d %10f %10f %5d %5d %5d\n",l, icol, ii, x[ii+icol], lusup[luptr_tmp1+l*iknsupc+i], luptr_tmp1, ncol, iknsupc);


                  // printf("lsum %5d %5d %5d %10f %10f %10f\n",uptr-1, jj, irow - ikfrow, uval[uptr-1], xtemp, temp2[irow - ikfrow]);

              }
              // temp=z_atomicSub(&lsum[il+i],temp1);
                doublecomplex tempm;
                tempm.r=-temp1.r;
                tempm.i=-temp1.i;
                z_atomicAdd((doublecomplex *)&s_lsum[i+iknsupc-LDA], tempm);

          }
          __syncwarp();

          /*only the first thread in a warp modify bmod */
          if(lne==0)tmp=atomicSub((int *)&s_bmod,1);
          // if(bid==4 && lne==0)printf("  Row 4 kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

        }

      //   if(lne==0)printf("  Done kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

      #if 1
        if(wrp==0 && mycol==kcol){

              if(lne==0){  /*only the first thread in a warp handles the lock */
                  // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
                  do{
                      // tmp=s_bmod;
                      // __threadfence();
                  }while(s_bmod>0);
              }
              __syncwarp();

              // if(lne==0)printf("  jibaba kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);



              ii = X_BLK( lk );
              for (i = lne; i < iknsupc; i+=WARP_SIZE){
                z_add(&x[i + ii ],
                &x[i + ii ],
                &s_lsum[i  ]);
              }

              __syncwarp();

              Uinv = &Uinv_bc_dat[Uinv_bc_offset[jk]];

              if(nrhs==1){
                  for (i = lne; i < iknsupc; i+=WARP_SIZE){
                      temp1=zero;
                      for (l=0 ; l<iknsupc ; l++){
                        doublecomplex ctemp;
                        zz_mult(&ctemp, &Uinv[l*iknsupc+i], &x[ii+l]);
                        z_add(&temp1, &temp1, &ctemp);
                      }
                      s_lsum[i]=temp1; //reuse lsum as temporary output as it's no longer accessed
                  }
                  __syncwarp();

                  for (i = lne; i < iknsupc; i+=WARP_SIZE){
                      x[i + ii] = s_lsum[i];
                      // // if(lk==69)
                      // printf("lk %5d %5d %lf\n",lk,i, x[i + ii]);
                      }
                  __syncwarp();
                  // if(lne==0)bmod_tmp=atomicSub(&bmod[lk*aln_i],1); // set bmod[lk*aln_i] to -1
                  if(lne==0)bmod[lk*aln_i]=-1; // set bmod[lk*aln_i] to -1
              }
          }

      #else
      // __syncthreads();
      if(mycol==kcol){

          // if(tid==0){  /*only the first thread in the block handles the lock */
          //     // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
          //     do{
          //         tmp=s_bmod;
          //         __threadfence();
          //     }while(tmp>0);
          // }
          __syncthreads();

          // if(lne==0)printf("  jibaba kernel:   %i %i %i %i %i %i %i %i %i %i\n", threadIdx_x, bid, grid->npcol, nsupers,myrow,krow,wrp,tid,uind_br[0],bmod[lk*aln_i]);

          ii = X_BLK( lk );
          for (i = tid; i < iknsupc; i+=block_size){
                z_add(&s_lsum[i],
                &s_lsum[i],
                &x[i + ii ]);
          }
          __syncthreads();

          Uinv = &Uinv_bc_dat[Uinv_bc_offset[jk]];

          if(nrhs==1){
              for (i = tid; i < iknsupc; i+=block_size){
                  temp1=zero;
                  for (l=0 ; l<iknsupc ; l++){
                    doublecomplex ctemp;
                    zz_mult(&ctemp, &Uinv[l*iknsupc+i], &s_lsum[l]);
                    z_add(&temp1, &temp1, &ctemp);

                  }
                  x[i + ii]=temp1; //reuse lsum as temporary output as it's no longer accessed
              }
              __syncthreads();

              if(tid==0)bmod[lk*aln_i]=-1; // set bmod[lk*aln_i] to -1
          }
      }
      #endif





      }
      }
} /* zlsum_bmod_inv_gpu_1rhs_new_rowdata */









__global__ void zlsum_bmod_inv_gpu_1rhs_warp
/************************************************************************/
(
 int nbcol_loc,
 doublecomplex *lsum,    /* Sum of local modifications.                        */
 doublecomplex *x,       /* X array (local)                                    */
 int   nrhs,      /* Number of right-hand sides.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int *bmod,     /* Modification count for U-solve.                    */
 C_Tree  *UBtree_ptr,
 C_Tree  *URtree_ptr,
 int_t *ilsum,
 int_t *Ucolind_bc_dat,
 long int *Ucolind_bc_offset,
 doublecomplex *Unzval_bc_dat,
 long int *Unzval_bc_offset,
doublecomplex *Uinv_bc_dat,
long int *Uinv_bc_offset,
int_t *Uindval_loc_bc_dat,
long int *Uindval_loc_bc_offset,
int_t *xsup,
gridinfo_t *grid
)
{
    doublecomplex zero = {0.0, 0.0};
	doublecomplex *Uinv;/* Inverse of diagonal block */
	int    iam, iknsupc, myrow, mycol, krow;
	int_t  k,i, l,ii, ik, il, j, lk, lib, ub;
	int_t gik, rel, lptr, ncol, icol;
	doublecomplex temp1;
	// __shared__ doublecomplex temp2[MAXSUPER];
	int aln_i;
	aln_i = 1;//ceil(CACHELINE/(double)iword);
	int   knsupc;    /* Size of supernode k.                               */
	int nub;       /* Number of L blocks.                                */

	// int bid;
	int_t tmp;
	// int bmod_tmp;
	int tid = threadIdx_x + threadIdx_y * blockDim_x;
	const int block_size = blockDim_x*blockDim_y; /* number of threads per block*/
	// doublecomplex rC[THR_N][THR_M];
	// __shared__ doublecomplex x_share[DIM_X*DIM_Y];

	// bid= nbcol_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
	// int idx = threadIdx_x;  // thread's m dimension
	//int idy = threadIdx_y;  // thread's n dimension
	int_t  *usub, *lloc;
	doublecomplex *lusup;
	int_t nrow, nnz_offset, offset;
	int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;

    int wrp;
    int lne;
    // int ready = 0;
    // int lock = 0;
    wrp= tid + blockIdx_x*block_size;
    lne=wrp%WARP_SIZE;
	// printf("  Entering kernel:   %i %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,wrp,wrp/WARP_SIZE,tid);
	wrp/=WARP_SIZE;

	// rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


	// the first nbcol_loc handles all computations and broadcast communication
	if(wrp<nbcol_loc){
        wrp= nbcol_loc-wrp-1;  // This makes sure higher warp IDs are checked first in spin wait
		if(Uinv_bc_offset[wrp]==-1 && Ucolind_bc_offset[wrp]==-1){
		return;
		}

		lk=wrp;
		iam = grid->iam;
		mycol = MYCOL( iam, grid );
		myrow = MYROW( iam, grid );
		k = mycol+lk*grid->npcol;
		knsupc = SuperSize( k );
		krow = PROW( k, grid );
		usub = &Ucolind_bc_dat[Ucolind_bc_offset[lk]];
		lusup = &Unzval_bc_dat[Unzval_bc_offset[lk]];
		lloc = &Uindval_loc_bc_dat[Uindval_loc_bc_offset[lk]];
		rel = xsup[k]; /* Global column index of block ik. */

	    // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

		if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

			if(lne==0){  /*only the first thread in a warp handles the lock */


			// for (i=0 ; i<maxsup ; i++){
				// rtemp_loc[i]=0.0;
			// }

				lib = LBi( k, grid ); /* Local block number, row-wise. */
			    // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
				do{
					tmp=bmod[lib*aln_i];
					__threadfence();
				}while(tmp>0);

			}
			__syncwarp();
		  //   if(tid==0)
		  //   printf("spin: %d %d \n",threadIdx_x, blockIdx_x);


				lib = LBi( k, grid ); /* Local block number, row-wise. */
				il = LSUM_BLK( lib );
				ii = X_BLK( lib );

				RHS_ITERATE(j)
                    for (i = lne; i < knsupc; i+=WARP_SIZE){
			            z_add(&x[i + ii + j*knsupc],
                        &x[i + ii + j*knsupc],
                        &lsum[i + il + j*knsupc]);
						// if(lib==1){
						// printf("lib %5d %5d %5d %lf\n",lib,i, il, lsum[i + il + j*knsupc ]);
						// // printf("lib %5d %5d %lf\n",lib,i, x[i + ii + j*knsupc]);
						// }
					}
                __syncwarp();



			   //  if(Llu->inv == 1){

					Uinv = &Uinv_bc_dat[Uinv_bc_offset[lk]];

					if(nrhs==1){
						for (i = lne; i < knsupc; i+=WARP_SIZE){
							temp1=zero;
							for (l=0 ; l<knsupc ; l++){
                                doublecomplex ctemp;
                                zz_mult(&ctemp, &Uinv[l*knsupc+i], &x[ii+l]);
                                z_add(&temp1, &temp1, &ctemp);
							}
							lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
						}
						__syncwarp();

						for (i = lne; i < knsupc; i+=WARP_SIZE){
							x[i + ii] = lsum[il+i];
							// // if(lk==69)
							// printf("lk %5d %5d %lf\n",lk,i, x[i + ii]);
							}
                            __syncwarp();
					}//if(nrhs==1)
			   //  }

			  //   RHS_ITERATE(j)
			  //   for (i = tid; i < knsupc; i+=block_size)
			  // 	  recvbuf_BC_gpu[i + maxrecvsz*lk + j*knsupc ] = x[i + ii + j*knsupc];

              __syncwarp();
		}else{   /* off-diagonal block forward the message*/
			/* waiting for the x subvector and forward*/
		}


	  //   if(tid==0){  //YL: only the first thread in a block forwards the x subvector using NVSHMEM
	  //   cnt=LBtree_ptr[lk].destCnt_;
	  //  //  printf("good1 %5d%5d\n",lk,cnt);
	  //   if(cnt>0){
	  // 	 cnt=LBtree_ptr[lk].msgSize_;
	  // 	  zC_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&recvbuf_BC_gpu[maxrecvsz*lk],cnt*nrhs+XK_H);
	  //   }
	  //   }

		if(Ucolind_bc_offset[lk]!=-1){
			nub = usub[0];      /* Number of U blocks in block column lk */
		}else{
			nub = 0;
		}
		if(nub>0){
                nrow = usub[1];  // total number of nonzero rows
                nnz_offset = usub[2]; // total number of nonzero column segments

				lib = LBi( k, grid ); /* Local block number, row-wise. */
				ii = X_BLK( lib );

				if(nrhs==1){
				// // if(0){
				// 	for (i=lne;i<knsupc;i+=WARP_SIZE)
				// 		temp2[i]=x[ii+i];
                //         __syncwarp();

                    for (i = lne; i < nrow; i+=WARP_SIZE){
                        // printf("good1 bid nub i nrow %5d %5d %5d %5d\n",bid, nub, i, nrow);
                        ub = usub[nnz_offset+i*2];
                        offset = usub[nnz_offset+i*2+1];
                        ik = lloc[ub];
                        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                        iknsupc = SuperSize( gik );
                        // // if(lk==2 && ik==1)
                        // // printf("ub offset %5d %5d %5d %5d\n",ub, i, offset,SuperSize( gik ));

                        idx_v=2*nub+ub;
                        idx_i=nub+ub;
                        luptr_tmp1 = lloc[idx_v];
                        lptr1_tmp = lloc[idx_i];
                        lptr= lptr1_tmp+2;
                        ncol = usub[lptr1_tmp+1];
                        il = LSUM_BLK( ik );

                        temp1=zero;
                        for (l=0 ; l<ncol ; l++){
                            icol = usub[lptr+l] - rel; /* Relative col. */
                            doublecomplex ctemp;
                            zz_mult(&ctemp, &lusup[luptr_tmp1+l*iknsupc+offset], &x[icol+ii]);
                            z_add(&temp1, &temp1, &ctemp);
                            // // if(offset==159 && ik==1)
                            // if(lk==2 && ik==1)
                            // printf("lsum %5d %5d %5d %10f %10f %5d %5d %5d\n",l, icol, offset, x[ii+j*knsupc+icol], lusup[luptr_tmp1+l*iknsupc+offset], luptr_tmp1, ncol, iknsupc);


                            // printf("lsum %5d %5d %5d %10f %10f %10f\n",uptr-1, jj, irow - ikfrow, uval[uptr-1], xtemp, temp2[irow - ikfrow]);

                        }
                        doublecomplex tempm;
                        tempm.r=-temp1.r;
                        tempm.i=-temp1.i;
                        z_atomicAdd(&lsum[il+offset], tempm);

                    }
                    __syncwarp();

                    for (ub = lne; ub < nub; ub+=WARP_SIZE){
                        ik = lloc[ub];
                        atomicSub(&bmod[ik*aln_i],1);
                        // printf("ik %5d bmod[ik*aln_i] %5d\n",ik,bmod[ik*aln_i]);
                    }
				}//if(nrhs==1)
                __syncwarp();
			// } /*if tid<Nchunk*/
		} /* if nlb>0*/

		// printf("nimbgood \n");

//   }else if(bid<nbcol_loc+nblock_ex){  //the next nblock_ex blocks handle all reduction communication

}



} /* zlsum_bmod_inv_gpu_1rhs_warp */






/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */

 __global__ void zlsum_bmod_inv_gpu_mrhs_nvshmem
 /************************************************************************/
         (
                 int nbcol_loc,
                 doublecomplex *lsum,    /* Sum of local modifications.                        */
                 doublecomplex *x,       /* X array (local)                                    */
                 int   nrhs,      /* Number of right-hand sides.                        */
                 int_t   nsupers,      /* Number of total supernodes.                        */
                 int *bmod,     /* Modification count for U-solve.                    */
                 C_Tree  *UBtree_ptr,
                 C_Tree  *URtree_ptr,
                 int_t *ilsum,
                 int_t *Ucolind_bc_dat,
                 long int *Ucolind_bc_offset,
                 doublecomplex *Unzval_bc_dat,
                 long int *Unzval_bc_offset,
                 doublecomplex *Uinv_bc_dat,
                 long int *Uinv_bc_offset,
                 int_t *Uindval_loc_bc_dat,
                 long int *Uindval_loc_bc_offset,
                 int_t *xsup,
                 gridinfo_t *grid,
                 int_t maxrecvsz,
                 int mype,
                 volatile uint64_t* flag_bc_q,
                 volatile uint64_t* flag_rd_q,
                 doublecomplex* zready_x,
                 doublecomplex* zready_lsum,
                 int* my_flag_bc,
                 int* my_flag_rd,
                 int* d_nfrecv,
                 volatile int* d_status,
                 volatile int* d_statusmod,
                 int nblock_ex,
                 int maxsuper,
         int* d_flag_mod_u
         )
 {
    doublecomplex zero = {0.0, 0.0};
    doublecomplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
     doublecomplex *Uinv;/* Inverse of diagonal block */
     int    iam, iknsupc, myrow, mycol, krow;
     int_t  k,i, l, ii, ik, il, j, lk, lib, ub;
     int_t gik, rel, lptr, ncol, icol;
     doublecomplex temp1;
     __shared__ doublecomplex temp2[MAXSUPER];
     int aln_i;
     aln_i = 1;//ceil(CACHELINE/(double)iword);
     int   knsupc;    /* Size of supernode k.                               */
     int nub;       /* Number of L blocks.                                */

     int bid;
     int_t tmp;
     int bmod_tmp;
     int tid = threadIdx_x + threadIdx_y * blockDim_x;
     const int block_size = blockDim_x*blockDim_y; /* number of threads per warp*/
     doublecomplex rC[THR_N][THR_M];
     // __shared__ doublecomplex x_share[DIM_X*DIM_Y];

     bid= nbcol_loc-blockIdx_x-1;  // This makes sure higher block IDs are checked first in spin wait
     int idx = threadIdx_x;  // thread's m dimension
     int idy = threadIdx_y;  // thread's n dimension
     int ni,mi;
     int_t  *usub, *lloc;
     doublecomplex *lusup;
     int_t nrow, nnz_offset, offset;
     int_t  luptr_tmp1,lptr1_tmp, idx_i, idx_v;
     int cnt;


     // printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);


     // rtemp_loc = (doublecomplex*)malloc(maxsup*nrhs*Nbk*sizeof(doublecomplex));


     // the first nbcol_loc handles all computations and broadcast communication
     //if(bid<nbcol_loc){
     if(Uinv_bc_offset[bid]==-1 && Ucolind_bc_offset[bid]==-1){
         return;
     }

     lk=bid;
     iam = grid->iam;
     mycol = MYCOL( iam, grid );
     myrow = MYROW( iam, grid );
     k = mycol+lk*grid->npcol;
     knsupc = SuperSize( k );
     krow = PROW( k, grid );
     usub = &Ucolind_bc_dat[Ucolind_bc_offset[lk]];
     lusup = &Unzval_bc_dat[Unzval_bc_offset[lk]];
     lloc = &Uindval_loc_bc_dat[Uindval_loc_bc_offset[lk]];
     rel = xsup[k]; /* Global column index of block ik. */

     // printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

     if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

         if(tid==0){  /*only the first thread in a block handles the lock */


             // for (i=0 ; i<maxsup ; i++){
             // rtemp_loc[i]=0.0;
             // }

             lib = LBi( k, grid ); /* Local block number, row-wise. */
             // printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,bmod[lib*aln_i],myrow,krow);
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
             for (i = tid; i < knsupc; i+=block_size){
			z_add(&x[i + ii + j*knsupc],
                &x[i + ii + j*knsupc],
                &lsum[i + il + j*knsupc]);
                 // if(lib==1){
                 // printf("lib %5d %5d %5d %lf\n",lib,i, il, lsum[i + il + j*knsupc ]);
                 // // printf("lib %5d %5d %lf\n",lib,i, x[i + ii + j*knsupc]);
                 // }
             }
         __syncthreads();



         //  if(Llu->inv == 1){

         Uinv = &Uinv_bc_dat[Uinv_bc_offset[lk]];

         if(nrhs==1){
             for (i = tid; i < knsupc; i+=block_size){
                 temp1=zero;
                 for (l=0 ; l<knsupc ; l++){
                    doublecomplex ctemp;
                    zz_mult(&ctemp, &Uinv[l*knsupc+i], &x[ii+l]);
                    z_add(&temp1, &temp1, &ctemp);

                 }
                 lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
             }
             __syncthreads();

             for (i = tid; i < knsupc; i+=block_size){
                 x[i + ii] = lsum[il+i];
                 // // if(lk==69)
                 // printf("lk %5d %5d %lf\n",lk,i, x[i + ii]);
             }
             __syncthreads();
         }else{
             __syncthreads();
             for (int blx = 0; blx*BLK_M < knsupc; blx++){
                 for (int bly = 0; bly*BLK_N < nrhs; bly++){
                     gemm_device_zlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
                                            Uinv, knsupc, &x[ii], knsupc, rC,
                                            alpha, beta);
 #pragma unroll
                     for (ni = 0; ni < THR_N; ni++) {
                         int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
 #pragma unroll
                         for (mi = 0; mi < THR_M; mi++) {
                             int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                             if (coord_dCm < knsupc && coord_dCn < nrhs) {
                                 doublecomplex &regC = rC[ni][mi];
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

         RHS_ITERATE(j)
             for (i = tid; i < knsupc; i+=block_size)
                 zready_x[i + maxrecvsz*lk + j*knsupc ] = x[i + ii + j*knsupc];

         __syncthreads();
     }else{   /* off-diagonal block forward the message*/
         /* waiting for the x subvector and forward*/
         volatile uint64_t msg_recv = 0;
         if (tid == 0) {
             //printf("in solve WAIT1 (%d,%d) wait for col %d,flag=%d\n", mype, bid, gc,flag_bc_q[lk]);
             do {
                 msg_recv = flag_bc_q[lk];
                 //msg_recv=d_status[gc];
                 //msg_recv=flag_bc_q[gc];
                 __threadfence();
             } while (msg_recv != 1);
             //double sum=0;
             //for (int myi=0;myi<UBtree_ptr[lk].msgSize_*nrhs+XK_H;myi++){
             //    sum+=zready_x[maxrecvsz*lk+myi];
             //    printf("--- (%d,%d,%d), gc=%d,lk=%d, maxrecvsz=%d, myi=%d, idx=%d, val=%lf\n",
             //                 mype,bid,tid,gc,lk,maxrecvsz, myi,maxrecvsz*lk+myi,zready_x[maxrecvsz*lk+myi]);
             //}
             //printf("(%d,%d,%d), gc=%d,lk=%d, sum=%lf, msgsz=%d\n",mype,bid,tid,gc,lk,sum,UBtree_ptr[lk].msgSize_*nrhs+XK_H);
         }
         __syncthreads();
     } // end waiting for msg
     cnt = UBtree_ptr[lk].destCnt_;
     //if (tid==0) printf("in solve forward (%d,%d) done my col %d, cnt=%d, nub=%d\n", mype, bid, gc, cnt,nub);
     if (cnt > 0) {
         //cnt=LBtree_ptr[lk].msgSize_;
         my_flag_bc[k * RDMA_FLAG_SIZE] = lk;
         my_flag_bc[k * RDMA_FLAG_SIZE + 1] = UBtree_ptr[lk].msgSize_ * nrhs + XK_H;
         zC_BcTree_forwardMessageSimple_Device(&UBtree_ptr[lk], flag_bc_q, &my_flag_bc[k * RDMA_FLAG_SIZE],
                                              mype, tid, &zready_x[0], maxrecvsz);
         //if (tid==0) printf("(%d,%d,%d), lk=%d, gc=%d\n",mype,bid,tid,lk,gc);
     }
     int keep_lk = lk;
     __syncthreads();

     if(Ucolind_bc_offset[bid]!=-1){
         nub = usub[0];      /* Number of U blocks in block column lk */
     }else{
         nub = 0;
     }
     if(nub>0){
        nrow = usub[1];  // total number of nonzero rows
        nnz_offset = usub[2]; // total number of nonzero column segments

         lib = LBi( k, grid ); /* Local block number, row-wise. */
         ii = X_BLK( lib );

         if(nrhs==1){
             for (i=tid;i<knsupc;i+=block_size)
                 temp2[i]=zready_x[i + maxrecvsz*keep_lk]; // Nan
             __syncthreads();
             for (i = tid; i < nrow; i+=block_size){
                 // printf("good1 bid nub i nrow %5d %5d %5d %5d\n",bid, nub, i, nrow);
                 ub = usub[nnz_offset+i*2];
                 offset = usub[nnz_offset+i*2+1];
                 ik = lloc[ub]; /* Local block number, row-wise. */
                 gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                 iknsupc = SuperSize( gik );
                 // // if(lk==2 && ik==1)
                 // // printf("ub offset %5d %5d %5d %5d\n",ub, i, offset,SuperSize( gik ));

                 idx_v=2*nub+ub;
                 idx_i=nub+ub;
                 luptr_tmp1 = lloc[idx_v];
                 lptr1_tmp = lloc[idx_i];
                 lptr= lptr1_tmp+2;
                 ncol = usub[lptr1_tmp+1];
                 il = LSUM_BLK( ik );

                 temp1=zero;
                 for (l=0 ; l<ncol ; l++){
                     icol = usub[lptr+l] - rel; /* Relative col. */
                    doublecomplex ctemp;
                    zz_mult(&ctemp, &lusup[luptr_tmp1+l*iknsupc+offset], &temp2[icol]);
                    z_add(&temp1, &temp1, &ctemp);
                     // // if(offset==159 && ik==1)
                     // if(lk==2 && ik==1)
                     // printf("lsum %5d %5d %5d %10f %10f %5d %5d %5d\n",l, icol, offset, x[ii+j*knsupc+icol], lusup[luptr_tmp1+l*iknsupc+offset], luptr_tmp1, ncol, iknsupc);


                     // printf("lsum %5d %5d %5d %10f %10f %10f\n",uptr-1, jj, irow - ikfrow, uval[uptr-1], xtemp, temp2[irow - ikfrow]);

                 }

                doublecomplex tempm;
                tempm.r=-temp1.r;
                tempm.i=-temp1.i;
                z_atomicAdd(&lsum[il+offset], tempm);

             }
         }else{
             for (ub = 0; ub < nub; ub++){
                 ik = lloc[ub];
                 gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                 iknsupc = SuperSize( gik );
                 idx_v=2*nub+ub;
                 idx_i=nub+ub;
                 luptr_tmp1 = lloc[idx_v];
                 lptr1_tmp = lloc[idx_i];
                 lptr= lptr1_tmp+2;
                 ncol = usub[lptr1_tmp+1];
                 il = LSUM_BLK( ik );


                 for (int blx = 0; blx*BLK_M < iknsupc; blx++){
                     for (int bly = 0; bly*BLK_N < nrhs; bly++){

                         gemm_device_zlsum_bmod_stridedB(iknsupc, nrhs, ncol, blx, bly,
                         &lusup[luptr_tmp1], iknsupc, &x[ii], knsupc, rC,
                         alpha, beta, lptr, rel, usub);

                         #pragma unroll
                         for (ni = 0; ni < THR_N; ni++) {
                             int coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
                             #pragma unroll
                             for (mi = 0; mi < THR_M; mi++) {
                                 int coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                                 if (coord_dCm < iknsupc && coord_dCn < nrhs) {
                                     doublecomplex &regC = rC[ni][mi];
                                    doublecomplex tempm;
                                    tempm.r=-regC.r;
                                    tempm.i=-regC.i;
                                    z_atomicAdd(&lsum[il+coord_dCm + coord_dCn*iknsupc], tempm);
                                 }
                             }
                         }
                     }
                 }
             }
         }
         __syncthreads();

         for (ub = tid; ub < nub; ub+=block_size){
             ik = lloc[ub];
             gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
             iknsupc = SuperSize( gik );
             l = LSUM_BLK(ik);
             bmod_tmp=atomicSub(&bmod[ik*aln_i],1);
             // printf("ik %5d bmod[ik*aln_i] %5d\n",ik,bmod[ik*aln_i]);
             if(bmod_tmp==1) {// forward RD
                 //senddone[lk]=1;
                 if(URtree_ptr[ik].myRoot_ != URtree_ptr[ik].myRank_){
                     //cnt=LRtree_ptr[lib].msgSize_;

                     my_flag_rd[ik*RDMA_FLAG_SIZE]=ik;
                     my_flag_rd[ik*RDMA_FLAG_SIZE+1]=URtree_ptr[ik].msgSize_;
                     // double tmp_sum=0;
                     RHS_ITERATE(j) {
                         for (int aab = 0; aab < iknsupc; aab++) {
                             zready_lsum[ik * maxrecvsz * 2 + aab +j * iknsupc] = lsum[l + aab +j * iknsupc];
                             //zready_lsum[lk * maxrecvsz * 2 + aab +j * iknsupc] = lsum[il + aab +j * iknsupc];
                             // tmp_sum += zready_lsum[ik * maxrecvsz * 2 + aab +j * iknsupc];
                             //printf("u data3-(%d,%d,%d),lib=%d,k=%d,sum=%lf,zready_lsum[%d]=%lf, size=%d\n", mype, bid, tid, ik, gik, tmp_sum,
                             //       ik * maxrecvsz * 2 + aab +j * iknsupc,
                             //       zready_lsum[ik * maxrecvsz * 2 + aab +j * iknsupc],my_flag_rd[gik*RDMA_FLAG_SIZE+1]);

                         }
                     }
                     // int temp_mysendcout=atomicAdd(&d_flag_mod_u[0], 1);
                     // int temp_flag_mod=atomicExch(&d_flag_mod_u[temp_mysendcout+1],ik);
                     //printf("iam=%d in solve,lib=%d,%d,%d, "
                     //       "pos=%d, temp %d,%d, "
                     //       "maxrecvsz=%d\n",mype,ik,gik, d_flag_mod_u[temp_mysendcout+1],
                     //       temp_mysendcout+1,
                     //       temp_mysendcout,temp_flag_mod,
                     //       maxrecvsz);
                     //printf("(%d,%d,%d) in u solve,lib=%d,gr=%d,myflagrd=%d,%d, sum=%lf\n",mype,bid,tid,ik,gik,my_flag_rd[gik*RDMA_FLAG_SIZE],my_flag_rd[gik*RDMA_FLAG_SIZE+1], tmp_sum);
                     zC_RdTree_forwardMessageSimple_Device(&URtree_ptr[ik], flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*ik], mype, bid, tid, &zready_lsum[0],maxrecvsz,URtree_ptr[ik].myRoot_);
                 }
             }
         }
         __syncthreads();
         // } /*if tid<Nchunk*/
     } /* if nub>0*/

     // printf("nimbgood \n");

 //   }else if(bid<nbcol_loc+nblock_ex){  //the next nblock_ex blocks handle all reduction communication

     //}

 } /* zlsum_bmod_inv_gpu_mrhs_nvshmem */



void zlsum_bmod_inv_gpu_wrap
(
    superlu_dist_options_t *options,
    int nbcol_loc,    /*number of local supernode columns*/
    int nbrow_loc,    /*number of local supernode rows*/
    int nthread_x,     /*kernel launch parameter*/
    int nthread_y,     /*kernel launch parameter*/
    doublecomplex *lsum,    /* Sum of local modifications.                        */
    doublecomplex *x,       /* X array (local)                                    */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   maxsup,      /* Max supernode size.                        */
    int_t   nsupers,      /* Number of total supernodes.                        */
    int *bmod,     /* Modification count for L-solve.                    */
    C_Tree  *UBtree_ptr,
    C_Tree  *URtree_ptr,
    int_t *ilsum,
    int_t *Ucolind_bc_dat,
    int64_t *Ucolind_bc_offset,
    int_t *Ucolind_br_dat,
    int64_t *Ucolind_br_offset,
    int_t *Uind_br_dat,
    int64_t *Uind_br_offset,
    doublecomplex *Unzval_bc_dat,
    int64_t *Unzval_bc_offset,
    doublecomplex *Unzval_br_new_dat,
    int64_t *Unzval_br_new_offset,
    doublecomplex *Uinv_bc_dat,
    int64_t *Uinv_bc_offset,
    int_t *Uindval_loc_bc_dat,
    int64_t *Uindval_loc_bc_offset,
    int_t *xsup,
    gridinfo_t *grid,
    int_t maxrecvsz,
    uint64_t* flag_bc_q,
    uint64_t* flag_rd_q,
    doublecomplex* zready_x,
    doublecomplex* zready_lsum,
    int* my_flag_bc,
    int* my_flag_rd,
    int* d_nfrecv_u,
    int* h_nfrecv_u,
    int* d_status,
    int* d_colnum_u,
    int* d_mynum_u,
    int* d_mymaskstart_u,
    int* d_mymasklength_u,
    int* d_nfrecvmod_u,
    int* d_statusmod,
    int* d_colnummod_u,
    int* d_mynummod_u,
    int* d_mymaskstartmod_u,
    int* d_mymasklengthmod_u,
    int* d_recv_cnt_u,
    int* d_msgnum,
    int* d_flag_mod_u,
    int procs
) {

//printf("pinv %d\n",Llu->inv);
//fflush(stdout);
int maxsuper = sp_ienv_dist(3, options);
if (MAXSUPER < maxsuper) {
printf("increase MAXSUPER\n");
exit(1);
}

if(procs==1){
#ifdef SINGLE_RHS_OPT
    if(nrhs>1){
#else
    if(1){
#endif
        dim3 dimBlock(nthread_x, nthread_y);
        zlsum_bmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Ucolind_bc_dat,Ucolind_bc_offset,Unzval_bc_dat,Unzval_bc_offset,Uinv_bc_dat,Uinv_bc_offset,Uindval_loc_bc_dat,Uindval_loc_bc_offset,xsup,grid);
    }else{
        dim3 dimBlock(nthread_x, nthread_y,1);
        // zlsum_bmod_inv_gpu_1rhs_warp<<< CEILING(nbcol_loc,NWARP), dimBlock >>>(nbcol_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Ucolind_bc_dat,Ucolind_bc_offset,Unzval_bc_dat,Unzval_bc_offset,Uinv_bc_dat,Uinv_bc_offset,Uindval_loc_bc_dat,Uindval_loc_bc_offset,xsup,grid);
#ifdef U_BLOCK_PER_ROW_ROWDATA
        zlsum_bmod_inv_gpu_1rhs_new_rowdata<<< nbrow_loc, dimBlock >>>(nbrow_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Ucolind_br_dat,Ucolind_br_offset,Unzval_br_new_dat,Unzval_br_new_offset,Uinv_bc_dat,Uinv_bc_offset,xsup,grid);
#else
        zlsum_bmod_inv_gpu_1rhs_new<<< nbrow_loc, dimBlock >>>(nbrow_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Ucolind_bc_dat,Ucolind_bc_offset,Uind_br_dat,Uind_br_offset,Unzval_bc_dat,Unzval_bc_offset,Uinv_bc_dat,Uinv_bc_offset,Uindval_loc_bc_dat,Uindval_loc_bc_offset,xsup,grid);
#endif
    }


    gpuDeviceSynchronize();
}else{
    #ifdef HAVE_NVSHMEM
    int nblock_ex = CEILING(nbrow_loc, ((nthread_x * nthread_y) / 32)); //32 (warp) * 8 =256
    //int nblock_ex = nbrow_loc; //CEILING(nbrow_loc, ((nthread_x * nthread_y) / 32)); //32 (warp) * 8 =256
    //int nblock_ex = CEILING(nbrow_loc, ((nthread_x * nthread_y) / 64)); //32 (warp) * 8 =256
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
    //cudaStreamCreate(&stream[i]);
    cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }

    // int mype, npes;
    int mype;
    mype = nvshmem_my_pe();
    // npes = nvshmem_n_pes();
    //printf("(%d) nbcol_loc %d\n", mype, nbcol_loc);
    //printf("(%d), U Enter,mynode=%d\n",mype,mype_node);
    //fflush(stdout);
    dim3 dimGrid(nbcol_loc);
    dim3 dimBlock(nthread_x, nthread_y);

    //if (npes == 1) {
    //    zlsum_bmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc, lsum, x, nrhs, nsupers, bmod, UBtree_ptr,
    //                                                       URtree_ptr, ilsum,
    //                                                       Ucolind_bc_dat, Ucolind_bc_offset, Unzval_bc_dat,
    //                                                       Unzval_bc_offset, Uinv_bc_dat, Uinv_bc_offset,
    //                                                       Uindval_loc_bc_dat, Uindval_loc_bc_offset, xsup, grid);
    //} else {

    cudaFuncAttributes cuattr;
    cudaFuncGetAttributes(&cuattr, zlsum_bmod_inv_gpu_mrhs_nvshmem);
    cudaDeviceSetLimit(cudaLimitStackSize, cuattr.localSizeBytes);
    //printf("(%d) CUDA kernel localSizeByte=%d\n",mype,cuattr.localSizeBytes);
    //fflush(stdout);

    int minGridSize;
    int myblockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &myblockSize, (const void *) zwait_bcrd_u, 0, 0);
    if (myblockSize < h_nfrecv_u[1]) {
    h_nfrecv_u[1] = myblockSize;
    gpuMemcpy(d_nfrecv_u, h_nfrecv_u, 3 * sizeof(int), gpuMemcpyHostToDevice);
    }
    //printf("(%d) U solve=%d,%d, minGridSize=%d,myblockSize%d, nvshmem_kernel=%d,%d\n",
    //       mype,nbcol_loc,nthread_x*nthread_y,
    //       minGridSize,myblockSize,h_nfrecv_u[2],h_nfrecv_u[1]);
    //fflush(stdout);

    dim3 dimGrid_nv(h_nfrecv_u[2]); //2
    dim3 dimBlock_nv(h_nfrecv_u[1]); //1024



    void *args[] = {&nrhs, &URtree_ptr, &maxrecvsz, &mype, &flag_bc_q, &flag_rd_q, &zready_x, &zready_lsum,
            &my_flag_bc, &my_flag_rd,
            &d_nfrecv_u, &d_status,
            &d_colnum_u, &d_mynum_u, &d_mymaskstart_u, &d_mymasklength_u,
            &d_nfrecvmod_u, &d_statusmod, &d_colnummod_u, &d_mynummod_u, &d_mymaskstartmod_u,
            &d_mymasklengthmod_u,
            &d_recv_cnt_u, &d_msgnum, &d_flag_mod_u, &lsum, &bmod, &grid, &xsup, &ilsum, &nbrow_loc,
            &nsupers};

    int status = 1;
    status = nvshmemx_collective_launch((const void *) zwait_bcrd_u, dimGrid_nv, dimBlock_nv, args, 0, stream[0]);
    //status1 = nvshmemx_collective_launch((const void *) send_rd, dimGrid_rd, dimBlock_bc, args, 0, stream[1]);
    //printf("(%d), status=%d,%d\n",mype, status,status1);
    //fflush(stdout);

    if ((status != NVSHMEMX_SUCCESS)) {
        fprintf(stderr, "shmemx_collective_launch U failed %d\n", status);
        exit(-1);
    } else {
        zlsum_bmod_inv_gpu_mrhs_nvshmem<<< dimGrid, dimBlock, 0, stream[1] >>>(nbcol_loc,
                                                                    lsum, x,
                                                                    nrhs, nsupers, bmod,
                                                                    UBtree_ptr, URtree_ptr,
                                                                    ilsum,
                                                                    Ucolind_bc_dat, Ucolind_bc_offset,
                                                                    Unzval_bc_dat, Unzval_bc_offset,
                                                                    Uinv_bc_dat, Uinv_bc_offset,
                                                                    Uindval_loc_bc_dat,
                                                                    Uindval_loc_bc_offset,
                                                                    xsup, grid,
                                                                    maxrecvsz,
                                                                    mype, flag_bc_q,
                                                                    flag_rd_q,
                                                                    zready_x, zready_lsum,
                                                                    my_flag_bc, my_flag_rd,
                                                                    d_nfrecv_u, d_status,
                                                                    d_statusmod, nblock_ex,
                                                                    maxsuper, d_flag_mod_u); //temp2_offset, temp2,maxsuper);
        CUDA_CHECK(cudaGetLastError());
    }

    //CUDA_CHECK(cudaGetLastError());
    //CUDA_CHECK(cudaDeviceSynchronize());
    //zlsum_bmod_inv_gpu_mrhs<<< nbcol_loc, dimBlock >>>(nbcol_loc,lsum,x,nrhs,nsupers,bmod, UBtree_ptr,URtree_ptr,ilsum,Urbs,Ufstnz_br_dat,Ufstnz_br_offset,Unzval_br_dat,Unzval_br_offset,Ucb_valdat,Ucb_valoffset,Ucb_inddat,Ucb_indoffset,Uinv_bc_dat,Uinv_bc_offset,xsup,grid);
    gpuDeviceSynchronize();
    //printf("(%d) back to CPU !!!!! \n",mype);
    //}
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    #else
    printf("NVSHMEM is needed for multi-GPU solve\n");
    exit(1);
    #endif
    }
}



#ifdef __cplusplus
}
#endif
