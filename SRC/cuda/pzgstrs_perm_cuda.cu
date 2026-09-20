
#include <math.h>
#include "superlu_zdefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

#ifndef MAXSUPER
#define MAXSUPER 1024
#endif

#ifdef HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <cooperative_groups.h>
#include <nvml.h>
#endif

/* Typed helpers for templated scalar/complex GPU code.
 * Complex precisions use SuperLU_DIST's plain struct complex types;
 * assignment/copy is valid, but casts/arithmetic are not.
 */
typedef double pxgstrs_real_t;
static __host__ __device__ inline doublecomplex pxgstrs_type_zero(void) { doublecomplex z; z.r = 0.0; z.i = 0.0; return z; }
static __host__ __device__ inline doublecomplex pxgstrs_type_from_int(int_t v) { doublecomplex z; z.r = (double)v; z.i = 0.0; return z; }
static __host__ __device__ inline int pxgstrs_type_to_int(doublecomplex v) { return (int)v.r; }
static __host__ __device__ inline doublecomplex pxgstrs_type_scale_real(doublecomplex v, pxgstrs_real_t s) { v.r *= s; v.i *= s; return v; }
static __host__ __device__ inline doublecomplex pxgstrs_type_add(doublecomplex a, doublecomplex b) { a.r += b.r; a.i += b.i; return a; }
#define pxgstrs_nvshmem_put_nbi(dst, src, count, pe) \
    nvshmem_double_put_nbi((double*)(dst), (const double*)(src), 2*(count), (pe))


// This now only deals with 1 MPI case
__global__ void
pzReDistribute_B_to_X_gpu_proc1(doublecomplex *B, int_t m_loc, int nrhs, int_t ldb,
                          int_t fst_row, int_t *ilsum, doublecomplex *x,
		                    int_t *perm_r, int_t *perm_c,
                          int_t *xsup, int_t *supno, gridinfo_t *grid, int_t row_per_th)
{
  int_t  i, irow, j, k, knsupc, l, crow;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM PERMUTATION AND REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < m_loc) {
        irow = perm_c[perm_r[crow+fst_row]];
        k = BlockNum(irow);
        l = X_BLK(k);
        x[l - XK_H] = pxgstrs_type_from_int(k);      /* Block number prepended in the header. */

        knsupc = SuperSize(k);
        irow = irow - FstBlockC(k); /* Relative row number in X-block */
        for (j = 0; j < nrhs; j++) {
           x[l + irow + j*knsupc] = B[j*ldb+crow];
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pzReDistribute_B_to_X_gpu_send(doublecomplex *B, int_t m_loc, int nrhs, int_t ldb,
                          int_t *perm_r, int_t *perm_c,
                          int_t fst_row, int_t *ilsum, doublecomplex *send_idbuf,
                          int_t *xsup, int_t *supno, gridinfo_t *grid, int *ptr_to_idbuf, int_t row_per_th)
{
  int_t  i, irow, j, k, gbi, crow;
  int p;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM PERMUTATION AND REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < m_loc) {
        irow = perm_c[perm_r[crow+fst_row]];
        gbi = BlockNum(irow);
        p = PNUM( PROW(gbi,grid), PCOL(gbi,grid), grid ); /* Diagonal process */

        k = atomicAdd(&ptr_to_idbuf[p], nrhs+1);
        send_idbuf[k] = pxgstrs_type_from_int(irow);
        // printf("i = %d, and send_idbuf[%d] is %d.\n", i, k, (int) send_idbuf[k]);
        for (j = 0; j < nrhs; j++) {
           send_idbuf[k+j+1] = B[j*ldb+crow];
           // printf("i = %d, and send_idbuf[%d] is %f.\n", i, k+j+1, send_idbuf[k+j+1]);
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pzReDistribute_B_to_X_gpu_recv(doublecomplex *x, int nrhs, int_t recvl,
                          int_t fst_row, int_t *ilsum, doublecomplex *recv_idbuf,
                          int_t *xsup, int_t *supno, gridinfo_t *grid, int_t row_per_th)
{
  int_t  i, j, k, knsupc, l, lk, crow;
  int irow;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM PERMUTATION AND REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < recvl) {
        irow = pxgstrs_type_to_int(recv_idbuf[crow*(nrhs+1)]); /* The permuted row index. */
        k = BlockNum(irow);
        knsupc = SuperSize(k);
        lk = LBi(k, grid);  /* Local block number. */
        l = X_BLK(lk);
        x[l - XK_H] = pxgstrs_type_from_int(k);      /* Block number prepended in the header. */
        // printf("i = %d, recv_idbuf[i*(nrhs+1)] is %f, irow = %d, and x[%d] is %d.\n", i, recv_idbuf[i*(nrhs+1)], irow, l-XK_H, (int) x[l-XK_H]);

        irow = irow - FstBlockC(k); /* Relative row number in X-block */
        for (j = 0; j < nrhs; j++) {
           x[l + irow + j*knsupc] = recv_idbuf[crow*(nrhs+1)+j+1];
           // printf("i = %d, and send_idbuf[%d] is %f.\n", i, k+j+1, send_idbuf[k+j+1]);
        }
     }
  }
  __syncthreads();

  return;
}
void pzReDistribute_B_to_X_gpu_wrap(doublecomplex *B, int_t m_loc, int_t n, int nrhs, int_t ldb,
    int_t fst_row,
    doublecomplex *d_x,
    zScalePermstruct_t *ScalePermstruct,
    zSOLVEstruct_t *SOLVEstruct,
    Glu_persist_t *Glu_persist,
    gridinfo_t *grid, gridinfo_t *d_grid, int_t *d_ilsum, int_t *d_xsup, int_t *d_supno)
 {

 double t = SuperLU_timer_();
 /* ------------------------------------------------------------
 Host metadata.
 ------------------------------------------------------------ */
 int *d_perm_r = ScalePermstruct->d_perm_r;
 int *d_perm_c = ScalePermstruct->d_perm_c;
 int *d_ptr_to_idbuf_B2X =SOLVEstruct-> d_ptr_to_idbuf_B2X;

 pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
 int procs = grid->nprow * grid->npcol;
 int iam   = grid->iam;

 int *SendCnt      = gstrs_comm->B_to_X_SendCnt;
 int *SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt +     procs;
 int *RecvCnt      = gstrs_comm->B_to_X_SendCnt + 2 * procs;
 int *RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
 int *sdispls      = gstrs_comm->B_to_X_SendCnt + 4 * procs;
 int *rdispls      = gstrs_comm->B_to_X_SendCnt + 6 * procs;

 int *SendCnt_new  = (int*) SUPERLU_MALLOC((size_t)procs * sizeof(int));
 int *sdispls_new  = (int*) SUPERLU_MALLOC((size_t)procs * sizeof(int));
 int *rdispls_new  = (int*) SUPERLU_MALLOC((size_t)procs * sizeof(int));

 if (!SendCnt_new || !sdispls_new || !rdispls_new)
 ABORT("Malloc fails for communication metadata.");

 for (int p = 0; p < procs; ++p) {
 SendCnt_new[p]  = SendCnt_nrhs[p] + SendCnt[p];
 sdispls_new[p]  = sdispls[p] * (nrhs + 1);
 rdispls_new[p]  = rdispls[p] * (nrhs + 1);
 }

 /* Total local send/recv record counts before expanding by (nrhs+1). */
 int_t k = (procs > 0) ? (sdispls[procs - 1] + SendCnt[procs - 1]) : 0;
 int_t l = (procs > 0) ? (rdispls[procs - 1] + RecvCnt[procs - 1]) : 0;

 size_t send_elems = (size_t)k * ((size_t)nrhs + 1);
 size_t recv_elems = (size_t)l * ((size_t)nrhs + 1);


 checkGPU(gpuMemcpy(SOLVEstruct->d_ptr_to_idbuf_B2X, SOLVEstruct->d_ptr_to_idbuf_B2X_save,
     (size_t)procs * sizeof(int), gpuMemcpyDeviceToDevice));

 /* ------------------------------------------------------------
 Device data.
 ------------------------------------------------------------ */
 doublecomplex *d_B = B;

 doublecomplex *d_send_idbuf_B2X = NULL;
 doublecomplex *d_recv_idbuf_B2X = NULL;


 int_t nthreads = 256;
 int_t row_per_th = 16;
 int_t nblocks = (m_loc + nthreads * row_per_th - 1) / (nthreads * row_per_th);



 if (procs == 1) {
    if (m_loc > 0) {
    pzReDistribute_B_to_X_gpu_proc1<<<nblocks, nthreads>>>(
    d_B, m_loc, nrhs, ldb, fst_row, d_ilsum, d_x,
    d_perm_r, d_perm_c, d_xsup, d_supno, d_grid, row_per_th);

    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
    }
#if ( PROFlevel>=1 )
    t = SuperLU_timer_() - t;
    if (!grid->iam)
    printf(".. B to X GPU redistribute time \t%8.6f\n", t);
    fflush(stdout);
#endif

 } else {
    /* ------------------------------------------------------------
    If NVSHMEM is used for the solve, we cannot use cuda-aware MPI here and need to stick with NVSHMEM.
    ------------------------------------------------------------ */
#ifdef HAVE_NVSHMEM
    size_t send_alloc_elems = send_elems ? send_elems : 1;
    size_t recv_alloc_elems = recv_elems ? recv_elems : 1;

    d_send_idbuf_B2X = (doublecomplex*) nvshmem_malloc(send_alloc_elems * sizeof(doublecomplex));
    d_recv_idbuf_B2X = (doublecomplex*) nvshmem_malloc(recv_alloc_elems * sizeof(doublecomplex));

    if (!d_send_idbuf_B2X || !d_recv_idbuf_B2X)
    ABORT("nvshmem_malloc fails for B_to_X send/recv buffers.");

    checkGPU(gpuMemset(d_send_idbuf_B2X, 0, send_alloc_elems * sizeof(doublecomplex)));
    checkGPU(gpuMemset(d_recv_idbuf_B2X, 0, recv_alloc_elems * sizeof(doublecomplex)));

    /*
    Metadata exchange for remote receive offsets.

    After Alltoall, on source PE iam:
    remote_rdisp_for_dst_h[dst] == rdispls_new[iam] on PE dst
    which is exactly where iam must write in dst's d_recv_idbuf.
    */
    int *remote_rdisp_for_dst_h = (int*) SUPERLU_MALLOC((size_t)procs * sizeof(int));
    if (!remote_rdisp_for_dst_h)
    ABORT("Malloc fails for remote_rdisp_for_dst_h[].");

    MPI_Alltoall(rdispls_new, 1, MPI_INT,
    remote_rdisp_for_dst_h, 1, MPI_INT,
    grid->comm);

    /* ------------------------------------------------------------
    Fill local send buffer on GPU.
    ------------------------------------------------------------ */
    nvshmem_barrier_all();

    if (m_loc > 0) {
    pzReDistribute_B_to_X_gpu_send<<<nblocks, nthreads>>>(
    d_B, m_loc, nrhs, ldb, d_perm_r, d_perm_c,
    fst_row, d_ilsum, d_send_idbuf_B2X, d_xsup, d_supno,
    d_grid, d_ptr_to_idbuf_B2X, row_per_th);

    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
    }

    /* ------------------------------------------------------------
    Communication: NVSHMEM one-sided put or CUDA-aware MPI.
    ------------------------------------------------------------ */

    nvshmem_barrier_all();

    for (int pp = 0; pp < procs; ++pp) {
    int dst = iam + 1 + pp;
    if (dst >= procs) dst -= procs;

    if (SendCnt_new[dst] > 0) {
    int remote_offset = remote_rdisp_for_dst_h[dst];

    pxgstrs_nvshmem_put_nbi(d_recv_idbuf_B2X + remote_offset,
          d_send_idbuf_B2X + sdispls_new[dst],
          SendCnt_new[dst],
          dst);
    }
    }

    nvshmem_quiet();
    nvshmem_barrier_all();


    /* ------------------------------------------------------------
    Unpack receive buffer into x on GPU.
    Avoid launching with zero blocks when l == 0.
    ------------------------------------------------------------ */
    if (l > 0) {
    int_t nblocks_new = (l + nthreads * row_per_th - 1) /
    (nthreads * row_per_th);

    pzReDistribute_B_to_X_gpu_recv<<<nblocks_new, nthreads>>>(
    d_x, nrhs, l, fst_row, d_ilsum, d_recv_idbuf_B2X,
    d_xsup, d_supno, d_grid, row_per_th);

    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
    }

    /* ------------------------------------------------------------
    Communication cleanup.
    ------------------------------------------------------------ */

    SUPERLU_FREE(remote_rdisp_for_dst_h);
    nvshmem_free(d_send_idbuf_B2X);
    nvshmem_free(d_recv_idbuf_B2X);

#if ( PROFlevel>=1 )
    t = SuperLU_timer_() - t;
    if (!grid->iam) {
    printf(".. B to X GPU NVSHMEM redistribute time\t%8.6f\n", t);
    }
    fflush(stdout);
#endif

#endif
 }



 SUPERLU_FREE(SendCnt_new);
 SUPERLU_FREE(sdispls_new);
 SUPERLU_FREE(rdispls_new);
 }


// This also only works for one MPI rank
__global__ void
pzReDistribute_X_to_B_gpu_proc1(int_t n, doublecomplex *B, int_t m_loc, int_t ldb,
                                int_t fst_row, int_t nrhs, doublecomplex *x, int_t *ilsum,
                                int_t *xsup, int_t *supno, gridinfo_t *grid, int_t row_per_th)
{
  int_t  i, ii, irow, icomp, j, k, knsupc, l, lk, crow;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < m_loc) {
        icomp = crow + fst_row;
        k = supno[icomp];

        knsupc = SuperSize(k);
        irow = FstBlockC(k);
        ii = icomp - irow;

        lk = LBi(k, grid); /* Local block number */
        l = X_BLK(lk);

        for (j = 0; j < nrhs; j++) {
           B[j*ldb+crow] = x[l + ii + j*knsupc];
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pzReDistribute_X_to_B_gpu_send(doublecomplex *x, int nrhs, int_t ldb, int_t sendk, int_t *row_to_proc,
                          int_t num_diag_procs, int_t *diag_procs, int_t nsupers, int_t m_loc, int_t n,
                          int_t fst_row, int_t *ilsum, doublecomplex *send_idbuf,
                          int_t *xsup, int_t *supno, gridinfo_t *grid, int *ptr_to_idbuf, int_t row_per_th)
{
  int_t  i, irow, j, k, ii, jj, kk, knsupc, lk, l, crow;
  int p, pkk, q;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM PERMUTATION AND REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < n) {
        ii = crow;
        k = supno[ii];
        p = k % num_diag_procs;
        pkk = diag_procs[p];

        if (grid->iam == pkk) {
           knsupc = SuperSize(k);
           irow = FstBlockC(k);

           lk = LBi(k, grid); /* Local block number */
           l = X_BLK(lk);

           q = row_to_proc[ii];
           jj = atomicAdd(&ptr_to_idbuf[q], nrhs+1);

           send_idbuf[jj] = pxgstrs_type_from_int(ii);
           // printf("On processor %d, crow = %d, p = %d, ii = %d, k = %d, irow = %d, and send_idbuf[%d] is %d.\n",
           //    grid->iam, crow, p, ii, k, irow, jj, (int) send_idbuf[jj]);
           kk = ii - irow;
           for (j = 0; j < nrhs; j++) {
              send_idbuf[jj+j+1] = x[l + kk + j*knsupc];
              // printf("index = %d, i = %d, and send_idbuf[%d] is %f.\n", index, i, jj+j+1, send_idbuf[jj+j+1]);
           }
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pzReDistribute_X_to_B_gpu_recv(doublecomplex *B, int_t m_loc, int nrhs, int_t recvl, int_t ldb,
                          int_t fst_row, int_t *ilsum, doublecomplex *recv_idbuf,
                          int_t *xsup, int_t *supno, gridinfo_t *grid, int_t row_per_th)
{
  int_t  i, j, crow;
  int irow;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  int_t bid = blockIdx_x;
  int_t tid = threadIdx_x;
  int_t bdim = blockDim_x;
  int_t row_offset = bid*bdim*row_per_th;

  /* ------------------------------------------------------------
     NOW PERFORM PERMUTATION AND REDISTRIBUTION.
     ------------------------------------------------------------*/
  for (i = 0; i < row_per_th; i++) {
     crow = row_offset + i*bdim + tid;
     if (crow < recvl) {
        irow = pxgstrs_type_to_int(recv_idbuf[crow*(nrhs+1)]); /* The permuted row index. */
        irow = irow - fst_row;

        for (j = 0; j < nrhs; j++) {
           B[irow + j*ldb] = recv_idbuf[crow*(nrhs+1)+j+1];
           // printf("On processor %d, i = %d, and B[%d] is %f.\n", grid->iam, i, irow+j*ldb, B[irow+j*ldb]);
        }
     }
  }
  __syncthreads();

  return;
}


void pzReDistribute_X_to_B_gpu_wrap(doublecomplex *B, int_t m_loc, int_t n, int nrhs, int_t ldb,
    int_t fst_row,
    int_t nsupers, doublecomplex *d_x,
    zScalePermstruct_t *ScalePermstruct, zSOLVEstruct_t *SOLVEstruct,
    Glu_persist_t *Glu_persist, gridinfo_t *grid, gridinfo_t *d_grid, int_t *d_ilsum, int_t *d_xsup, int_t *d_supno)
 {

 /* ------------------------------------------------------------
 DECLARATION.
 ------------------------------------------------------------*/
 int_t num_diag_procs = SOLVEstruct->num_diag_procs;

 int_t *d_row_to_proc = SOLVEstruct->d_row_to_proc;
 int_t *d_diag_procs = SOLVEstruct->d_diag_procs;

 doublecomplex *d_B = B;


 int *d_ptr_to_idbuf_X2B = SOLVEstruct->d_ptr_to_idbuf_X2B;

 doublecomplex *d_send_idbuf_X2B = NULL;
 doublecomplex *d_recv_idbuf_X2B = NULL;

 double t;

 /* ------------------------------------------------------------
 MPI / communication metadata.
 ------------------------------------------------------------*/
 int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
 int *SendCnt_new;
 int *sdispls, *rdispls, *sdispls_new, *rdispls_new;
 int p, pp, pps;
 int procs = grid->nprow * grid->npcol;
 int iam = grid->iam;

 pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;





 /* ------------------------------------------------------------
 GPU DEVICE RUNNING.
 ------------------------------------------------------------*/
 int_t nthreads = 256;
 int_t row_per_th = 16;
 int_t nblocks = (m_loc + nthreads * row_per_th - 1) / (nthreads * row_per_th);

 if (procs == 1) {
    t = SuperLU_timer_();

    pzReDistribute_X_to_B_gpu_proc1<<<nblocks, nthreads>>>(
    n, d_B, m_loc, ldb, fst_row, nrhs, d_x,
    d_ilsum, d_xsup, d_supno, d_grid, row_per_th);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());

#if ( PROFlevel>=1 )
    t = SuperLU_timer_() - t;
    if (!grid->iam)
    printf(".. X to B GPU redistribute time \t%8.6f\n", t);
    fflush(stdout);
#endif

 } else {
    #ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(SOLVEstruct->d_ptr_to_idbuf_X2B, SOLVEstruct->d_ptr_to_idbuf_X2B_save,
        (size_t)procs * sizeof(int), gpuMemcpyDeviceToDevice));

    /* ------------------------------------------------------------
    COMM INITIALIZATION.
    ------------------------------------------------------------*/
    SendCnt      = gstrs_comm->X_to_B_SendCnt;
    SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt +     procs;
    RecvCnt      = gstrs_comm->X_to_B_SendCnt + 2 * procs;
    RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
    sdispls      = gstrs_comm->X_to_B_SendCnt + 4 * procs;
    rdispls      = gstrs_comm->X_to_B_SendCnt + 6 * procs;

    if (!(SendCnt_new = (int*) SUPERLU_MALLOC(procs * sizeof(int))))
    ABORT("Malloc fails for SendCnt_new[].");
    if (!(sdispls_new = (int*) SUPERLU_MALLOC(procs * sizeof(int))))
    ABORT("Malloc fails for sdispls_new[].");
    if (!(rdispls_new = (int*) SUPERLU_MALLOC(procs * sizeof(int))))
    ABORT("Malloc fails for rdispls_new[].");

    for (p = 0; p < procs; ++p) {
    SendCnt_new[p]  = SendCnt_nrhs[p] + SendCnt[p];
    sdispls_new[p]  = sdispls[p] * (nrhs + 1);
    rdispls_new[p]  = rdispls[p] * (nrhs + 1);
    }

    MPI_Barrier(grid->comm);
    t = SuperLU_timer_();
    int_t k = sdispls[procs - 1] + SendCnt[procs - 1];
    int_t l = rdispls[procs - 1] + RecvCnt[procs - 1];

    size_t send_elems = (size_t) k * ((size_t) nrhs + 1);
    size_t recv_elems = (size_t) l * ((size_t) nrhs + 1);
    if (send_elems == 0) send_elems = 1;
    if (recv_elems == 0) recv_elems = 1;


    /*
    NVSHMEM allocations are collective and symmetric.  Use the global
    maximum local buffer sizes.  This is still much smaller than
    n*(nrhs+1), but avoids non-uniform symmetric object sizes.
    */
    unsigned long long send_elems_ull = (unsigned long long) send_elems;
    unsigned long long recv_elems_ull = (unsigned long long) recv_elems;
    unsigned long long max_send_elems_ull = 0;
    unsigned long long max_recv_elems_ull = 0;

    MPI_Allreduce(&send_elems_ull, &max_send_elems_ull,
    1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
    MPI_Allreduce(&recv_elems_ull, &max_recv_elems_ull,
    1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);

    size_t alloc_send_elems = (size_t) max_send_elems_ull;
    size_t alloc_recv_elems = (size_t) max_recv_elems_ull;

    d_send_idbuf_X2B = (doublecomplex*) nvshmem_malloc(alloc_send_elems * sizeof(doublecomplex));
    d_recv_idbuf_X2B = (doublecomplex*) nvshmem_malloc(alloc_recv_elems * sizeof(doublecomplex));
    if (!d_send_idbuf_X2B || !d_recv_idbuf_X2B)
    ABORT("nvshmem_malloc fails for X_to_B send/recv buffers.");



    checkGPU(gpuMemset(d_send_idbuf_X2B, 0, alloc_send_elems * sizeof(doublecomplex)));
    checkGPU(gpuMemset(d_recv_idbuf_X2B, 0, alloc_recv_elems * sizeof(doublecomplex)));


    /* ------------------------------------------------------------
    FILL IN THE SEND BUFFERS ON GPU.
    ------------------------------------------------------------*/

    int_t nblocks_send = (n + nthreads * row_per_th - 1) / (nthreads * row_per_th);
    if (n > 0) {
    pzReDistribute_X_to_B_gpu_send<<<nblocks_send, nthreads>>>(
    d_x, nrhs, ldb, k, d_row_to_proc, num_diag_procs, d_diag_procs,
    nsupers, m_loc, n, fst_row, d_ilsum, d_send_idbuf_X2B,
    d_xsup, d_supno, d_grid, d_ptr_to_idbuf_X2B, row_per_th);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
    }


    /* ------------------------------------------------------------
    NVSHMEM COMMUNICATION.

    MPI receive layout was:
    source src lands at d_recv_idbuf + rdispls_new[src]

    With one-sided NVSHMEM, sender must know destination's offset.
    After Alltoall, on rank iam:
    remote_rdisp_for_dst_h[dst] == rdispls_new[iam] on rank dst.
    ------------------------------------------------------------*/
    int *remote_rdisp_for_dst_h =
    (int*) SUPERLU_MALLOC(procs * sizeof(int));
    if (!remote_rdisp_for_dst_h)
    ABORT("Malloc fails for remote_rdisp_for_dst_h[].");

    MPI_Alltoall(rdispls_new, 1, MPI_INT,
    remote_rdisp_for_dst_h, 1, MPI_INT,
    grid->comm);

    nvshmem_barrier_all();

    for (pp = 0; pp < procs; ++pp) {
    pps = iam + 1 + pp;
    if (pps >= procs) pps -= procs;
    if (pps < 0) pps += procs;

    if (SendCnt_new[pps] > 0) {
    int remote_offset = remote_rdisp_for_dst_h[pps];

    pxgstrs_nvshmem_put_nbi(d_recv_idbuf_X2B + remote_offset,
          d_send_idbuf_X2B + sdispls_new[pps],
          SendCnt_new[pps],
          pps);
    }
    }

    nvshmem_quiet();
    nvshmem_barrier_all();

    SUPERLU_FREE(remote_rdisp_for_dst_h);


    /* ------------------------------------------------------------
    FILL IN B FROM RECEIVE BUFFERS ON GPU.
    ------------------------------------------------------------*/
    if (l > 0) {
    int_t nblocks_recv = (l + nthreads * row_per_th - 1) / (nthreads * row_per_th);
    pzReDistribute_X_to_B_gpu_recv<<<nblocks_recv, nthreads>>>(
    d_B, m_loc, nrhs, l, ldb, fst_row, d_ilsum, d_recv_idbuf_X2B,
    d_xsup, d_supno, d_grid, row_per_th);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
    }

#if ( PROFlevel>=1 )
    t = SuperLU_timer_() - t;
    if (!grid->iam) {
       printf(".. X to B NVSHMEM redistribute time\t%8.6f\n", t);
    }
    fflush(stdout);
 #endif

    /* ------------------------------------------------------------
    CLEANING COMMUNICATION BUFFERS.
    ------------------------------------------------------------*/
    nvshmem_free(d_send_idbuf_X2B);
    nvshmem_free(d_recv_idbuf_X2B);
    SUPERLU_FREE(SendCnt_new);
    SUPERLU_FREE(sdispls_new);
    SUPERLU_FREE(rdispls_new);
    #endif
 }


 }





__global__ void
pzPermute_Dense_Matrix_gpu_send_kernel(const doublecomplex *X, int_t m_loc, int_t fst_row,
                                       const int_t *row_to_proc, const int *perm,
                                       int_t ldx, int nrhs,
                                       doublecomplex *send_idbuf, int *ptr_to_idbuf,
                                       int_t row_per_th)
{
    int_t bid  = blockIdx.x;
    int_t tid  = threadIdx.x;
    int_t bdim = blockDim.x;
    int_t row_offset = bid * bdim * row_per_th;

    for (int_t rr = 0; rr < row_per_th; ++rr) {
        int_t local_i = row_offset + rr * bdim + tid;
        if (local_i < m_loc) {
            int_t global_i = fst_row + local_i;
            int_t dest_row = (int_t) perm[global_i];
            int p = (int) row_to_proc[dest_row];

            int off = atomicAdd(&ptr_to_idbuf[p], nrhs + 1);
            send_idbuf[off] = pxgstrs_type_from_int(dest_row);

            for (int rhs = 0; rhs < nrhs; ++rhs) {
                send_idbuf[off + 1 + rhs] = X[local_i + (int_t)rhs * ldx];
            }
        }
    }
}

__global__ void
pzPermute_Dense_Matrix_gpu_recv_kernel(doublecomplex *B, int_t m_loc, int_t fst_row,
                                       int_t ldb, int nrhs,
                                       const doublecomplex *recv_idbuf, int_t recvl,
                                       int_t row_per_th)
{
    int_t bid  = blockIdx.x;
    int_t tid  = threadIdx.x;
    int_t bdim = blockDim.x;
    int_t row_offset = bid * bdim * row_per_th;

    for (int_t rr = 0; rr < row_per_th; ++rr) {
        int_t rec_i = row_offset + rr * bdim + tid;
        if (rec_i < recvl) {
            int_t dest_row = (int_t) pxgstrs_type_to_int(recv_idbuf[rec_i * ((int_t)nrhs + 1)]);
            int_t local_j = dest_row - fst_row;

            /* Optional safety; remove if you want zero branch overhead. */
            if (local_j >= 0 && local_j < m_loc) {
                for (int rhs = 0; rhs < nrhs; ++rhs) {
                    B[local_j + (int_t)rhs * ldb] =
                        recv_idbuf[rec_i * ((int_t)nrhs + 1) + 1 + rhs];
                }
            }
        }
    }
}

__global__ void
pzPermute_Dense_Matrix_gpu_proc1(const doublecomplex *X, int_t m_loc, int_t fst_row,
                                 const int_t *perm, int_t ldx,
                                 doublecomplex *B, int_t ldb, int nrhs,
                                 int_t row_per_th)
{
    int_t bid  = blockIdx.x;
    int_t tid  = threadIdx.x;
    int_t bdim = blockDim.x;
    int_t row_offset = bid * bdim * row_per_th;

    for (int_t rr = 0; rr < row_per_th; ++rr) {
        int_t local_i = row_offset + rr * bdim + tid;
        if (local_i < m_loc) {
            int_t global_i = fst_row + local_i;
            int_t dest_row = perm[global_i];
            int_t local_j  = dest_row - fst_row;

            /* For procs == 1 this should always be local, but keep the guard. */
            if (local_j >= 0 && local_j < m_loc) {
                for (int rhs = 0; rhs < nrhs; ++rhs) {
                    B[local_j + (int_t)rhs * ldb] =
                        X[local_i + (int_t)rhs * ldx];
                }
            }
        }
    }
}

int pzPermute_Dense_Matrix_gpu_wrap(int_t fst_row,
    int_t m_loc,
    int_t n,
    doublecomplex *d_X,
    int_t ldx,
    doublecomplex *d_B,
    int_t ldb,
    int nrhs,
    gridinfo_t *grid, zSOLVEstruct_t *SOLVEstruct)
{
    int_t i;
    int procs = grid->nprow * grid->npcol;
    int iam   = grid->iam;

    int_t *row_to_proc = SOLVEstruct->row_to_proc;
    int_t *d_row_to_proc = SOLVEstruct->d_row_to_proc;
    int_t *perm = SOLVEstruct->inv_perm_c;
    int_t *d_perm = SOLVEstruct->d_inv_perm_c;
    int *d_ptr_to_idbuf_PermuteC = SOLVEstruct->d_ptr_to_idbuf_PermuteC;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Enter pzPermute_Dense_Matrix_gpu_wrap()");
#endif

    int_t nthreads = 256;
    int_t row_per_th = 16;
    int_t nblocks = (m_loc + nthreads * row_per_th - 1) /
                    (nthreads * row_per_th);

    double t = SuperLU_timer_();

    /*
     * Fast path for a single process.
     * No send/recv buffers or communication are needed.  The CPU routine does:
     *   j = perm[i];
     *   B[j - fst_row + rhs*ldb] = X[i - fst_row + rhs*ldx];
     * Here local_i = i - fst_row.
     */
    if (procs == 1) {
        if (m_loc > 0) {
            pzPermute_Dense_Matrix_gpu_proc1<<<nblocks, nthreads>>>(
                d_X, m_loc, fst_row, d_perm, ldx,
                d_B, ldb, nrhs, row_per_th);

            checkGPU(gpuGetLastError());
            checkGPU(gpuDeviceSynchronize());
        }

#if ( PROFlevel>=1 )
        t = SuperLU_timer_() - t;
        if (!grid->iam) {
            printf(".. Permute Dense Matrix GPU proc1 time\t%8.6f\n", t);
        }
        fflush(stdout);
#endif

#if ( DEBUGlevel>=1 )
        CHECK_MALLOC(grid->iam, "Exit pzPermute_Dense_Matrix_gpu_wrap()");
#endif
    }else{
#ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(d_ptr_to_idbuf_PermuteC,
                       SOLVEstruct->d_ptr_to_idbuf_PermuteC_save,
                       (size_t)procs * sizeof(int),
                       gpuMemcpyDeviceToDevice));

    int *sendcnts = (int*) SUPERLU_MALLOC((size_t)7 * procs * sizeof(int));
    if (!sendcnts) ABORT("Malloc fails for sendcnts[].");

    int *sendcnts_nrhs = sendcnts + procs;
    int *recvcnts      = sendcnts_nrhs + procs;
    int *sdispls       = recvcnts + procs;
    int *sdispls_nrhs  = sdispls + procs;
    int *rdispls       = sdispls_nrhs + procs;
    int *rdispls_nrhs  = rdispls + procs;

    for (i = 0; i < procs; ++i) sendcnts[i] = 0;

    /* CPU count pass; same as original pzPermute_Dense_Matrix. */
    for (i = fst_row; i < fst_row + m_loc; ++i) {
        int p = (int) row_to_proc[perm[i]];
        ++sendcnts[p];
    }

    MPI_Alltoall(sendcnts, 1, MPI_INT,
                 recvcnts, 1, MPI_INT,
                 grid->comm);

    sdispls[0] = rdispls[0] = 0;
    sdispls_nrhs[0] = rdispls_nrhs[0] = 0;
    sendcnts_nrhs[0] = sendcnts[0] * (nrhs + 1);

    for (i = 1; i < procs; ++i) {
        sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
        sdispls_nrhs[i] = sdispls[i] * (nrhs + 1);
        rdispls_nrhs[i] = rdispls[i] * (nrhs + 1);
        sendcnts_nrhs[i] = sendcnts[i] * (nrhs + 1);
    }

    int_t k = sdispls[procs - 1] + sendcnts[procs - 1]; /* send records */
    int_t l = rdispls[procs - 1] + recvcnts[procs - 1]; /* recv records */

    size_t send_elems = (size_t)k * ((size_t)nrhs + 1);
    size_t recv_elems = (size_t)l * ((size_t)nrhs + 1);
    size_t send_alloc_elems = send_elems ? send_elems : 1;
    size_t recv_alloc_elems = recv_elems ? recv_elems : 1;

    doublecomplex *d_send_idbuf_PermuteC = NULL;
    doublecomplex *d_recv_idbuf_PermuteC = NULL;


    d_send_idbuf_PermuteC = (doublecomplex*) nvshmem_malloc(send_alloc_elems * sizeof(doublecomplex));
    d_recv_idbuf_PermuteC = (doublecomplex*) nvshmem_malloc(recv_alloc_elems * sizeof(doublecomplex));
    if (!d_send_idbuf_PermuteC || !d_recv_idbuf_PermuteC)
        ABORT("nvshmem_malloc fails for pzPermute_Dense_Matrix buffers.");


    checkGPU(gpuMemset(d_send_idbuf_PermuteC, 0,
                       send_alloc_elems * sizeof(doublecomplex)));
    checkGPU(gpuMemset(d_recv_idbuf_PermuteC, 0,
                       recv_alloc_elems * sizeof(doublecomplex)));


    nvshmem_barrier_all();


    if (m_loc > 0) {
        pzPermute_Dense_Matrix_gpu_send_kernel<<<nblocks, nthreads>>>(
            d_X, m_loc, fst_row, d_row_to_proc, d_perm,
            ldx, nrhs, d_send_idbuf_PermuteC,
            d_ptr_to_idbuf_PermuteC, row_per_th);

        checkGPU(gpuGetLastError());
        checkGPU(gpuDeviceSynchronize());
    }

    /*
     * Metadata exchange for remote receive offsets.
     * After Alltoall, on source PE iam:
     *   remote_rdisp_for_dst_h[dst] == rdispls_nrhs[iam] on destination PE dst.
     */
    int *remote_rdisp_for_dst_h =
        (int*) SUPERLU_MALLOC((size_t)procs * sizeof(int));
    if (!remote_rdisp_for_dst_h)
        ABORT("Malloc fails for remote_rdisp_for_dst_h[].");

    MPI_Alltoall(rdispls_nrhs, 1, MPI_INT,
                 remote_rdisp_for_dst_h, 1, MPI_INT,
                 grid->comm);

    nvshmem_barrier_all();

    for (int pp = 0; pp < procs; ++pp) {
        int dst = iam + 1 + pp;
        if (dst >= procs) dst -= procs;

        if (sendcnts_nrhs[dst] > 0) {
            int remote_offset = remote_rdisp_for_dst_h[dst];

            pxgstrs_nvshmem_put_nbi(d_recv_idbuf_PermuteC + remote_offset,
                                   d_send_idbuf_PermuteC + sdispls_nrhs[dst],
                                   sendcnts_nrhs[dst],
                                   dst);
        }
    }

    nvshmem_quiet();
    nvshmem_barrier_all();

    SUPERLU_FREE(remote_rdisp_for_dst_h);


    if (l > 0) {
        int_t nblocks_recv = (l + nthreads * row_per_th - 1) /
                             (nthreads * row_per_th);

        pzPermute_Dense_Matrix_gpu_recv_kernel<<<nblocks_recv, nthreads>>>(
            d_B, m_loc, fst_row, ldb, nrhs,
            d_recv_idbuf_PermuteC, l, row_per_th);

        checkGPU(gpuGetLastError());
        checkGPU(gpuDeviceSynchronize());
    }

#if ( PROFlevel>=1 )
    t = SuperLU_timer_() - t;
    if (!grid->iam) {
        printf(".. Permute Dense Matrix GPU NVSHMEM time\t%8.6f\n", t);
    }
    fflush(stdout);
#endif

    nvshmem_free(d_send_idbuf_PermuteC);
    nvshmem_free(d_recv_idbuf_PermuteC);

    SUPERLU_FREE(sendcnts);
#endif
}
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Exit pzPermute_Dense_Matrix_gpu_wrap()");
#endif
    return 0;
}



__global__ void zscale_and_copy_rhs_kernel(
    doublecomplex *B, doublecomplex *X,
    const pxgstrs_real_t *R, const pxgstrs_real_t *C,
    int_t m_loc, int_t fst_row,
    int_t ldb, int_t ldx,
    int nrhs,
    int notran,
    int rowequ,
    int colequ)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m_loc * nrhs;

    if (tid >= total) return;

    int i = tid % m_loc;
    int j = tid / m_loc;

    int_t grow = fst_row + i;

    doublecomplex val = B[i + j * ldb];

    if (notran) {
        if (rowequ) val = pxgstrs_type_scale_real(val, R[grow]);
    } else {
        if (colequ) val = pxgstrs_type_scale_real(val, C[grow]);
    }

    B[i + j * ldb] = val;
    X[i + j * ldx] = val;
}



__global__ void zundo_equilibration_rhs_kernel(
    doublecomplex *B,
    const pxgstrs_real_t *R,
    const pxgstrs_real_t *C,
    int_t m_loc,
    int_t fst_row,
    int_t ldb,
    int nrhs,
    int notran,
    int rowequ,
    int colequ)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int_t total = m_loc * nrhs;

    if (tid >= total) return;

    int_t i = tid % m_loc;
    int_t j = tid / m_loc;
    int_t grow = fst_row + i;

    doublecomplex val = B[i + j * ldb];

    if (notran) {
        if (colequ) val = pxgstrs_type_scale_real(val, C[grow]);
    } else {
        if (rowequ) val = pxgstrs_type_scale_real(val, R[grow]);
    }

    B[i + j * ldb] = val;
}




void zscale_and_copy_rhs_wrap(doublecomplex *B, int_t ldb, doublecomplex *X, int_t ldx, int_t m_loc, int nrhs,
    int_t fst_row, int notran,int rowequ, int colequ,
    zScalePermstruct_t *ScalePermstruct)
 {

    int threads = 256;
    int_t total = m_loc * nrhs;
    int blocks = (total + threads - 1) / threads;

    if (total > 0) {
        if (rowequ==FALSE && colequ==FALSE){
            checkGPU(gpuMemcpy(X, B, sizeof(doublecomplex)*total, cudaMemcpyDeviceToDevice));
        }else{
            zscale_and_copy_rhs_kernel<<<blocks, threads>>>(
                B, X, ScalePermstruct->d_R, ScalePermstruct->d_C,
                m_loc, fst_row,
                ldb, ldx,
                nrhs,
                notran,
                rowequ,
                colequ);
        }

        checkGPU(gpuGetLastError());
        checkGPU(gpuDeviceSynchronize());
    }
 }



 void zundo_equilibration_rhs_wrap(doublecomplex *B, int_t ldb, int_t m_loc, int nrhs,
    int_t fst_row, int notran,int rowequ, int colequ,
    zScalePermstruct_t *ScalePermstruct)
 {
    int threads = 256;
    int_t total = m_loc * nrhs;
    int blocks = (total + threads - 1) / threads;

    if (total > 0) {
        if (rowequ==FALSE && colequ==FALSE){
        }else{
            zundo_equilibration_rhs_kernel<<<blocks, threads>>>(
                B, ScalePermstruct->d_R, ScalePermstruct->d_C,
                m_loc, fst_row,
                ldb, nrhs,
                notran, rowequ, colequ);

            checkGPU(gpuGetLastError());
            checkGPU(gpuDeviceSynchronize());
        }
    }
 }


__global__ void zdevice_matcopy_kernel(int_t m, int nrhs, doublecomplex *dst,
    int_t lddst, const doublecomplex *src, int_t ldsrc)
{
    int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int_t total = m * (int_t)nrhs;

    if (tid >= total) return;

    int_t i = tid % m;
    int_t j = tid / m;
    dst[i + j * lddst] = src[i + j * ldsrc];
}


void zdevice_matcopy_wrap(int_t m, int nrhs, doublecomplex *dst, int_t lddst,
    const doublecomplex *src, int_t ldsrc)
{
    int threads = 256;
    int_t total = m * (int_t)nrhs;
    int blocks = (total + threads - 1) / threads;

    if (total > 0) {
        zdevice_matcopy_kernel<<<blocks, threads>>>(m, nrhs, dst, lddst, src, ldsrc);
        checkGPU(gpuGetLastError());
        checkGPU(gpuDeviceSynchronize());
    }
}


__global__ void zdevice_add_to_vec_kernel(doublecomplex *dst, const doublecomplex *src, int_t n)
{
    int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = pxgstrs_type_add(dst[tid], src[tid]);
}


void zdevice_add_to_vec_wrap(doublecomplex *dst, const doublecomplex *src, int_t n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (n > 0) {
        zdevice_add_to_vec_kernel<<<blocks, threads>>>(dst, src, n);
        checkGPU(gpuGetLastError());
        checkGPU(gpuDeviceSynchronize());
    }
}


__global__ void ztrs_B_init3d_zero_inactive_kernel(doublecomplex *x, int nrhs,
    int_t nsupers, int_t nlb, const int_t *ilsum, const int_t *xsup,
    const SupernodeToGridMap_t *superGridMap, int nprow, int npcol,
    int myrow, int mycol)
{
    int_t lk = blockIdx.x;
    if (lk >= nlb) return;

    int_t k = lk * nprow + myrow;
    if (k >= nsupers || mycol != k % npcol ||
        superGridMap[k] == IN_GRID_AIJ) return;

    int_t knsupc = xsup[k + 1] - xsup[k];
    int_t ii = ilsum[lk] * nrhs + (lk + 1) * XK_H;
    int_t nvals = knsupc * (int_t)nrhs;

    for (int_t i = threadIdx.x; i < nvals; i += blockDim.x) {
        x[ii + i] = pxgstrs_type_zero();
    }
}


void ztrs_B_init3d_zero_inactive_gpu_wrap(doublecomplex *x, int nrhs,
    int_t nsupers, int_t nlb, const int_t *ilsum, const int_t *xsup,
    const SupernodeToGridMap_t *superGridMap, int nprow, int npcol,
    int myrow, int mycol)
{
    if (nlb <= 0) return;

    int threads = 256;
    ztrs_B_init3d_zero_inactive_kernel<<<(int)nlb, threads>>>(
        x, nrhs, nsupers, nlb, ilsum, xsup, superGridMap,
        nprow, npcol, myrow, mycol);
    checkGPU(gpuGetLastError());
}


__global__ void ztrs_X_gather3d_pack_kernel(doublecomplex *packbuf, const doublecomplex *x,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    int_t b = blockIdx.x;
    if (b >= nblocks) return;

    int_t src = offsets[b];
    int_t dst = pack_offsets[b];
    int_t len = lengths[b];
    for (int_t i = threadIdx.x; i < len; i += blockDim.x)
        packbuf[dst + i] = x[src + i];
}


__global__ void ztrs_X_gather3d_unpack_kernel(doublecomplex *x, const doublecomplex *packbuf,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    int_t b = blockIdx.x;
    if (b >= nblocks) return;

    int_t dst = offsets[b];
    int_t src = pack_offsets[b];
    int_t len = lengths[b];
    for (int_t i = threadIdx.x; i < len; i += blockDim.x)
        x[dst + i] = packbuf[src + i];
}


__global__ void ztrs_X_gather3d_pack_zero_kernel(doublecomplex *packbuf, doublecomplex *x,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    int_t b = blockIdx.x;
    if (b >= nblocks) return;

    int_t src = offsets[b];
    int_t dst = pack_offsets[b];
    int_t len = lengths[b];
    for (int_t i = threadIdx.x; i < len; i += blockDim.x) {
        packbuf[dst + i] = x[src + i];
        x[src + i] = pxgstrs_type_zero();
    }
}


__global__ void ztrs_X_gather3d_unpack_add_kernel(doublecomplex *x, const doublecomplex *packbuf,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    int_t b = blockIdx.x;
    if (b >= nblocks) return;

    int_t dst = offsets[b];
    int_t src = pack_offsets[b];
    int_t len = lengths[b];
    for (int_t i = threadIdx.x; i < len; i += blockDim.x)
        x[dst + i] = pxgstrs_type_add(x[dst + i], packbuf[src + i]);
}


void ztrs_X_gather3d_pack_gpu_wrap(doublecomplex *packbuf, const doublecomplex *x,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    if (nblocks <= 0) return;

    int threads = 256;
    ztrs_X_gather3d_pack_kernel<<<(int)nblocks, threads>>>(
        packbuf, x, offsets, lengths, pack_offsets, nblocks);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
}


void ztrs_X_gather3d_pack_zero_gpu_wrap(doublecomplex *packbuf, doublecomplex *x,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    if (nblocks <= 0) return;

    int threads = 256;
    ztrs_X_gather3d_pack_zero_kernel<<<(int)nblocks, threads>>>(
        packbuf, x, offsets, lengths, pack_offsets, nblocks);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
}


void ztrs_X_gather3d_unpack_add_gpu_wrap(doublecomplex *x, const doublecomplex *packbuf,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    if (nblocks <= 0) return;

    int threads = 256;
    ztrs_X_gather3d_unpack_add_kernel<<<(int)nblocks, threads>>>(
        x, packbuf, offsets, lengths, pack_offsets, nblocks);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
}


void ztrs_X_gather3d_unpack_gpu_wrap(doublecomplex *x, const doublecomplex *packbuf,
    const int_t *offsets, const int_t *lengths, const int_t *pack_offsets,
    int_t nblocks)
{
    if (nblocks <= 0) return;

    int threads = 256;
    ztrs_X_gather3d_unpack_kernel<<<(int)nblocks, threads>>>(
        x, packbuf, offsets, lengths, pack_offsets, nblocks);
    checkGPU(gpuGetLastError());
    checkGPU(gpuDeviceSynchronize());
}







/*! \brief Initialize the nvshmem data structure for the GPU-resident interfaces
 */
int zSolveInit_nvshmem_gpures(superlu_dist_options_t *options, int_t fst_row, int_t m_loc,
    int_t nrhs, int_t n, gridinfo_t *grid,zSOLVEstruct_t *SOLVEstruct)
{
int_t *row_to_proc=SOLVEstruct->row_to_proc;
int *inv_perm_c=SOLVEstruct->inv_perm_c;
int          procs = grid->nprow * grid->npcol;
int i;

#ifdef HAVE_NVSHMEM
{
int *SendCnt      = SOLVEstruct->gstrs_comm->B_to_X_SendCnt;
int *RecvCnt      = SOLVEstruct->gstrs_comm->B_to_X_SendCnt + 2 * procs;
int *sdispls      = SOLVEstruct->gstrs_comm->B_to_X_SendCnt + 4 * procs;
int *rdispls      = SOLVEstruct->gstrs_comm->B_to_X_SendCnt + 6 * procs;
/* Total local send/recv record counts before expanding by (nrhs+1). */
int_t k = (procs > 0) ? (sdispls[procs - 1] + SendCnt[procs - 1]) : 0;
int_t l = (procs > 0) ? (rdispls[procs - 1] + RecvCnt[procs - 1]) : 0;
size_t send_elems = (size_t)k * ((size_t)nrhs + 1);
size_t recv_elems = (size_t)l * ((size_t)nrhs + 1);
if (send_elems == 0) send_elems = 1;
if (recv_elems == 0) recv_elems = 1;
unsigned long long send_elems_ull = (unsigned long long) send_elems;
unsigned long long recv_elems_ull = (unsigned long long) recv_elems;
unsigned long long max_send_elems_ull = 0;
unsigned long long max_recv_elems_ull = 0;
MPI_Allreduce(&send_elems_ull, &max_send_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
MPI_Allreduce(&recv_elems_ull, &max_recv_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
size_t alloc_send_elems = (size_t) max_send_elems_ull;
size_t alloc_recv_elems = (size_t) max_recv_elems_ull;
SOLVEstruct->d_send_idbuf_B2X = (doublecomplex*) nvshmem_malloc(alloc_send_elems * sizeof(doublecomplex));
SOLVEstruct->d_recv_idbuf_B2X = (doublecomplex*) nvshmem_malloc(alloc_recv_elems * sizeof(doublecomplex));
}

{
pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
int *SendCnt      = gstrs_comm->X_to_B_SendCnt;
int *SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt +     procs;
int *RecvCnt      = gstrs_comm->X_to_B_SendCnt + 2 * procs;
int *RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
int *sdispls      = gstrs_comm->X_to_B_SendCnt + 4 * procs;
int *rdispls      = gstrs_comm->X_to_B_SendCnt + 6 * procs;
int_t k = sdispls[procs - 1] + SendCnt[procs - 1];
int_t l = rdispls[procs - 1] + RecvCnt[procs - 1];
size_t send_elems = (size_t) k * ((size_t) nrhs + 1);
size_t recv_elems = (size_t) l * ((size_t) nrhs + 1);
if (send_elems == 0) send_elems = 1;
if (recv_elems == 0) recv_elems = 1;
unsigned long long send_elems_ull = (unsigned long long) send_elems;
unsigned long long recv_elems_ull = (unsigned long long) recv_elems;
unsigned long long max_send_elems_ull = 0;
unsigned long long max_recv_elems_ull = 0;
MPI_Allreduce(&send_elems_ull, &max_send_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
MPI_Allreduce(&recv_elems_ull, &max_recv_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
size_t alloc_send_elems = (size_t) max_send_elems_ull;
size_t alloc_recv_elems = (size_t) max_recv_elems_ull;
SOLVEstruct->d_send_idbuf_X2B = (doublecomplex*) nvshmem_malloc(alloc_send_elems * sizeof(doublecomplex));
SOLVEstruct->d_recv_idbuf_X2B = (doublecomplex*) nvshmem_malloc(alloc_recv_elems * sizeof(doublecomplex));
if (!SOLVEstruct->d_send_idbuf_X2B || !SOLVEstruct->d_recv_idbuf_X2B)
ABORT("nvshmem_malloc fails for X_to_B send/recv buffers.");
}

{
int *sendcnts = (int*) SUPERLU_MALLOC((size_t)7 * procs * sizeof(int));
if (!sendcnts) ABORT("Malloc fails for sendcnts[].");

int *sendcnts_nrhs = sendcnts + procs;
int *recvcnts      = sendcnts_nrhs + procs;
int *sdispls       = recvcnts + procs;
int *sdispls_nrhs  = sdispls + procs;
int *rdispls       = sdispls_nrhs + procs;
int *rdispls_nrhs  = rdispls + procs;

for (i = 0; i < procs; ++i) sendcnts[i] = 0;

/* CPU count pass; same as original pzPermute_Dense_Matrix. */
for (i = fst_row; i < fst_row + m_loc; ++i) {
 int p = (int) row_to_proc[inv_perm_c[i]];
 ++sendcnts[p];
}

MPI_Alltoall(sendcnts, 1, MPI_INT,
          recvcnts, 1, MPI_INT,
          grid->comm);
sdispls[0] = rdispls[0] = 0;
for (i = 1; i < procs; ++i) {
 sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
 rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
}

int_t k = sdispls[procs - 1] + sendcnts[procs - 1]; /* send records */
int_t l = rdispls[procs - 1] + recvcnts[procs - 1]; /* recv records */

size_t send_elems = (size_t)k * ((size_t)nrhs + 1);
size_t recv_elems = (size_t)l * ((size_t)nrhs + 1);
if (send_elems == 0) send_elems = 1;
if (recv_elems == 0) recv_elems = 1;
unsigned long long send_elems_ull = (unsigned long long) send_elems;
unsigned long long recv_elems_ull = (unsigned long long) recv_elems;
unsigned long long max_send_elems_ull = 0;
unsigned long long max_recv_elems_ull = 0;
MPI_Allreduce(&send_elems_ull, &max_send_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
MPI_Allreduce(&recv_elems_ull, &max_recv_elems_ull,
1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, grid->comm);
size_t alloc_send_elems = (size_t) max_send_elems_ull;
size_t alloc_recv_elems = (size_t) max_recv_elems_ull;
SOLVEstruct->d_send_idbuf_PermuteC = (doublecomplex*) nvshmem_malloc(alloc_send_elems * sizeof(doublecomplex));
SOLVEstruct->d_recv_idbuf_PermuteC = (doublecomplex*) nvshmem_malloc(alloc_recv_elems * sizeof(doublecomplex));
if (!SOLVEstruct->d_send_idbuf_PermuteC || !SOLVEstruct->d_recv_idbuf_PermuteC)
 ABORT("nvshmem_malloc fails for pzPermute_Dense_Matrix buffers.");
SUPERLU_FREE(sendcnts);
}

#endif

return 0;
} /* zSolveInit_nvshmem_gpures */



void zFree_nvshmem_gpures(superlu_dist_options_t *options,zSOLVEstruct_t *SOLVEstruct)
{
#ifdef HAVE_NVSHMEM
    nvshmem_free(SOLVEstruct->d_send_idbuf_PermuteC);
    nvshmem_free(SOLVEstruct->d_recv_idbuf_PermuteC);
    nvshmem_free(SOLVEstruct->d_send_idbuf_X2B);
    nvshmem_free(SOLVEstruct->d_recv_idbuf_X2B);
    nvshmem_free(SOLVEstruct->d_send_idbuf_B2X);
    nvshmem_free(SOLVEstruct->d_recv_idbuf_B2X);
#endif
}
