#include <math.h>
#include "superlu_ddefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

#ifndef MAXSUPER
#define MAXSUPER 1024
#endif

// This now only deals with 1 MPI case
__global__ void
pdReDistribute_B_to_X_gpu_proc1(double *B, int_t m_loc, int nrhs, int_t ldb,
                          int_t fst_row, int_t *ilsum, double *x,
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
        x[l - XK_H] = k;      /* Block number prepended in the header. */

        knsupc = SuperSize(k);
        irow = irow - FstBlockC(k); /* Relative row number in X-block */
        for (j = 0; j < nrhs; j++) {
           x[l + irow + j*knsupc] = B[j*m_loc+crow];
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pdReDistribute_B_to_X_gpu_send(double *B, int_t m_loc, int nrhs, int_t ldb,
                          int_t *perm_r, int_t *perm_c,
                          int_t fst_row, int_t *ilsum, double *send_idbuf,
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
        send_idbuf[k] = (double) irow;
        // printf("i = %d, and send_idbuf[%d] is %d.\n", i, k, (int) send_idbuf[k]);
        for (j = 0; j < nrhs; j++) {
           send_idbuf[k+j+1] = B[j*m_loc+crow];
           // printf("i = %d, and send_idbuf[%d] is %f.\n", i, k+j+1, send_idbuf[k+j+1]);
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pdReDistribute_B_to_X_gpu_recv(double *x, int nrhs, int_t recvl,
                          int_t fst_row, int_t *ilsum, double *recv_idbuf,
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
        irow = (int) recv_idbuf[crow*(nrhs+1)]; /* The permuted row index. */
        k = BlockNum(irow);
        knsupc = SuperSize(k);
        lk = LBi(k, grid);  /* Local block number. */
        l = X_BLK(lk);
        x[l - XK_H] = k;      /* Block number prepended in the header. */
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

void pdReDistribute_B_to_X_gpu_wrap(double *B, int_t m_loc, int_t n, int nrhs, int_t ldb,
                                    int_t fst_row, int_t *ilsum, int_t ldalsum, int_t nlb, int_t nsupers, double *x,
		                              dScalePermstruct_t *ScalePermstruct, dSOLVEstruct_t *SOLVEstruct,
		                              Glu_persist_t *Glu_persist, gridinfo_t *grid)
{
  /* ------------------------------------------------------------
     CONSTRUCT A TEMPORARY B FOR GPU COPY.
     ------------------------------------------------------------*/
  double *Btmp;
  int_t ii, jj;
  if ( !(Btmp = (double*)SUPERLU_MALLOC((nrhs*m_loc) * sizeof(double))) )
     ABORT("Calloc fails for Btmp[].");
  for (ii = 0; ii < m_loc; ++ii) {
     RHS_ITERATE(jj) {
       Btmp[ii + jj*m_loc] = B[ii + jj*ldb];
     }
  }

  /* ------------------------------------------------------------
     DECLARATION.
     ------------------------------------------------------------*/
  int_t *perm_r, *perm_c, *xsup, *supno;
  double *d_B = NULL;
  double *d_x = NULL;
  int_t *d_ilsum = NULL;
  gridinfo_t *d_grid = NULL;
  int_t *d_perm_r = NULL;
  int_t *d_perm_c = NULL;
  int_t *d_xsup = NULL;
  int_t *d_supno = NULL;

  double *send_idbuf, *recv_idbuf;
  double *d_send_idbuf = NULL;
  double *d_recv_idbuf = NULL;

  double t;

  /* ------------------------------------------------------------
     MPI DECLARATION.
     ------------------------------------------------------------*/
  int  *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs, *SendCnt_new, *RecvCnt_new;
  int  *sdispls, *rdispls, *sdispls_new, *rdispls_new;
  int  *ptr_to_idbuf;
  int  p, procs;
  pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
  MPI_Request *req_send, *req_recv;
  MPI_Status *status_send, *status_recv;
  int Nreq_recv, Nreq_send, pp, pps, ppr;

  int *d_ptr_to_idbuf;

  /* ------------------------------------------------------------
     CPU INITIALIZATION.
     ------------------------------------------------------------*/
  perm_r = ScalePermstruct->perm_r;
  perm_c = ScalePermstruct->perm_c;
  xsup = Glu_persist->xsup;
  supno = Glu_persist->supno;

  /* ------------------------------------------------------------
     MPI INITIALIZATION.
     ------------------------------------------------------------*/
  procs = grid->nprow * grid->npcol;
  SendCnt      = gstrs_comm->B_to_X_SendCnt;
  SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt +   procs;
  RecvCnt      = gstrs_comm->B_to_X_SendCnt + 2*procs;
  RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3*procs;
  sdispls      = gstrs_comm->B_to_X_SendCnt + 4*procs;
  rdispls      = gstrs_comm->B_to_X_SendCnt + 6*procs;
  if ( !(SendCnt_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for SendCnt_new[].");
  if ( !(RecvCnt_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for RecvCnt_new[].");
  if ( !(sdispls_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for sdispls_new[].");
  if ( !(rdispls_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for rdispls_new[].");
  if ( !(ptr_to_idbuf = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for ptr_to_idbuf[].");

  /* ------------------------------------------------------------
     GPU INITIALIZATION.
     ------------------------------------------------------------*/
  checkGPU(gpuMalloc( (void**)&d_B, sizeof(double)*m_loc*nrhs ));
  checkGPU(gpuMalloc( (void**)&d_x, sizeof(double)*(ldalsum*nrhs+nlb*XK_H) ));
  checkGPU(gpuMalloc( (void**)&d_ilsum, sizeof(int_t)*(nsupers+1) ));
  checkGPU(gpuMalloc( (void**)&d_grid, sizeof(gridinfo_t) ));
  checkGPU(gpuMalloc( (void**)&d_perm_r, sizeof(int_t)*n ));
  checkGPU(gpuMalloc( (void**)&d_perm_c, sizeof(int_t)*n ));
  checkGPU(gpuMalloc( (void**)&d_xsup, sizeof(int_t)*(nsupers+1) ));
  checkGPU(gpuMalloc( (void**)&d_supno, sizeof(int_t)*n ));
  checkGPU(gpuMalloc( (void**)&d_ptr_to_idbuf, sizeof(int)*procs ));

  /* ------------------------------------------------------------
     GPU DATA ALLOCATION.
     ------------------------------------------------------------*/
  checkGPU(gpuMemcpy(d_B, Btmp, sizeof(double)*m_loc*nrhs, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_x, x, sizeof(double)*(ldalsum*nrhs+nlb*XK_H), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_ilsum, ilsum, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_perm_r, perm_r, sizeof(int_t)*n, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_perm_c, perm_c, sizeof(int_t)*n, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_xsup, xsup, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_supno, supno, sizeof(int_t)*n, gpuMemcpyHostToDevice));

  /* ------------------------------------------------------------
     GPU DEVICE RUNNING.
     ------------------------------------------------------------*/
  int_t nthreads = 256;
  int_t row_per_th = 8;
  int_t nblocks = (m_loc+nthreads*row_per_th-1)/(nthreads*row_per_th);
  if (procs == 1) {
     MPI_Barrier( grid->comm );
     t = SuperLU_timer_();

     pdReDistribute_B_to_X_gpu_proc1<<< nthreads, nblocks >>>(d_B, m_loc, nrhs, ldb, fst_row, d_ilsum, d_x, 
                                                              d_perm_r, d_perm_c, d_xsup, d_supno, d_grid, row_per_th);

     t = SuperLU_timer_() - t;
     if ( !grid->iam ) printf(".. B to X redistribute time on host\t%8.4f\n", t);
     fflush(stdout);
  }
  else {
     /* ------------------------------------------------------------
     MPI ALLOCATION FOR GPU.
     ------------------------------------------------------------*/
     int_t k = sdispls[procs-1] + SendCnt[procs-1]; /* Total number of sends */
     int_t l = rdispls[procs-1] + RecvCnt[procs-1]; /* Total number of receives */

     // printf("k is %d and l is %d\n", k, l);

     if ( !(send_idbuf = (double*) SUPERLU_MALLOC(k*((size_t)nrhs+1)*sizeof(double))) )
        ABORT("Malloc fails for send_idbuf[].");
     if ( !(recv_idbuf = (double*) SUPERLU_MALLOC(l*((size_t)nrhs+1)*sizeof(double))) )
        ABORT("Malloc fails for recv_idbuf[].");

     checkGPU(gpuMalloc( (void**)&d_send_idbuf, sizeof(double)*k*((size_t)nrhs+1) ));
     checkGPU(gpuMalloc( (void**)&d_recv_idbuf, sizeof(double)*l*((size_t)nrhs+1) ));

     checkGPU(gpuMemcpy(d_send_idbuf, send_idbuf, sizeof(double)*k*((size_t)nrhs+1), gpuMemcpyHostToDevice));
     checkGPU(gpuMemcpy(d_recv_idbuf, recv_idbuf, sizeof(double)*l*((size_t)nrhs+1), gpuMemcpyHostToDevice));

     if ( !(req_send = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
        ABORT("Malloc fails for req_send[].");
     if ( !(req_recv = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
        ABORT("Malloc fails for req_recv[].");
     if ( !(status_send = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
        ABORT("Malloc fails for status_send[].");
     if ( !(status_recv = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
        ABORT("Malloc fails for status_recv[].");


     // printf("m_loc is %d, nrhs is %d, n is %d, and k is %d.\n", m_loc, nrhs, n, k);
     for (p = 0; p < procs; ++p) {
        ptr_to_idbuf[p] = sdispls[p] * (nrhs+1);
        SendCnt_new[p] = SendCnt_nrhs[p]+SendCnt[p];
        RecvCnt_new[p] = RecvCnt_nrhs[p]+RecvCnt[p];
        sdispls_new[p] = sdispls[p]*(nrhs+1);
        rdispls_new[p] = rdispls[p]*(nrhs+1);
        // printf("ptr_to_idbuf at p = %d is %d.\n", p, ptr_to_idbuf[p]);
        // printf("New send count for p = %d is %d and send disp is %d\n", p, SendCnt_new[p], sdispls_new[p]);
        // printf("New recv count for p = %d is %d and recv disp is %d\n", p, RecvCnt_new[p], rdispls_new[p]);
     }
     checkGPU(gpuMemcpy(d_ptr_to_idbuf, ptr_to_idbuf, sizeof(int)*procs, gpuMemcpyHostToDevice));

     // printf("GPU B_to_X preparation.\n");

     /* ------------------------------------------------------------
     FILL IN THE SEND BUFFERS ON GPU.
     ------------------------------------------------------------*/
     MPI_Barrier( grid->comm );
     t = SuperLU_timer_();

     pdReDistribute_B_to_X_gpu_send<<< nthreads, nblocks >>>(d_B, m_loc, nrhs, ldb, d_perm_r, d_perm_c,
                              fst_row, d_ilsum, d_send_idbuf, d_xsup, d_supno, d_grid, d_ptr_to_idbuf, row_per_th);
     // printf("GPU B_to_X send buffers generated.\n");

     // checkGPU(gpuMemcpy(send_idbuf, d_send_idbuf, k*((size_t)nrhs+1)*sizeof(double), gpuMemcpyDeviceToHost));

     // for (int ct = 0; ct < k*(nrhs+1); ct++) {
     //    printf("send_idbuf[%d] is %f.\n", ct, send_idbuf[ct]);
     // }

     // double thr = 0.0000000001;
     // for (int ct = 0; ct < k; ct++) {
     //     if ((send_idbuf[ct*(nrhs+1)]<thr) || (send_idbuf[ct*(nrhs+1)]+thr<0)) {
     //         printf("send_idbuf has a nonzero int at position %d, with value %d \n", ct*(nrhs+1), (int) send_idbuf[ct*(nrhs+1)]);
     //     }
     //     // for (int ctt = 0; ctt < nrhs; ctt++) {
     //     //     if ((send_idbuf[ct*(nrhs+1)+ctt+1]>thr) || (send_idbuf[ct*(nrhs+1)+ctt+1]+thr<0)) {
     //     //         printf("send_idbuf has a nonzero double at position %d, with value %f \n", ct*(nrhs+1)+ctt+1, send_idbuf[ct*(nrhs+1)+ctt+1]);
     //     //     }
     //     // }
     // }

     /* ------------------------------------------------------------
     USE CUDA_AWARE_MPI FOR COMMUNICATION
     ------------------------------------------------------------*/
     MPI_Barrier( grid->comm );
     Nreq_send = 0;
     Nreq_recv = 0;

     for (pp = 0; pp < procs; pp++) {
        pps = grid->iam + 1 + pp;
        if (pps >= procs) pps -= procs;
        if (pps < 0) pps += procs;

        if (SendCnt_new[pps] > 0) {
           MPI_Isend(&d_send_idbuf[sdispls_new[pps]], SendCnt_new[pps], MPI_DOUBLE, pps, 1, grid->comm, &req_send[Nreq_send]);
           // MPI_Isend(&send_idbuf[sdispls_new[pps]], SendCnt_new[pps], MPI_DOUBLE, pps, 1, grid->comm, &req_send[Nreq_send]);
           Nreq_send++;
        }
      
        ppr = grid->iam - 1 + pp;
        if (ppr >= procs) ppr -= procs;
        if (ppr < 0) ppr += procs;
      
        if (RecvCnt_new[ppr] > 0) {
           MPI_Irecv(&d_recv_idbuf[rdispls_new[ppr]], RecvCnt_new[ppr], MPI_DOUBLE, ppr, 1, grid->comm, &req_recv[Nreq_recv]);
           // MPI_Irecv(&recv_idbuf[rdispls_new[ppr]], RecvCnt_new[ppr], MPI_DOUBLE, ppr, 1, grid->comm, &req_recv[Nreq_recv]);
           Nreq_recv++;
        }
     }

     if (Nreq_send > 0) MPI_Waitall(Nreq_send, req_send, status_send);
     if (Nreq_recv > 0) MPI_Waitall(Nreq_recv, req_recv, status_recv);

     // for (int ct = 0; ct < l*(nrhs+1); ct++) {
     //    printf("recv_idbuf[%d] is %f.\n", ct, recv_idbuf[ct]);
     // }

     // double thr = 0.0000000001;
     // for (int ct = 0; ct < l; ct++) {
     //     if ((recv_idbuf[ct*(nrhs+1)]>thr) || (recv_idbuf[ct*(nrhs+1)]+thr<0)) {
     //         printf("recv_idbuf has a nonzero int at position %d, with value %d \n", ct*(nrhs+1), (int) recv_idbuf[ct*(nrhs+1)]);
     //     }
     //     for (int ctt = 0; ctt < nrhs; ctt++) {
     //         if ((recv_idbuf[ct*(nrhs+1)+ctt+1]>thr) || (recv_idbuf[ct*(nrhs+1)+ctt+1]+thr<0)) {
     //             printf("recv_idbuf has a nonzero double at position %d, with value %f \n", ct*(nrhs+1)+ctt+1, recv_idbuf[ct*(nrhs+1)+ctt+1]);
     //         }
     //     }
     // }

     // printf("GPU B_to_X communication.\n");

     /* ------------------------------------------------------------
     FILL IN X FROM RECEIVE BUFFERS ON GPU.
     ------------------------------------------------------------*/
     // checkGPU(gpuMemcpy(d_recv_idbuf, recv_idbuf, sizeof(double)*l*((size_t)nrhs+1), gpuMemcpyHostToDevice));

     int_t nblocks_new = (l+nthreads*row_per_th-1)/(nthreads*row_per_th);
     pdReDistribute_B_to_X_gpu_recv<<< nthreads, nblocks_new >>>(d_x, nrhs, l, fst_row, d_ilsum, d_recv_idbuf,
                                 d_xsup, d_supno, d_grid, row_per_th);
     
     t = SuperLU_timer_() - t;
     if ( !grid->iam ) printf(".. B to X redistribute time on host\t%8.4f\n", t);
     fflush(stdout);
     // printf("GPU B_to_X receive buffers used.\n");

     /* ------------------------------------------------------------
     CLEANING.
     ------------------------------------------------------------*/
     checkGPU(gpuFree(d_send_idbuf));
     checkGPU(gpuFree(d_recv_idbuf));

     SUPERLU_FREE(req_send);
     SUPERLU_FREE(req_recv);
     SUPERLU_FREE(status_send);
     SUPERLU_FREE(status_recv);

     SUPERLU_FREE(send_idbuf);
     SUPERLU_FREE(recv_idbuf);
  }

  /* ------------------------------------------------------------
     COPY RESULTS BACK TO CPU.
     ------------------------------------------------------------*/
  checkGPU(gpuMemcpy(x, d_x, (ldalsum*nrhs+nlb*XK_H)*sizeof(double), gpuMemcpyDeviceToHost));

  /* ------------------------------------------------------------
     CLEANING.
     ------------------------------------------------------------*/
  checkGPU(gpuFree(d_B));
  checkGPU(gpuFree(d_x));
  checkGPU(gpuFree(d_ilsum));
  checkGPU(gpuFree(d_grid));
  checkGPU(gpuFree(d_perm_r));
  checkGPU(gpuFree(d_perm_c));
  checkGPU(gpuFree(d_xsup));
  checkGPU(gpuFree(d_supno));
  checkGPU(gpuFree(d_ptr_to_idbuf));
  
  SUPERLU_FREE(Btmp);
  SUPERLU_FREE(SendCnt_new);
  SUPERLU_FREE(RecvCnt_new);
  SUPERLU_FREE(sdispls_new);
  SUPERLU_FREE(rdispls_new);
  SUPERLU_FREE(ptr_to_idbuf);
  // printf("GPU B_to_X done.\n");
}

// This also only works for one MPI rank
__global__ void
pdReDistribute_X_to_B_gpu_proc1(int_t n, double *B, int_t m_loc, int_t ldb,
                                int_t fst_row, int_t nrhs, double *x, int_t *ilsum,
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
           B[j*m_loc+crow] = x[l + ii + j*knsupc];
        }
     }
  }
  __syncthreads();

  return;
}

// This deals with multiple MPIs case
__global__ void
pdReDistribute_X_to_B_gpu_send(double *x, int nrhs, int_t ldb, int_t sendk, int_t *row_to_proc,
                          int_t num_diag_procs, int_t *diag_procs, int_t nsupers, int_t m_loc, int_t n,
                          int_t fst_row, int_t *ilsum, double *send_idbuf,
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
           
           send_idbuf[jj] = (double) ii;
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
pdReDistribute_X_to_B_gpu_recv(double *B, int_t m_loc, int nrhs, int_t recvl, int_t ldb,
                          int_t fst_row, int_t *ilsum, double *recv_idbuf,
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
        irow = (int) recv_idbuf[crow*(nrhs+1)]; /* The permuted row index. */
        irow = irow - fst_row;

        for (j = 0; j < nrhs; j++) {
           B[irow + j*m_loc] = recv_idbuf[crow*(nrhs+1)+j+1];
           // printf("On processor %d, i = %d, and B[%d] is %f.\n", grid->iam, i, irow+j*ldb, B[irow+j*ldb]);
        }
     }
  }
  __syncthreads();

  return;
}

void pdReDistribute_X_to_B_gpu_wrap(double *B, int_t m_loc, int_t n, int nrhs, int_t ldb,
                                    int_t fst_row, int_t *ilsum, int_t ldalsum, int_t nlb,
                                    int_t nsupers, double *x,
                                    dScalePermstruct_t *ScalePermstruct, dSOLVEstruct_t *SOLVEstruct,
                                    Glu_persist_t *Glu_persist, gridinfo_t *grid)
{
  /* ------------------------------------------------------------
     CONSTRUCT A TEMPORARY B FOR GPU COPY.
     ------------------------------------------------------------*/
  double *Btmp;
  int_t ii, jj;
  if ( !(Btmp = (double*)SUPERLU_MALLOC((nrhs*m_loc) * sizeof(double))) )
     ABORT("Calloc fails for Btmp[].");
  for (ii = 0; ii < m_loc; ++ii) {
     RHS_ITERATE(jj) {
       Btmp[ii + jj*m_loc] = B[ii + jj*ldb];
     }
  }

  /* ------------------------------------------------------------
     DECLARATION.
     ------------------------------------------------------------*/
  int_t *xsup, *supno;
  int_t  *row_to_proc = SOLVEstruct->row_to_proc; /* row-process mapping */
  int_t num_diag_procs = SOLVEstruct->num_diag_procs;
  int_t *diag_procs = SOLVEstruct->diag_procs;
  double *d_B = NULL;
  double *d_x = NULL;
  int_t *d_ilsum = NULL;
  gridinfo_t *d_grid = NULL;
  int_t *d_xsup = NULL;
  int_t *d_supno = NULL;
  int_t *d_row_to_proc = NULL;
  int_t *d_diag_procs = NULL;

  double *send_idbuf, *recv_idbuf;
  double *d_send_idbuf = NULL;
  double *d_recv_idbuf = NULL;

  double t;

  // printf("nsupers = %d, num_diag_procs = %d\n", nsupers, num_diag_procs);

  /* ------------------------------------------------------------
     MPI DECLARATION.
     ------------------------------------------------------------*/
  int  *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs, *SendCnt_new, *RecvCnt_new;
  int  *sdispls, *rdispls, *sdispls_new, *rdispls_new;
  int  *ptr_to_idbuf;
  int  p, procs;
  pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
  MPI_Request *req_send, *req_recv;
  MPI_Status *status_send, *status_recv;
  int Nreq_recv, Nreq_send, pp, pps, ppr;
  int *d_ptr_to_idbuf = NULL;

  /* ------------------------------------------------------------
     CPU INITIALIZATION.
     ------------------------------------------------------------*/
  xsup = Glu_persist->xsup;
  supno = Glu_persist->supno;

  /* ------------------------------------------------------------
     MPI INITIALIZATION.
     ------------------------------------------------------------*/
  procs = grid->nprow * grid->npcol;
  SendCnt      = gstrs_comm->X_to_B_SendCnt;
  SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt +   procs;
  RecvCnt      = gstrs_comm->X_to_B_SendCnt + 2*procs;
  RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3*procs;
  sdispls      = gstrs_comm->X_to_B_SendCnt + 4*procs;
  rdispls      = gstrs_comm->X_to_B_SendCnt + 6*procs;
  if ( !(SendCnt_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for SendCnt_new[].");
  if ( !(RecvCnt_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for RecvCnt_new[].");
  if ( !(sdispls_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for sdispls_new[].");
  if ( !(rdispls_new = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for rdispls_new[].");
  if ( !(ptr_to_idbuf = (int*) SUPERLU_MALLOC(procs*sizeof(int))) )
     ABORT("Malloc fails for ptr_to_idbuf[].");

  /* ------------------------------------------------------------
     GPU INITIALIZATION.
     ------------------------------------------------------------*/
  checkGPU(gpuMalloc( (void**)&d_B, sizeof(double)*m_loc*nrhs ));
  checkGPU(gpuMalloc( (void**)&d_x, sizeof(double)*(ldalsum*nrhs+nlb*XK_H) ));
  checkGPU(gpuMalloc( (void**)&d_ilsum, sizeof(int_t)*(nsupers+1) ));
  checkGPU(gpuMalloc( (void**)&d_grid, sizeof(gridinfo_t) ));
  checkGPU(gpuMalloc( (void**)&d_xsup, sizeof(int_t)*(nsupers+1) ));
  checkGPU(gpuMalloc( (void**)&d_supno, sizeof(int_t)*n ));
  checkGPU(gpuMalloc( (void**)&d_ptr_to_idbuf, sizeof(int)*procs ));
  checkGPU(gpuMalloc( (void**)&d_row_to_proc, sizeof(int_t)*n ));
  checkGPU(gpuMalloc( (void**)&d_diag_procs, sizeof(int_t)*num_diag_procs ));

  /* ------------------------------------------------------------
     GPU DATA ALLOCATION.
     ------------------------------------------------------------*/
  checkGPU(gpuMemcpy(d_B, Btmp, sizeof(double)*m_loc*nrhs, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_x, x, sizeof(double)*(ldalsum*nrhs+nlb*XK_H), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_ilsum, ilsum, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_xsup, xsup, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_supno, supno, sizeof(int_t)*n, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_row_to_proc, row_to_proc, sizeof(int_t)*n, gpuMemcpyHostToDevice));
  checkGPU(gpuMemcpy(d_diag_procs, diag_procs, sizeof(int_t)*num_diag_procs, gpuMemcpyHostToDevice));

  /* ------------------------------------------------------------
     GPU DEVICE RUNNING.
     ------------------------------------------------------------*/

  int_t nthreads = 256;
  int_t row_per_th = 8;
  int_t nblocks = (m_loc+nthreads*row_per_th-1)/(nthreads*row_per_th);
  if (procs == 1) {
     MPI_Barrier( grid->comm );
     t = SuperLU_timer_();

     pdReDistribute_X_to_B_gpu_proc1<<< nthreads, nblocks >>>(n, d_B, m_loc, ldb, fst_row, nrhs, d_x,
                                                              d_ilsum, d_xsup, d_supno, d_grid, row_per_th);
     
     t = SuperLU_timer_() - t;
     if ( !grid->iam ) printf(".. X to B redistribute time on host\t%8.4f\n", t);
     fflush(stdout);
  }
  else {
     /* ------------------------------------------------------------
     MPI ALLOCATION FOR GPU.
     ------------------------------------------------------------*/
     int_t k = sdispls[procs-1] + SendCnt[procs-1]; /* Total number of sends */
     int_t l = rdispls[procs-1] + RecvCnt[procs-1]; /* Total number of receives */

     if ( !(send_idbuf = (double*) SUPERLU_MALLOC(k*((size_t)nrhs+1)*sizeof(double))) )
        ABORT("Malloc fails for send_idbuf[].");
     if ( !(recv_idbuf = (double*) SUPERLU_MALLOC(l*((size_t)nrhs+1)*sizeof(double))) )
        ABORT("Malloc fails for recv_idbuf[].");

     checkGPU(gpuMalloc( (void**)&d_send_idbuf, sizeof(double)*k*((size_t)nrhs+1) ));
     checkGPU(gpuMalloc( (void**)&d_recv_idbuf, sizeof(double)*l*((size_t)nrhs+1) ));

     checkGPU(gpuMemcpy(d_send_idbuf, send_idbuf, sizeof(double)*k*((size_t)nrhs+1), gpuMemcpyHostToDevice));
     checkGPU(gpuMemcpy(d_recv_idbuf, recv_idbuf, sizeof(double)*l*((size_t)nrhs+1), gpuMemcpyHostToDevice));

     if ( !(req_send = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
        ABORT("Malloc fails for req_send[].");
     if ( !(req_recv = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
        ABORT("Malloc fails for req_recv[].");
     if ( !(status_send = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
        ABORT("Malloc fails for status_send[].");
     if ( !(status_recv = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
        ABORT("Malloc fails for status_recv[].");


     // printf("m_loc is %d, nrhs is %d, n is %d, and k is %d.\n", m_loc, nrhs, n, k);
     for (p = 0; p < procs; ++p) {
        ptr_to_idbuf[p] = sdispls[p] * (nrhs+1);
        SendCnt_new[p] = SendCnt_nrhs[p]+SendCnt[p];
        RecvCnt_new[p] = RecvCnt_nrhs[p]+RecvCnt[p];
        sdispls_new[p] = sdispls[p]*(nrhs+1);
        rdispls_new[p] = rdispls[p]*(nrhs+1);
        // printf("ptr_to_idbuf at p = %d is %d.\n", p, ptr_to_idbuf[p]);
        // printf("New send count for p = %d is %d and send disp is %d\n", p, SendCnt_new[p], sdispls_new[p]);
        // printf("New recv count for p = %d is %d and recv disp is %d\n", p, RecvCnt_new[p], rdispls_new[p]);
     }
     checkGPU(gpuMemcpy(d_ptr_to_idbuf, ptr_to_idbuf, sizeof(int)*procs, gpuMemcpyHostToDevice));

     // printf("GPU X_to_B preparation.\n");

     /* ------------------------------------------------------------
     FILL IN THE SEND BUFFERS ON GPU.
     ------------------------------------------------------------*/
     MPI_Barrier( grid->comm );
     t = SuperLU_timer_();

     int_t nblocks_new = (n+nthreads*row_per_th-1)/(nthreads*row_per_th);
     pdReDistribute_X_to_B_gpu_send<<< nthreads, nblocks_new >>>(d_x, nrhs, ldb, k, d_row_to_proc, num_diag_procs, d_diag_procs, 
                          nsupers, m_loc, n, fst_row, d_ilsum, d_send_idbuf, d_xsup, d_supno, d_grid, d_ptr_to_idbuf, row_per_th);
     // printf("GPU X_to_B send buffers generated.\n");

     // checkGPU(gpuMemcpy(send_idbuf, d_send_idbuf, k*((size_t)nrhs+1)*sizeof(double), gpuMemcpyDeviceToHost));
     // for (int ct = 0; ct < k*(nrhs+1); ct++) {
     //    printf("On processor %d, send_idbuf[%d] is %f.\n", grid->iam, ct, send_idbuf[ct]);
     // }

     /* ------------------------------------------------------------
     USE CUDA_AWARE_MPI FOR COMMUNICATION
     ------------------------------------------------------------*/
     MPI_Barrier( grid->comm );
     Nreq_send = 0;
     Nreq_recv = 0;

     for (pp = 0; pp < procs; pp++) {
        pps = grid->iam + 1 + pp;
        if (pps >= procs) pps -= procs;
        if (pps < 0) pps += procs;

        if (SendCnt_new[pps] > 0) {
           MPI_Isend(&d_send_idbuf[sdispls_new[pps]], SendCnt_new[pps], MPI_DOUBLE, pps, 1, grid->comm, &req_send[Nreq_send]);
           // MPI_Isend(&send_idbuf[sdispls_new[pps]], SendCnt_new[pps], MPI_DOUBLE, pps, 1, grid->comm, &req_send[Nreq_send]);
           Nreq_send++;
        }
      
        ppr = grid->iam - 1 + pp;
        if (ppr >= procs) ppr -= procs;
        if (ppr < 0) ppr += procs;
      
        if (RecvCnt_new[ppr] > 0) {
           MPI_Irecv(&d_recv_idbuf[rdispls_new[ppr]], RecvCnt_new[ppr], MPI_DOUBLE, ppr, 1, grid->comm, &req_recv[Nreq_recv]);
           // MPI_Irecv(&recv_idbuf[rdispls_new[ppr]], RecvCnt_new[ppr], MPI_DOUBLE, ppr, 1, grid->comm, &req_recv[Nreq_recv]);
           Nreq_recv++;
        }
     }

     if (Nreq_send > 0) MPI_Waitall(Nreq_send, req_send, status_send);
     if (Nreq_recv > 0) MPI_Waitall(Nreq_recv, req_recv, status_recv);

     // printf("GPU X_to_B communication.\n");

     /* ------------------------------------------------------------
     FILL IN B FROM RECEIVE BUFFERS ON GPU.
     ------------------------------------------------------------*/
     // printf("Processor is %d, l is %d, fst_row is %d.\n", grid->iam, l, fst_row);
     // checkGPU(gpuMemcpy(d_recv_idbuf, recv_idbuf, sizeof(double)*l*((size_t)nrhs+1), gpuMemcpyHostToDevice));
     // for (int ct = 0; ct < l; ct++) {
     //    printf("On processor %d, recv_idbuf[%d] is %d.\n", grid->iam, ct*(nrhs+1), (int) recv_idbuf[ct*(nrhs+1)]);
     // }

     nblocks_new = (l+nthreads*row_per_th-1)/(nthreads*row_per_th);
     pdReDistribute_X_to_B_gpu_recv<<< nthreads, nblocks_new >>>(d_B, m_loc, nrhs, l, ldb, fst_row, d_ilsum, d_recv_idbuf,
                          d_xsup, d_supno, d_grid, row_per_th);
     
     t = SuperLU_timer_() - t;
     if ( !grid->iam ) printf(".. X to B redistribute time on host\t%8.4f\n", t);
     fflush(stdout);
     // printf("GPU X_to_B receive buffers used.\n");

     /* ------------------------------------------------------------
     CLEANING.
     ------------------------------------------------------------*/
     checkGPU(gpuFree(d_send_idbuf));
     checkGPU(gpuFree(d_recv_idbuf));

     SUPERLU_FREE(req_send);
     SUPERLU_FREE(req_recv);
     SUPERLU_FREE(status_send);
     SUPERLU_FREE(status_recv);

     SUPERLU_FREE(send_idbuf);
     SUPERLU_FREE(recv_idbuf);
  }

  /* ------------------------------------------------------------
     COPY RESULTS BACK TO CPU.
     ------------------------------------------------------------*/
  checkGPU(gpuMemcpy(Btmp, d_B, m_loc*nrhs*sizeof(double), gpuMemcpyDeviceToHost));
  // for (int ct = 0; ct < m_loc*nrhs; ct++) {
  //    printf("Btmp[%d] has element %f.\n", ct, Btmp[ct]);
  // }
  for (ii = 0; ii < m_loc; ++ii) {
     RHS_ITERATE(jj) {
       B[ii + jj*ldb] = Btmp[ii + jj*m_loc];
     }
  }

  /* ------------------------------------------------------------
     CLEANING.
     ------------------------------------------------------------*/
  checkGPU(gpuFree(d_B));
  checkGPU(gpuFree(d_x));
  checkGPU(gpuFree(d_ilsum));
  checkGPU(gpuFree(d_grid));
  checkGPU(gpuFree(d_xsup));
  checkGPU(gpuFree(d_supno));
  checkGPU(gpuFree(d_row_to_proc));
  checkGPU(gpuFree(d_diag_procs));
  checkGPU(gpuFree(d_ptr_to_idbuf));
  
  SUPERLU_FREE(Btmp);
  SUPERLU_FREE(SendCnt_new);
  SUPERLU_FREE(RecvCnt_new);
  SUPERLU_FREE(sdispls_new);
  SUPERLU_FREE(rdispls_new);
  SUPERLU_FREE(ptr_to_idbuf);
}