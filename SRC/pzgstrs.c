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
#include "superlu_zdefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

/*
 * Sketch of the algorithm for L-solve:
 * =======================
 *
 * Self-scheduling loop:
 *
 *   while ( not finished ) { .. use message counter to control
 *
 *      reveive a message;
 *
 * 	if ( message is Xk ) {
 * 	    perform local block modifications into lsum[];
 *                 lsum[i] -= L_i,k * X[k]
 *          if all local updates done, Isend lsum[] to diagonal process;
 *
 *      } else if ( message is LSUM ) { .. this must be a diagonal process
 *          accumulate LSUM;
 *          if ( all LSUM are received ) {
 *              perform triangular solve for Xi;
 *              Isend Xi down to the current process column;
 *              perform local block modifications into lsum[];
 *          }
 *      }
 *   }
 *
 *
 * Auxiliary data structures: lsum[] / ilsum (pointer to lsum array)
 * =======================
 *
 * lsum[] array (local)
 *   + lsum has "nrhs" columns, row-wise is partitioned by supernodes
 *   + stored by row blocks, column wise storage within a row block
 *   + prepend a header recording the global block number.
 *
 *         lsum[]                        ilsum[nsupers + 1]
 *
 *         -----
 *         | | |  <- header of size 2     ---
 *         --------- <--------------------| |
 *         | | | | |			  ---
 * 	   | | | | |	      |-----------| |
 *         | | | | | 	      |           ---
 *	   ---------          |   |-------| |
 *         | | |  <- header   |   |       ---
 *         --------- <--------|   |  |----| |
 *         | | | | |		  |  |    ---
 * 	   | | | | |              |  |
 *         | | | | |              |  |
 *	   ---------              |  |
 *         | | |  <- header       |  |
 *         --------- <------------|  |
 *         | | | | |                 |
 * 	   | | | | |                 |
 *         | | | | |                 |
 *	   --------- <---------------|
 */

/*#define ISEND_IRECV*/

/*
 * Function prototypes
 */
#ifdef _CRAY
fortran void CTRSM(_fcd, _fcd, _fcd, _fcd, int*, int*, doublecomplex*,
		   doublecomplex*, int*, doublecomplex*, int*);
_fcd ftcs1;
_fcd ftcs2;
_fcd ftcs3;
#endif

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Re-distribute B on the diagonal processes of the 2D process mesh.
 *
 * Note
 * ====
 *   This routine can only be called after the routine pzgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 *
 * Arguments
 * =========
 *
 * B      (input) doublecomplex*
 *        The distributed right-hand side matrix of the possibly
 *        equilibrated system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * ldb    (input) int (local)
 *        Leading dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ilsum  (input) int* (global)
 *        Starting position of each supernode in a full array.
 *
 * x      (output) doublecomplex*
 *        The solution vector. It is valid only on the diagonal processes.
 *
 * ScalePermstruct (input) zScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * SOLVEstruct (input) zSOLVEstruct_t*
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * Return value
 * ============
 * </pre>
 */

int_t
pzReDistribute_B_to_X(doublecomplex *B, int_t m_loc, int nrhs, int_t ldb,
                      int_t fst_row, int_t *ilsum, doublecomplex *x,
		      zScalePermstruct_t *ScalePermstruct,
		      Glu_persist_t *Glu_persist,
		      gridinfo_t *grid, zSOLVEstruct_t *SOLVEstruct)
{
    int  *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int  *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int  *ptr_to_ibuf, *ptr_to_dbuf;
    int_t  *perm_r, *perm_c; /* row and column permutation vectors */
    int_t  *send_ibuf, *recv_ibuf;
    doublecomplex *send_dbuf, *recv_dbuf;
    int_t  *xsup, *supno;
    int_t  i, ii, irow, gbi, j, jj, k, knsupc, l, lk, nbrow;
    int    p, procs;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
	MPI_Request req_i, req_d, *req_send, *req_recv;
	MPI_Status status, *status_send, *status_recv;
	int Nreq_recv, Nreq_send, pp, pps, ppr;
	double t;
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Enter pzReDistribute_B_to_X()");
#endif

    /* ------------------------------------------------------------
       INITIALIZATION.
       ------------------------------------------------------------*/
    perm_r = ScalePermstruct->perm_r;
    perm_c = ScalePermstruct->perm_c;
    procs = grid->nprow * grid->npcol;
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    SendCnt      = gstrs_comm->B_to_X_SendCnt;
    SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt +   procs;
    RecvCnt      = gstrs_comm->B_to_X_SendCnt + 2*procs;
    RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3*procs;
    sdispls      = gstrs_comm->B_to_X_SendCnt + 4*procs;
    sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5*procs;
    rdispls      = gstrs_comm->B_to_X_SendCnt + 6*procs;
    rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7*procs;
    ptr_to_ibuf  = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf  = gstrs_comm->ptr_to_dbuf;

    /* ------------------------------------------------------------
       NOW COMMUNICATE THE ACTUAL DATA.
       ------------------------------------------------------------*/

	if(procs==1){ // faster memory copy when procs=1

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	{
		// t = SuperLU_timer_();
#ifdef _OPENMP
#pragma	omp	taskloop private (i,l,irow,k,j,knsupc) untied
#endif
		for (i = 0; i < m_loc; ++i) {
			irow = perm_c[perm_r[i+fst_row]]; /* Row number in Pc*Pr*B */

			k = BlockNum( irow );
			knsupc = SuperSize( k );
			l = X_BLK( k );

			x[l - XK_H].r = k; /* Block number prepended in the header. */
			x[l - XK_H].i = 0;

			irow = irow - FstBlockC(k); /* Relative row number in X-block */
			RHS_ITERATE(j) {
			x[l + irow + j*knsupc] = B[i + j*ldb];
			}
		}
	}
	}
	}else{
		k = sdispls[procs-1] + SendCnt[procs-1]; /* Total number of sends */
		l = rdispls[procs-1] + RecvCnt[procs-1]; /* Total number of receives */
		if ( !(send_ibuf = intMalloc_dist(k + l)) )
			ABORT("Malloc fails for send_ibuf[].");
		recv_ibuf = send_ibuf + k;
		if ( !(send_dbuf = doublecomplexMalloc_dist((k + l)* (size_t)nrhs)) )
			ABORT("Malloc fails for send_dbuf[].");
		recv_dbuf = send_dbuf + k * nrhs;
		if ( !(req_send = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
			ABORT("Malloc fails for req_send[].");
		if ( !(req_recv = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
			ABORT("Malloc fails for req_recv[].");
		if ( !(status_send = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
			ABORT("Malloc fails for status_send[].");
		if ( !(status_recv = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
			ABORT("Malloc fails for status_recv[].");

		for (p = 0; p < procs; ++p) {
			ptr_to_ibuf[p] = sdispls[p];
			ptr_to_dbuf[p] = sdispls[p] * nrhs;
		}

		/* Copy the row indices and values to the send buffer. */
		// t = SuperLU_timer_();
		for (i = 0, l = fst_row; i < m_loc; ++i, ++l) {
			irow = perm_c[perm_r[l]]; /* Row number in Pc*Pr*B */
		gbi = BlockNum( irow );
		p = PNUM( PROW(gbi,grid), PCOL(gbi,grid), grid ); /* Diagonal process */
		k = ptr_to_ibuf[p];
		send_ibuf[k] = irow;
		++ptr_to_ibuf[p];

		k = ptr_to_dbuf[p];
		RHS_ITERATE(j) { /* RHS is stored in row major in the buffer. */
			send_dbuf[k++] = B[i + j*ldb];
		}
		ptr_to_dbuf[p] += nrhs;
		}

		// t = SuperLU_timer_() - t;
		// printf(".. copy to send buffer time\t%8.4f\n", t);

#if 0
	#if 1
		/* Communicate the (permuted) row indices. */
		MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
			  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);
 		/* Communicate the numerical values. */
		MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
			  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
			  grid->comm);
	#else
 		/* Communicate the (permuted) row indices. */
		MPI_Ialltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
				recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm, &req_i);
 		/* Communicate the numerical values. */
		MPI_Ialltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
				recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
				grid->comm, &req_d);
		MPI_Wait(&req_i,&status);
		MPI_Wait(&req_d,&status);
 	#endif
#endif
	MPI_Barrier( grid->comm );


	Nreq_send=0;
	Nreq_recv=0;
	for (pp=0;pp<procs;pp++){
		pps = grid->iam+1+pp;
		if(pps>=procs)pps-=procs;
		if(pps<0)pps+=procs;
		ppr = grid->iam-1+pp;
		if(ppr>=procs)ppr-=procs;
		if(ppr<0)ppr+=procs;

		if(SendCnt[pps]>0){
			MPI_Isend(&send_ibuf[sdispls[pps]], SendCnt[pps], mpi_int_t, pps, 0, grid->comm,
			&req_send[Nreq_send] );
			Nreq_send++;
		}
		if(RecvCnt[ppr]>0){
			MPI_Irecv(&recv_ibuf[rdispls[ppr]], RecvCnt[ppr], mpi_int_t, ppr, 0, grid->comm,
			&req_recv[Nreq_recv] );
			Nreq_recv++;
		}
	}


	if(Nreq_send>0)MPI_Waitall(Nreq_send,req_send,status_send);
	if(Nreq_recv>0)MPI_Waitall(Nreq_recv,req_recv,status_recv);


	Nreq_send=0;
	Nreq_recv=0;
	for (pp=0;pp<procs;pp++){
		pps = grid->iam+1+pp;
		if(pps>=procs)pps-=procs;
		if(pps<0)pps+=procs;
		ppr = grid->iam-1+pp;
		if(ppr>=procs)ppr-=procs;
		if(ppr<0)ppr+=procs;
		if(SendCnt_nrhs[pps]>0){
			MPI_Isend(&send_dbuf[sdispls_nrhs[pps]], SendCnt_nrhs[pps], SuperLU_MPI_DOUBLE_COMPLEX, pps, 1, grid->comm,
			&req_send[Nreq_send] );
			Nreq_send++;
		}
		if(RecvCnt_nrhs[ppr]>0){
			MPI_Irecv(&recv_dbuf[rdispls_nrhs[ppr]], RecvCnt_nrhs[ppr], SuperLU_MPI_DOUBLE_COMPLEX, ppr, 1, grid->comm,
			&req_recv[Nreq_recv] );
			Nreq_recv++;
		}
	}

	if(Nreq_send>0)MPI_Waitall(Nreq_send,req_send,status_send);
	if(Nreq_recv>0)MPI_Waitall(Nreq_recv,req_recv,status_recv);


		/* ------------------------------------------------------------
		   Copy buffer into X on the diagonal processes.
		   ------------------------------------------------------------*/

		// t = SuperLU_timer_();
		ii = 0;
		for (p = 0; p < procs; ++p) {
			jj = rdispls_nrhs[p];
			for (i = 0; i < RecvCnt[p]; ++i) {
			/* Only the diagonal processes do this; the off-diagonal processes
			   have 0 RecvCnt. */
			irow = recv_ibuf[ii]; /* The permuted row index. */
			k = BlockNum( irow );
			knsupc = SuperSize( k );
			lk = LBi( k, grid );  /* Local block number. */
			l = X_BLK( lk );
			x[l - XK_H].r = k; /* Block number prepended in the header. */
			x[l - XK_H].i = 0;

			irow = irow - FstBlockC(k); /* Relative row number in X-block */
			RHS_ITERATE(j) {
				x[l + irow + j*knsupc] = recv_dbuf[jj++];
			}
			++ii;
		}
		}

		// t = SuperLU_timer_() - t;
		// printf(".. copy to x time\t%8.4f\n", t);

		SUPERLU_FREE(send_ibuf);
		SUPERLU_FREE(send_dbuf);
		SUPERLU_FREE(req_send);
		SUPERLU_FREE(req_recv);
		SUPERLU_FREE(status_send);
		SUPERLU_FREE(status_recv);
	}


#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Exit pzReDistribute_B_to_X()");
#endif
    return 0;
} /* pzReDistribute_B_to_X */

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Re-distribute X on the diagonal processes to B distributed on all
 *   the processes.
 *
 * Note
 * ====
 *   This routine can only be called after the routine pzgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 * </pre>
 */

int_t
pzReDistribute_X_to_B(int_t n, doublecomplex *B, int_t m_loc, int_t ldb, int_t fst_row,
		      int_t nrhs, doublecomplex *x, int_t *ilsum,
		      zScalePermstruct_t *ScalePermstruct,
		      Glu_persist_t *Glu_persist, gridinfo_t *grid,
		      zSOLVEstruct_t *SOLVEstruct)
{
    int_t  i, ii, irow, j, jj, k, knsupc, nsupers, l, lk;
    int_t  *xsup, *supno;
    int  *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int  *sdispls, *rdispls, *sdispls_nrhs, *rdispls_nrhs;
    int  *ptr_to_ibuf, *ptr_to_dbuf;
    int_t  *send_ibuf, *recv_ibuf;
    doublecomplex *send_dbuf, *recv_dbuf;
    int_t  *row_to_proc = SOLVEstruct->row_to_proc; /* row-process mapping */
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    int  iam, p, q, pkk, procs;
    int_t  num_diag_procs, *diag_procs;
	MPI_Request req_i, req_d, *req_send, *req_recv;
	MPI_Status status, *status_send, *status_recv;
	int Nreq_recv, Nreq_send, pp,pps,ppr;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Enter pzReDistribute_X_to_B()");
#endif

    /* ------------------------------------------------------------
       INITIALIZATION.
       ------------------------------------------------------------*/
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n-1] + 1;
    iam = grid->iam;
    procs = grid->nprow * grid->npcol;

    SendCnt      = gstrs_comm->X_to_B_SendCnt;
    SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt +   procs;
    RecvCnt      = gstrs_comm->X_to_B_SendCnt + 2*procs;
    RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3*procs;
    sdispls      = gstrs_comm->X_to_B_SendCnt + 4*procs;
    sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5*procs;
    rdispls      = gstrs_comm->X_to_B_SendCnt + 6*procs;
    rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7*procs;
    ptr_to_ibuf  = gstrs_comm->ptr_to_ibuf;
    ptr_to_dbuf  = gstrs_comm->ptr_to_dbuf;


	if(procs==1){ //faster memory copy when procs=1

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	{
		// t = SuperLU_timer_();
#ifdef _OPENMP
#pragma	omp	taskloop private (k,knsupc,lk,irow,l,i,j) untied
#endif
		for (k = 0; k < nsupers; k++) {
		knsupc = SuperSize( k );
		lk = LBi( k, grid ); /* Local block number */
		irow = FstBlockC( k );
		l = X_BLK( lk );
		for (i = 0; i < knsupc; ++i) {
			RHS_ITERATE(j) { /* RHS is stored in row major in the buffer. */
				B[irow-fst_row +i + j*ldb] = x[l + i + j*knsupc];
			}
			}
		}
	}
	}
	}else{
		k = sdispls[procs-1] + SendCnt[procs-1]; /* Total number of sends */
		l = rdispls[procs-1] + RecvCnt[procs-1]; /* Total number of receives */
		if ( !(send_ibuf = intMalloc_dist(k + l)) )
			ABORT("Malloc fails for send_ibuf[].");
		recv_ibuf = send_ibuf + k;
		if ( !(send_dbuf = doublecomplexMalloc_dist((k + l)*nrhs)) )
			ABORT("Malloc fails for send_dbuf[].");
		if ( !(req_send = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
			ABORT("Malloc fails for req_send[].");
		if ( !(req_recv = (MPI_Request*) SUPERLU_MALLOC(procs*sizeof(MPI_Request))) )
			ABORT("Malloc fails for req_recv[].");
		if ( !(status_send = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
			ABORT("Malloc fails for status_send[].");
		if ( !(status_recv = (MPI_Status*) SUPERLU_MALLOC(procs*sizeof(MPI_Status))) )
			ABORT("Malloc fails for status_recv[].");
		recv_dbuf = send_dbuf + k * nrhs;
		for (p = 0; p < procs; ++p) {
			ptr_to_ibuf[p] = sdispls[p];
			ptr_to_dbuf[p] = sdispls_nrhs[p];
		}
		num_diag_procs = SOLVEstruct->num_diag_procs;
		diag_procs = SOLVEstruct->diag_procs;
 		for (p = 0; p < num_diag_procs; ++p) {  /* For all diagonal processes. */
		pkk = diag_procs[p];
		if ( iam == pkk ) {
			for (k = p; k < nsupers; k += num_diag_procs) {
			knsupc = SuperSize( k );
			lk = LBi( k, grid ); /* Local block number */
			irow = FstBlockC( k );
			l = X_BLK( lk );
			for (i = 0; i < knsupc; ++i) {
	#if 0
				ii = inv_perm_c[irow]; /* Apply X <== Pc'*Y */
	#else
				ii = irow;
	#endif
				q = row_to_proc[ii];
				jj = ptr_to_ibuf[q];
				send_ibuf[jj] = ii;
				jj = ptr_to_dbuf[q];
				RHS_ITERATE(j) { /* RHS stored in row major in buffer. */
					send_dbuf[jj++] = x[l + i + j*knsupc];
				}
				++ptr_to_ibuf[q];
				ptr_to_dbuf[q] += nrhs;
				++irow;
			}
			}
		}
		}

		/* ------------------------------------------------------------
			COMMUNICATE THE (PERMUTED) ROW INDICES AND NUMERICAL VALUES.
		   ------------------------------------------------------------*/
#if 0
	#if 1
		MPI_Alltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
			  recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);
		MPI_Alltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs,SuperLU_MPI_DOUBLE_COMPLEX,
			  recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
			  grid->comm);
	#else
		MPI_Ialltoallv(send_ibuf, SendCnt, sdispls, mpi_int_t,
				recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm,&req_i);
		MPI_Ialltoallv(send_dbuf, SendCnt_nrhs, sdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
				recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
				grid->comm,&req_d);
 		MPI_Wait(&req_i,&status);
		MPI_Wait(&req_d,&status);
	#endif
#endif

	MPI_Barrier( grid->comm );
	Nreq_send=0;
	Nreq_recv=0;
	for (pp=0;pp<procs;pp++){
		pps = grid->iam+1+pp;
		if(pps>=procs)pps-=procs;
		if(pps<0)pps+=procs;
		ppr = grid->iam-1+pp;
		if(ppr>=procs)ppr-=procs;
		if(ppr<0)ppr+=procs;
		if(SendCnt[pps]>0){
			MPI_Isend(&send_ibuf[sdispls[pps]], SendCnt[pps], mpi_int_t, pps, 0, grid->comm,
			&req_send[Nreq_send] );
			Nreq_send++;
		}
		if(RecvCnt[ppr]>0){
			MPI_Irecv(&recv_ibuf[rdispls[ppr]], RecvCnt[ppr], mpi_int_t, ppr, 0, grid->comm,
			&req_recv[Nreq_recv] );
			Nreq_recv++;
		}
	}


	if(Nreq_send>0)MPI_Waitall(Nreq_send,req_send,status_send);
	if(Nreq_recv>0)MPI_Waitall(Nreq_recv,req_recv,status_recv);
	// MPI_Barrier( grid->comm );

	Nreq_send=0;
	Nreq_recv=0;
	for (pp=0;pp<procs;pp++){
		pps = grid->iam+1+pp;
		if(pps>=procs)pps-=procs;
		if(pps<0)pps+=procs;
		ppr = grid->iam-1+pp;
		if(ppr>=procs)ppr-=procs;
		if(ppr<0)ppr+=procs;
		if(SendCnt_nrhs[pps]>0){
			MPI_Isend(&send_dbuf[sdispls_nrhs[pps]], SendCnt_nrhs[pps], SuperLU_MPI_DOUBLE_COMPLEX, pps, 1, grid->comm,
			&req_send[Nreq_send] );
			Nreq_send++;
		}
		if(RecvCnt_nrhs[ppr]>0){
			MPI_Irecv(&recv_dbuf[rdispls_nrhs[ppr]], RecvCnt_nrhs[ppr], SuperLU_MPI_DOUBLE_COMPLEX, ppr, 1, grid->comm,
			&req_recv[Nreq_recv] );
			Nreq_recv++;
		}
	}


	if(Nreq_send>0)MPI_Waitall(Nreq_send,req_send,status_send);
	if(Nreq_recv>0)MPI_Waitall(Nreq_recv,req_recv,status_recv);
	// MPI_Barrier( grid->comm );


		/* ------------------------------------------------------------
		   COPY THE BUFFER INTO B.
		   ------------------------------------------------------------*/
		for (i = 0, k = 0; i < m_loc; ++i) {
		irow = recv_ibuf[i];
		irow -= fst_row; /* Relative row number */
		RHS_ITERATE(j) { /* RHS is stored in row major in the buffer. */
			B[irow + j*ldb] = recv_dbuf[k++];
		}
		}

    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
	SUPERLU_FREE(req_send);
	SUPERLU_FREE(req_recv);
	SUPERLU_FREE(status_send);
	SUPERLU_FREE(status_recv);
}
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Exit pzReDistribute_X_to_B()");
#endif
    return 0;

} /* pzReDistribute_X_to_B */




/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Compute the inverse of the diagonal blocks of the L and U
 *   triangular matrices.
 * </pre>
 */
void
pzCompute_Diag_Inv(int_t n, zLUstruct_t *LUstruct,gridinfo_t *grid,
                   SuperLUStat_t *stat, int *info)
{
#ifdef SLU_HAVE_LAPACK
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;

    doublecomplex *lusup;
    doublecomplex *recvbuf, *tempv;
    doublecomplex *Linv;/* Inverse of diagonal block */
    doublecomplex *Uinv;/* Inverse of diagonal block */

    int_t  kcol, krow, mycol, myrow;
    int_t  i, ii, il, j, jj, k, lb, ljb, lk, lptr, luptr;
    int_t  nb, nlb,nlb_nodiag, nub, nsupers;
    int_t  *xsup, *supno, *lsub, *usub;
    int_t  *ilsum;    /* Starting position of each supernode in lsum (LOCAL)*/
    int    Pc, Pr, iam;
    int    knsupc, nsupr;
    int    ldalsum;   /* Number of lsum entries locally owned. */
    int    maxrecvsz, p, pi;
    int_t  **Lrowind_bc_ptr;
    doublecomplex **Lnzval_bc_ptr;
    doublecomplex **Linv_bc_ptr;
    doublecomplex **Uinv_bc_ptr;
    int INFO;
    double t;

    doublecomplex one = {1.0, 0.0};
    doublecomplex zero = {0.0, 0.0};

#if ( PROFlevel>=1 )
    t = SuperLU_timer_();
#endif

#if ( PRNTlevel>=2 )
    if ( grid->iam==0 ) {
	printf("computing inverse of diagonal blocks...\n");
	fflush(stdout);
    }
#endif

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n-1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Linv_bc_ptr = Llu->Linv_bc_ptr;
    Uinv_bc_ptr = Llu->Uinv_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    nlb = CEILING( nsupers, Pr ); /* Number of local block rows. */

    Llu->inv = 1;

    /*---------------------------------------------------
     * Compute inverse of L(lk,lk).
     *---------------------------------------------------*/

     for (k = 0; k < nsupers; ++k) {
         krow = PROW( k, grid );
	 if ( myrow == krow ) {
	     lk = LBi( k, grid );    /* local block number */
	     kcol = PCOL( k, grid );
	     if ( mycol == kcol ) { /* diagonal process */

	     	  lk = LBj( k, grid ); /* Local block number, column-wise. */
		  lsub = Lrowind_bc_ptr[lk];
		  lusup = Lnzval_bc_ptr[lk];
		  Linv = Linv_bc_ptr[lk];
		  Uinv = Uinv_bc_ptr[lk];
		  nsupr = lsub[1];
		  knsupc = SuperSize( k );

		  for (j=0 ; j<knsupc; j++){
		      for (i=0 ; i<knsupc; i++){
		  	  Linv[j*knsupc+i] = zero;
			  Uinv[j*knsupc+i] = zero;
		      }
	          }

	   	  for (j=0 ; j<knsupc; j++){
		      Linv[j*knsupc+j] = one;
		      for (i=j+1 ; i<knsupc; i++){
		  	  z_copy(&Linv[j*knsupc+i],&lusup[j*nsupr+i]);
		      }

		      for (i=0 ; i<j+1; i++){
		          z_copy(&Uinv[j*knsupc+i],&lusup[j*nsupr+i]);
	              }
 		  }

		  /* Triangular inversion */
   		  ztrtri_("L","U",&knsupc,Linv,&knsupc,&INFO);

		  ztrtri_("U","N",&knsupc,Uinv,&knsupc,&INFO);

	    } /* end if (mycol === kcol) */
	} /* end if (myrow === krow) */
    } /* end fo k = ... nsupers */

#if ( PROFlevel>=1 )
    if( grid->iam==0 ) {
	t = SuperLU_timer_() - t;
	printf(".. L-diag_inv time\t%10.5f\n", t);
	fflush(stdout);
    }
#endif

    return;
#endif /* SLU_HAVE_LAPACK */
}


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PZGSTRS solves a system of distributed linear equations
 * A*X = B with a general N-by-N matrix A using the LU factorization
 * computed by PZGSTRF.
 * If the equilibration, and row and column permutations were performed,
 * the LU factorization was performed for A1 where
 *     A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 * and the linear system solved is
 *     A1 * Y = Pc*Pr*B1, where B was overwritten by B1 = diag(R)*B, and
 * the permutation to B1 by Pc*Pr is applied internally in this routine.
 *
 * Arguments
 * =========
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * LUstruct (input) zLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from PZGSTRF for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_zdefs.h for the definition of 'zLUstruct_t'.
 *        A may be scaled and permuted into A1, so that
 *        A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and my process rank. It is an input argument to all the
 *        parallel routines.
 *        Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *        See superlu_defs.h for the definition of 'gridinfo_t'.
 *
 * B      (input/output) doublecomplex*
 *        On entry, the distributed right-hand side matrix of the possibly
 *        equilibrated system. That is, B may be overwritten by diag(R)*B.
 *        On exit, the distributed solution matrix Y of the possibly
 *        equilibrated system if info = 0, where Y = Pc*diag(C)^(-1)*X,
 *        and X is the solution of the original system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ldb    (input) int (local)
 *        The leading dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * SOLVEstruct (input) zSOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the triangular solves.
 *        See util.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 * 	   = 0: successful exit
 *	   < 0: if info = -i, the i-th argument had an illegal value
 * </pre>
 */

void
pzgstrs(superlu_dist_options_t *options, int_t n,
        zLUstruct_t *LUstruct,
	zScalePermstruct_t *ScalePermstruct,
	gridinfo_t *grid, doublecomplex *B,
	int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
	zSOLVEstruct_t *SOLVEstruct,
	SuperLUStat_t *stat, int *info)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
    doublecomplex alpha = {1.0, 0.0};
    doublecomplex beta = {0.0, 0.0};
    doublecomplex zero = {0.0, 0.0};
    doublecomplex *lsum;  /* Local running sum of the updates to B-components */
    doublecomplex *x;     /* X component at step k. */
		    /* NOTE: x and lsum are of same size. */
    doublecomplex *lusup, *dest;
    doublecomplex *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin;
    doublecomplex *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    doublecomplex *Linv; /* Inverse of diagonal block */
    doublecomplex *Uinv; /* Inverse of diagonal block */
    int *ipiv;
    int_t *leaf_send;
    int_t nleaf_send, nleaf_send_tmp;
    int_t *root_send;
    int_t nroot_send, nroot_send_tmp;
    int_t  **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
    C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
    C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
    C_Tree  *URtree_ptr = Llu->URtree_ptr;
    int_t  *Urbs1; /* Number of row blocks in each block column of U. */
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t  kcol, krow, mycol, myrow;
    int_t  i, ii, il, j, jj, k, kk, lb, ljb, lk, lib, lptr, luptr, gb, nn;
    int_t  nb, nlb,nlb_nodiag, nub, nsupers, nsupers_j, nsupers_i,maxsuper;
    int_t  *xsup, *supno, *lsub, *usub;
    int_t  *ilsum;    /* Starting position of each supernode in lsum (LOCAL)*/
    int    Pc, Pr, iam;
    int    knsupc, nsupr, nprobe;
    int    nbtree, nrtree, outcount;
    int    ldalsum;   /* Number of lsum entries locally owned. */
    int    maxrecvsz, p, pi;
    int_t  **Lrowind_bc_ptr;
    doublecomplex **Lnzval_bc_ptr;
    doublecomplex **Linv_bc_ptr;
    doublecomplex **Uinv_bc_ptr;
    doublecomplex sum;
    MPI_Status status,status_on,statusx,statuslsum;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    SuperLUStat_t **stat_loc;

    double tmax;
    	/*-- Counts used for L-solve --*/
    int  *fmod;         /* Modification count for L-solve --
    			 Count the number of local block products to
    			 be summed into lsum[lk]. */
    int  fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx = Llu->nfrecvx; /* Number of X components to be recv'd. */
    int  nfrecvx_buf=0;
    int  *frecv;        /* Count of lsum[lk] contributions to be received
    			     from processes in this row.
    			     It is only valid on the diagonal processes. */
    int  frecv_tmp;
    int  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int  nfrecv = 0; /* Count of total messages to be recv'd. */
    int  nbrecv = 0; /* Count of total messages to be recv'd. */
    int  nleaf = 0, nroot = 0;
    int  nleaftmp = 0, nroottmp = 0;
    int_t  msgsize;
        /*-- Counts used for U-solve --*/
    int  *bmod;         /* Modification count for U-solve. */
    int  bmod_tmp;
    int  **bsendx_plist = Llu->bsendx_plist;
    int  nbrecvx = Llu->nbrecvx; /* Number of X components to be recv'd. */
    int  nbrecvx_buf=0;
    int  *brecv;        /* Count of modifications to be recv'd from
    			     processes in this row. */
    int_t  nbrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t flagx,flaglsum,flag;
    int_t *LBTree_active, *LRTree_active, *LBTree_finish, *LRTree_finish, *leafsups, *rootsups;
    int_t TAG;
    double t1_sol, t2_sol, t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    int_t gik,iklrow,fnz;

    int *mod_bit = Llu->mod_bit; /* flag contribution from each row block */
    int INFO, pad;
    int_t tmpresult;

    // #if ( PROFlevel>=1 )
    double t1, t2;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (doublecomplex);
    int Nwork;
    int_t procs = grid->nprow * grid->npcol;
    yes_no_t done;
    yes_no_t startforward;
    int nbrow;
    int_t  ik, rel, idx_r, jb, nrbl, irow, pc,iknsupc;
    int_t  lptr1_tmp, idx_i, idx_v,m;
    int_t ready;
    int thread_id = 0;
    yes_no_t empty;
    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1; //ceil(CACHELINE/(double)dword);
    aln_i = 1; //ceil(CACHELINE/(double)iword);
    int num_thread = 1;

    maxsuper = sp_ienv_dist(3, options);

//#ifdef _OPENMP
//#pragma omp threadprivate(thread_id)
//#endif

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
    }
#else
	num_thread=1;
#endif

#if ( PRNTlevel>=1 )
    if( grid->iam==0 ) {
	printf("num_thread: %5d\n", num_thread);
	fflush(stdout);
    }
#endif

    MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PZGSTRS", grid, -*info);
	return;
    }

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n-1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    Linv_bc_ptr = Llu->Linv_bc_ptr;
    Uinv_bc_ptr = Llu->Uinv_bc_ptr;
    nlb = CEILING( nsupers, Pr ); /* Number of local block rows. */

    stat->utime[SOL_COMM] = 0.0;
    stat->utime[SOL_GEMM] = 0.0;
    stat->utime[SOL_TRSM] = 0.0;
    stat->utime[SOL_TOT] = 0.0;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter pzgstrs()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PZGSTRS. */
    if ( !(fmod = int32Malloc_dist(nlb*aln_i)) )
	ABORT("Malloc fails for fmod[].");
    for (i = 0; i < nlb; ++i) fmod[i*aln_i] = Llu->fmod[i];
    if ( !(frecv = int32Calloc_dist(nlb)) )
	ABORT("Calloc fails for frecv[].");
    Llu->frecv = frecv;

    if ( !(leaf_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for leaf_send[].");
    nleaf_send=0;
    if ( !(root_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for root_send[].");
    nroot_send=0;

#ifdef _CRAY
    ftcs1 = _cptofcd("L", strlen("L"));
    ftcs2 = _cptofcd("N", strlen("N"));
    ftcs3 = _cptofcd("U", strlen("U"));
#endif


    /* Obtain ilsum[] and ldalsum for process column 0. */
    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist(3, options);
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX( XK_H, LSUM_H );
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;

#ifdef _OPENMP
    if ( !(lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(doublecomplex))))
	ABORT("Malloc fails for lsum[].");
#pragma omp parallel default(shared) private(ii,thread_id)
    {
	thread_id = omp_get_thread_num(); //mjc
	for (ii=0; ii<sizelsum; ii++)
    	    lsum[thread_id*sizelsum+ii]=zero;
    }
#else
    if ( !(lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(doublecomplex))))
  	    ABORT("Malloc fails for lsum[].");
    for ( ii=0; ii < sizelsum*num_thread; ii++ )
	lsum[ii]=zero;
#endif
    /* intermediate solution x[] vector has same structure as lsum[], see leading comment */
    if ( !(x = doublecomplexCalloc_dist(ldalsum * nrhs + nlb * XK_H)) )
	ABORT("Calloc fails for x[].");

    sizertemp=ldalsum * nrhs;
    sizertemp = ((sizertemp + (aln_d - 1)) / aln_d) * aln_d;
    if ( !(rtemp = (doublecomplex*)SUPERLU_MALLOC((sizertemp*num_thread + 1) * sizeof(doublecomplex))) )
	ABORT("Malloc fails for rtemp[].");
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii,thread_id)
    {
	thread_id=omp_get_thread_num();
	for ( ii=0; ii<sizertemp; ii++ )
		rtemp[thread_id*sizertemp+ii]=zero;
    }
#else
    for ( ii=0; ii<sizertemp*num_thread; ii++ )
	rtemp[ii]=zero;
#endif

    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }

#if ( DEBUGlevel>=2 )
    /* Dump the L factor using matlab triple-let format. */
    zDumpLblocks(iam, nsupers, grid, Glu_persist, Llu);
#endif

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/
    /* Redistribute B into X on the diagonal processes. */
    pzReDistribute_B_to_X(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
			  ScalePermstruct, Glu_persist, grid, SOLVEstruct);

#if ( PRNTlevel>=2 )
    t = SuperLU_timer_() - t;
    if ( !iam) printf(".. B to X redistribute time\t%8.4f\n", t);
    fflush(stdout);
    t = SuperLU_timer_();
#endif

    /* Set up the headers in lsum[]. */
    for (k = 0; k < nsupers; ++k) {
	krow = PROW( k, grid );
	if ( myrow == krow ) {
	    lk = LBi( k, grid );   /* Local block number. */
	    il = LSUM_BLK( lk );
	    lsum[il - LSUM_H].r = k;/* Block number prepended in the header.*/
	    lsum[il - LSUM_H].i = 0;
	}
    }

	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
			// printf("LBtree_ptr lk %5d\n",lk);
			//if(BcTree_IsRoot(LBtree_ptr[lk],'z')==NO){
			if(C_BcTree_IsRoot(&LBtree_ptr[lk])==NO){
				nbtree++;
				//if(BcTree_getDestCount(LBtree_ptr[lk],'z')>0)nfrecvx_buf++;
				if(LBtree_ptr[lk].destCnt_>0)nfrecvx_buf++;
			}
			//BcTree_allocateRequest(LBtree_ptr[lk],'z');
		}
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	leafsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for leafsups.");

	nrtree = 0;
	nleaf=0;
	nfrecvmod=0;

if(procs==1){
	for (lk=0;lk<nsupers_i;++lk){
		gb = myrow+lk*grid->nprow;  /* not sure */
		if(gb<nsupers){
			if (fmod[lk*aln_i]==0){
				leafsups[nleaf]=gb;
				++nleaf;
			}
		}
	}
}else{
	for (lk=0;lk<nsupers_i;++lk){
		if(LRtree_ptr[lk].empty_==NO){
			nrtree++;
			//RdTree_allocateRequest(LRtree_ptr[lk],'z');
			//frecv[lk] = RdTree_GetDestCount(LRtree_ptr[lk],'z');
			frecv[lk] = LRtree_ptr[lk].destCnt_;
			nfrecvmod += frecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (fmod[lk*aln_i]==0){
						leafsups[nleaf]=gb;
						++nleaf;
					}
				}
			}
		}
	}
}

#ifdef _OPENMP
#pragma omp simd
#endif
	for (i = 0; i < nlb; ++i) fmod[i*aln_i] += frecv[i];

	if ( !(recvbuf_BC_fwd = (doublecomplex*)SUPERLU_MALLOC(maxrecvsz*(nfrecvx+1) * sizeof(doublecomplex))) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nfrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword+(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*2.0*iword+ nsupers_i*iword + sizelsum*num_thread * dword*2.0 + (ldalsum * nrhs + nlb * XK_H) *dword*2.0 + (sizertemp*num_thread + 1)*dword*2.0+maxrecvsz*(nfrecvx+1)*dword*2.0, stat);	//account for fmod, frecv, leaf_send, root_send, leafsups, recvbuf_BC_fwd	, lsum, x, rtemp

#if ( DEBUGlevel>=2 )
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
	fflush(stdout);
#endif

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Setup L-solve time\t%8.4f\n", t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

#if ( VAMPIR>=1 )
	// VT_initialize();
	VT_traceon();
#endif

#ifdef USE_VTUNE
	__SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
	__itt_resume(); // start VTune, again use 2 underscores
#endif

	/* ---------------------------------------------------------
	   Solve the leaf nodes first by all the diagonal processes.
	   --------------------------------------------------------- */
#if ( DEBUGlevel>=2 )
	printf("(%2d) nleaf %4d\n", iam, nleaf);
	fflush(stdout);
#endif

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
	    int thread_id = omp_get_thread_num();
#else
	{
 	    thread_id=0;
#endif
		{

            if (Llu->inv == 1) { /* Diagonal is inverted. */

#ifdef _OPENMP
#pragma	omp for firstprivate(nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
		for (jj=0;jj<nleaf;jj++){
		    k=leafsups[jj];

// #ifdef _OPENMP
// #pragma omp task firstprivate (k,nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,knsupc,lk,luptr,lsub,nsupr,lusup,thread_id,t1,t2,Linv,i,lib,rtemp_loc)
// #endif
   		    {

#if ( PROFlevel>=1 )
			TIC(t1);
#endif
			rtemp_loc = &rtemp[sizertemp* thread_id];

			knsupc = SuperSize( k );
			lk = LBi( k, grid );

			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise. */
			lsub = Lrowind_bc_ptr[lk];
			lusup = Lnzval_bc_ptr[lk];

			nsupr = lsub[1];
			Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
			CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
					&alpha, Linv, &knsupc, &x[ii],
					&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
			zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
					&alpha, Linv, &knsupc, &x[ii],
					&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
			zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
					&alpha, Linv, &knsupc, &x[ii],
					&knsupc, &beta, rtemp_loc, &knsupc );
#endif

			for (i=0 ; i<knsupc*nrhs ; i++){
				z_copy(&x[ii+i],&rtemp_loc[i]);
			}

			// for (i=0 ; i<knsupc*nrhs ; i++){
			// printf("x_l: %f %f\n",x[ii+i].r,x[ii+i].i);
			// fflush(stdout);
			// }


#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

			stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
				+ 10 * knsupc * nrhs; /* complex division */
			// --nleaf;
#if ( DEBUGlevel>=2 )
			printf("(%2d) Solve X[%2d]\n", iam, k);
#endif
			/*
			 * Send Xk to process column Pc[k].
			 */

			if(LBtree_ptr[lk].empty_==NO){
				lib = LBi( k, grid ); /* Local block number, row-wise. */
				ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
				nleaf_send_tmp = ++nleaf_send;
				leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
				// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],'z');
			}
		    }
	     }
	} else { /* Diagonal is not inverted. */
#ifdef _OPENMP
#pragma	omp	for firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
	    for (jj=0;jj<nleaf;jj++) {
		k=leafsups[jj];
		{

#if ( PROFlevel>=1 )
		    TIC(t1);
#endif
		    rtemp_loc = &rtemp[sizertemp* thread_id];

		    knsupc = SuperSize( k );
		    lk = LBi( k, grid );

		    ii = X_BLK( lk );
		    lk = LBj( k, grid ); /* Local block number, column-wise. */
		    lsub = Lrowind_bc_ptr[lk];
		    lusup = Lnzval_bc_ptr[lk];

		    nsupr = lsub[1];

#ifdef _CRAY
   		    CTRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
		    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
 		    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc);
#endif

#if ( PROFlevel>=1 )
		    TOC(t2, t1);
		    stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

		    stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
				+ 10 * knsupc * nrhs; /* complex division */

		    // --nleaf;
#if ( DEBUGlevel>=2 )
		    printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

		    /*
		     * Send Xk to process column Pc[k].
		     */

		    if (LBtree_ptr[lk].empty_==NO) {
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
			nleaf_send_tmp = ++nleaf_send;
			leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
		    }
		    } /* end a block */
		} /* end for jj ... */
	    } /* end else ... diagonal is not inverted */
	  }
	} /* end parallel region */

	jj=0;

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{

#ifdef _OPENMP
#pragma omp master
#endif
	    {

#ifdef _OPENMP
#pragma	omp taskloop private (k,ii,lk,thread_id) num_tasks(num_thread*8) nogroup
#endif
		for (jj=0;jj<nleaf;jj++){
		    k=leafsups[jj];

		    {
#ifdef _OPENMP
                        thread_id=omp_get_thread_num();
#else
                        thread_id=0;
#endif
			/* Diagonal process */
			lk = LBi( k, grid );
			ii = X_BLK( lk );
			/*
			 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
			 */
			zlsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, fmod, xsup, grid, Llu, stat_loc, leaf_send, &nleaf_send,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
		    }
		} /* for jj ... */
	    }

	}

	for (i=0;i<nleaf_send;i++){
		lk = leaf_send[i*aln_i];
		if(lk>=0){ // this is a bcast forwarding
			gb = mycol+lk*grid->npcol;  /* not sure */
			lib = LBi( gb, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );
			//BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z');
			C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
			
		}else{ // this is a reduce forwarding
			lk = -lk - 1;
			il = LSUM_BLK( lk );
			//RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
			C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
			
		}
	}


#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

	/* -----------------------------------------------------------
	   Compute the internal nodes asynchronously by all processes.
	   ----------------------------------------------------------- */

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
        int thread_id = omp_get_thread_num();
#else
	{
	thread_id = 0;
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		{
		for ( nfrecv =0; nfrecv<nfrecvx+nfrecvmod;nfrecv++) { /* While not finished. */
			thread_id = 0;
#if ( PROFlevel>=1 )
		   	TIC(t1);
			// msgcnt[1] = maxrecvsz;
#endif

			recvbuf0 = &recvbuf_BC_fwd[nfrecvx_buf*maxrecvsz];

			/* Receive a message. */
			MPI_Recv( recvbuf0, maxrecvsz, SuperLU_MPI_DOUBLE_COMPLEX,
				MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );
			// MPI_Irecv(recvbuf0,maxrecvsz,SuperLU_MPI_DOUBLE_COMPLEX,MPI_ANY_SOURCE,MPI_ANY_TAG,grid->comm,&req);
			// ready=0;
			// while(ready==0){
			// MPI_Test(&req,&ready,&status);
			// #pragma omp taskyield
			// }

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_COMM] += t2;

			msg_cnt += 1;
			msg_vol += maxrecvsz * dword;
#endif

			{

			k = (*recvbuf0).r;

#if ( DEBUGlevel>=2 )
			printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
#endif

			if(status.MPI_TAG==BC_L){
				// --nfrecvx;
				nfrecvx_buf++;
				{
				lk = LBj( k, grid );    /* local block number */

				if(LBtree_ptr[lk].destCnt_>0){

					//BcTree_forwardMessageSimple(LBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z');
					C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], recvbuf0, LBtree_ptr[lk].msgSize_*nrhs+XK_H);
					// nfrecvx_buf++;
				}

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */

				lk = LBj( k, grid ); /* Local block number, column-wise. */
				lsub = Lrowind_bc_ptr[lk];
				lusup = Lnzval_bc_ptr[lk];
				if ( lsub ) {
					krow = PROW( k, grid );
					if(myrow==krow){
					    nb = lsub[0] - 1;
					    knsupc = SuperSize( k );
					    ii = X_BLK( LBi( k, grid ) );
					    xin = &x[ii];
				        }else{
					    nb   = lsub[0];
					    knsupc = SuperSize( k );
					    xin = &recvbuf0[XK_H] ;
					}
					zlsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k, 
					    fmod, nb, xsup, grid, Llu, 
					    stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);

				} /* if lsub */
			    }
			}else if(status.MPI_TAG==RD_L){
				// --nfrecvmod;
				lk = LBi( k, grid ); /* Local block number, row-wise. */

				knsupc = SuperSize( k );
				tempv = &recvbuf0[LSUM_H];
				il = LSUM_BLK( lk );
				RHS_ITERATE(j) {
				for (i = 0; i < knsupc; ++i)
					z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum], &lsum[i + il + j*knsupc + thread_id*sizelsum], &tempv[i + j*knsupc]);
				} 

			// #ifdef _OPENMP
			// #pragma omp atomic capture
			// #endif
				fmod_tmp=--fmod[lk*aln_i];
				{
				thread_id = 0;
				rtemp_loc = &rtemp[sizertemp* thread_id];
				if ( fmod_tmp==0 ) {
				    if(C_RdTree_IsRoot(&LRtree_ptr[lk])==YES){
				    // ii = X_BLK( lk );
					knsupc = SuperSize( k );
					for (ii=1;ii<num_thread;ii++)
					    for (jj=0;jj<knsupc*nrhs;jj++)
						z_add(&lsum[il + jj ], &lsum[il + jj ],
							 &lsum[il + jj + ii*sizelsum]);
					ii = X_BLK( lk );
					RHS_ITERATE(j)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
					    for (i = 0; i < knsupc; ++i)
					        z_add(&x[i + ii + j*knsupc],
						       &x[i + ii + j*knsupc],
						       &lsum[i + il + j*knsupc] );

					// fmod[lk] = -1; /* Do not solve X[k] in the future. */
					lk = LBj( k, grid ); /* Local block number, column-wise. */
					lsub = Lrowind_bc_ptr[lk];
					lusup = Lnzval_bc_ptr[lk];
					nsupr = lsub[1];

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					if(Llu->inv == 1){
						Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
						CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Linv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
						zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
						zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#endif
					#ifdef _OPENMP
					#pragma omp simd
					#endif
						for (i=0 ; i<knsupc*nrhs ; i++){
							z_copy(&x[ii+i],&rtemp_loc[i]);
						}
					} else{ /* Llu->innv == 0 */
#ifdef _CRAY
					    CTRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
					    lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
					    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					      lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
					    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					        lusup, &nsupr, &x[ii], &knsupc);
#endif
					} /* end if-else */

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif

					stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
					+ 10 * knsupc * nrhs; /* complex division */

#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */
					if(LBtree_ptr[lk].empty_==NO){
					    //BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z');
					    C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
					    
					}
					/*
					 * Perform local block modifications.
					 */
					lk = LBj( k, grid ); /* Local block number, column-wise. */
					lsub = Lrowind_bc_ptr[lk];
					lusup = Lnzval_bc_ptr[lk];
					if ( lsub ) {
						krow = PROW( k, grid );
						nb = lsub[0] - 1;
						knsupc = SuperSize( k );
						ii = X_BLK( LBi( k, grid ) );
						xin = &x[ii];
						zlsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k,
							fmod, nb, xsup, grid, Llu,
							stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
					} /* if lsub */
					// }

				    }else{ /* fmod_tmp != 0 */
					il = LSUM_BLK( lk );
					knsupc = SuperSize( k );
					for (ii=1;ii<num_thread;ii++)
					    for (jj=0;jj<knsupc*nrhs;jj++)
						z_add(&lsum[il + jj ],
							  &lsum[il + jj ],
							  &lsum[il + jj + ii*sizelsum]);
					//RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
					C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
				    } /* end if-else */

				    }

				} /* end else MPI_TAG==RD_L */
			    } /* check Tag */
			} /* end for nfrecv ... */
                    } /* while not finished ... */
       	    }
        } // end of parallel

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	stat->utime[SOL_TOT] += t;
	if ( !iam ) {
		printf(".. L-solve time\t%8.4f\n", t);
		fflush(stdout);
	}

	MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid->comm);
	if ( !iam ) {
		printf(".. L-solve time (MAX) \t%8.4f\n", tmax);
		fflush(stdout);
	}
	t = SuperLU_timer_();
#endif

#if ( DEBUGlevel==2 )
	{
		printf("(%d) .. After L-solve: y =\n", iam);
		for (i = 0, k = 0; k < nsupers; ++k) {
			krow = PROW( k, grid );
			kcol = PCOL( k, grid );
			if ( myrow == krow && mycol == kcol ) { /* Diagonal process */
				knsupc = SuperSize( k );
				lk = LBi( k, grid );
				ii = X_BLK( lk );
				for (j = 0; j < knsupc; ++j)
					printf("\t(%d)\t%4d\t%.10f\n", iam, xsup[k]+j, x[ii+j]);
					fflush(stdout);
				}
			MPI_Barrier( grid->comm );
		}
	}
#endif

	SUPERLU_FREE(fmod);
	SUPERLU_FREE(frecv);
	SUPERLU_FREE(leaf_send);
	SUPERLU_FREE(leafsups);
	SUPERLU_FREE(recvbuf_BC_fwd);
	log_memory(-nlb*aln_i*iword-nlb*iword-(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword- nsupers_i*iword -maxrecvsz*(nfrecvx+1)*dword*2.0, stat);	//account for fmod, frecv, leaf_send, leafsups, recvbuf_BC_fwd

	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
			// if(BcTree_IsRoot(LBtree_ptr[lk],'z')==YES){
			//BcTree_waitSendRequest(LBtree_ptr[lk],'z');
			C_BcTree_waitSendRequest(&LBtree_ptr[lk]);
			// }
			// deallocate requests here
		}
	}

	for (lk=0;lk<nsupers_i;++lk){
		if(LRtree_ptr[lk].empty_==NO){
			C_RdTree_waitSendRequest(&LRtree_ptr[lk]);
			// deallocate requests here
		}
	}
	MPI_Barrier( grid->comm );

#if ( VAMPIR>=1 )
	VT_traceoff();
	VT_finalize();
#endif


	/*---------------------------------------------------
	 * Back solve Ux = y.
	 *
	 * The Y components from the forward solve is already
	 * on the diagonal processes.
	 *---------------------------------------------------*/

	/* Save the count to be altered so it can be used by
	   subsequent call to PZGSTRS. */
	if ( !(bmod = int32Malloc_dist(nlb*aln_i)) )
		ABORT("Malloc fails for bmod[].");
	for (i = 0; i < nlb; ++i) bmod[i*aln_i] = Llu->bmod[i];
	if ( !(brecv = int32Calloc_dist(nlb)) )
		ABORT("Calloc fails for brecv[].");
	Llu->brecv = brecv;

	k = SUPERLU_MAX( Llu->nfsendx, Llu->nbsendx ) + nlb;

	/* Re-initialize lsum to zero. Each block header is already in place. */

#ifdef _OPENMP

#pragma omp parallel default(shared) private(ii)
	{
                int thread_id = omp_get_thread_num();
		for(ii=0;ii<sizelsum;ii++)
			lsum[thread_id*sizelsum+ii]=zero;
	}
  	/* Set up the headers in lsum[]. */
//#pragma omp simd lastprivate(krow,lk,il)
    	for (k = 0; k < nsupers; ++k) {
	    krow = PROW( k, grid );
	    if ( myrow == krow ) {
	       lk = LBi( k, grid );   /* Local block number. */
	       il = LSUM_BLK( lk );
	       lsum[il - LSUM_H].r = k;/* Block number prepended in the header.*/
	       lsum[il - LSUM_H].i = 0;
	    }
    	}

#else
	for (k = 0; k < nsupers; ++k) {
	    krow = PROW( k, grid );
	    if ( myrow == krow ) {
		knsupc = SuperSize( k );
		lk = LBi( k, grid );
		il = LSUM_BLK( lk );
		dest = &lsum[il];

		for (jj = 0; jj < num_thread; ++jj) {
		    RHS_ITERATE(j) {
			for (i = 0; i < knsupc; ++i) dest[i + j*knsupc + jj*sizelsum] = zero;
		    }
		}
	    }
	}
#endif

#if ( DEBUGlevel>=2 )
        nub = CEILING( nsupers, Pc ); /* Number of local block columns. */
        for (p = 0; p < Pr*Pc; ++p) {
	    if (iam == p) {
		printf("(%2d) .. Ublocks %d, nub %d\n",iam,Ublocks,nub); fflush(stdout);
		for (lb = 0; lb < nub; ++lb) {
		    printf("(%2d) Local col %2d: # row blocks %2d\n",
				iam, lb, Urbs[lb]); fflush(stdout);
		    if ( Urbs[lb] ) {
			for (i = 0; i < Urbs[lb]; ++i)
			    printf("(%2d) .. row blk %2d:\
				    lbnum %d, indpos %d, valpos %d\n",
				    iam, i,
				    Ucb_indptr[lb][i].lbnum,
				    Ucb_indptr[lb][i].indpos,
				    Ucb_valptr[lb][i]);
		     }
		}
	    }
	    MPI_Barrier( grid->comm );
	}
	for (p = 0; p < Pr*Pc; ++p) {
		if ( iam == p ) {
			printf("\n(%d) bsendx_plist[][]", iam);
			for (lb = 0; lb < nub; ++lb) {
				printf("\n(%d) .. local col %2d: ", iam, lb);
				for (i = 0; i < Pr; ++i)
					printf("%4d", bsendx_plist[lb][i]);
			}
			printf("\n");
		}
		MPI_Barrier( grid->comm );
	}
#endif /* DEBUGlevel */

	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
	    if(UBtree_ptr[lk].empty_==NO){
		// printf("UBtree_ptr lk %5d\n",lk);
		if(C_BcTree_IsRoot(&UBtree_ptr[lk])==NO){
			nbtree++;
			if(UBtree_ptr[lk].destCnt_>0)nbrecvx_buf++;
		}
		//BcTree_allocateRequest(UBtree_ptr[lk],'z');
	    }
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	rootsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for rootsups.");

	nrtree = 0;
	nroot=0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			//RdTree_allocateRequest(URtree_ptr[lk],'z');
			brecv[lk] = URtree_ptr[lk].destCnt_;
			nbrecvmod += brecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (bmod[lk*aln_i]==0){
						rootsups[nroot]=gb;
						++nroot;
					}
				}
			}
		}
	}

	for (i = 0; i < nlb; ++i) bmod[i*aln_i] += brecv[i];
	// for (i = 0; i < nlb; ++i)printf("bmod[i]: %5d\n",bmod[i]);

	if ( !(recvbuf_BC_fwd = (doublecomplex*)SUPERLU_MALLOC(maxrecvsz*(nbrecvx+1) * sizeof(doublecomplex))) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nbrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword*2.0, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd

#if ( DEBUGlevel>=2 )
	printf("(%2d) nbrecvx %4d,  nbrecvmod %4d,  nroot %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nbrecvx, nbrecvmod, nroot, nbtree, nrtree);
	fflush(stdout);
#endif


#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Setup U-solve time\t%8.4f\n", t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

	/*
	 * Solve the roots first by all the diagonal processes.
	 */
#if ( DEBUGlevel>=2 )
	printf("(%2d) nroot %4d\n", iam, nroot);
	fflush(stdout);
#endif

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	    {
#ifdef _OPENMP
#pragma	omp taskloop firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,jj,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Uinv,i,lib,rtemp_loc,nroot_send_tmp,thread_id) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
#if ( PROFlevel>=1 )
			TIC(t1);
#endif
#ifdef _OPENMP
			thread_id = omp_get_thread_num ();
#else
			thread_id = 0;
#endif
			rtemp_loc = &rtemp[sizertemp* thread_id];

			knsupc = SuperSize( k );
			lk = LBi( k, grid ); /* Local block number, row-wise. */

			// bmod[lk] = -1;       /* Do not solve X[k] in the future. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
			lsub = Lrowind_bc_ptr[lk];
			lusup = Lnzval_bc_ptr[lk];
			nsupr = lsub[1];

			if(Llu->inv == 1){

				Uinv = Uinv_bc_ptr[lk];
#ifdef _CRAY
				CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#endif
				for (i=0 ; i<knsupc*nrhs ; i++){
					z_copy(&x[ii+i],&rtemp_loc[i]);
				}
			}else{
#ifdef _CRAY
				CTRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#endif
			}

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
			stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc + 1) * nrhs
			+ 10 * knsupc * nrhs; /* complex division */

#if ( DEBUGlevel>=2 )
			printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

			/*
			 * Send Xk to process column Pc[k].
			 */

			if(UBtree_ptr[lk].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
				nroot_send_tmp = ++nroot_send;
				root_send[(nroot_send_tmp-1)*aln_i] = lk;

			}
		} /* for jj ... */
	    } /* omp master region */
	} /* omp parallel region */


#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	    {
#ifdef _OPENMP
#pragma	omp taskloop private (ii,jj,k,lk,thread_id) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
			lk = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
#ifdef _OPENMP
                        thread_id=omp_get_thread_num();
#else
                        thread_id=0;
#endif
			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 */
			if ( Urbs[lk] )
			    zlsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
					Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
					stat_loc, root_send, &nroot_send, sizelsum,sizertemp,thread_id,num_thread);

		} /* for jj ... */

	    } /* omp master region */
	} /* omp parallel */

for (i=0;i<nroot_send;i++){
	lk = root_send[(i)*aln_i];
	if(lk>=0){ // this is a bcast forwarding
		gb = mycol+lk*grid->npcol;  /* not sure */
		lib = LBi( gb, grid ); /* Local block number, row-wise. */
		ii = X_BLK( lib );
		//BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z');
		C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
	}else{ // this is a reduce forwarding
		lk = -lk - 1;
		il = LSUM_BLK( lk );
		//RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
		C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
	}
}

	/*
	 * Compute the internal nodes asychronously by all processes.
	 */

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
	    int thread_id=omp_get_thread_num();
#else 
        {
	    thread_id = 0;
#endif
#ifdef _OPENMP
#pragma omp master
#endif
	    for ( nbrecv =0; nbrecv<nbrecvx+nbrecvmod;nbrecv++) { /* While not finished. */
		// printf("iam %4d nbrecv %4d nbrecvx %4d nbrecvmod %4d\n", iam, nbrecv, nbrecvxnbrecvmod);
		// fflush(stdout);

		thread_id = 0;
#if ( PROFlevel>=1 )
			TIC(t1);
#endif

		recvbuf0 = &recvbuf_BC_fwd[nbrecvx_buf*maxrecvsz];

		/* Receive a message. */
		MPI_Recv( recvbuf0, maxrecvsz, SuperLU_MPI_DOUBLE_COMPLEX,
			MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_COMM] += t2;
			msg_cnt += 1;
			msg_vol += maxrecvsz * dword;
#endif

		k = (*recvbuf0).r;
#if ( DEBUGlevel>=2 )
		printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
		fflush(stdout);
#endif
		if(status.MPI_TAG==BC_U){
		    // --nfrecvx;
		    nbrecvx_buf++;
		    lk = LBj( k, grid );    /* local block number */
		    if(UBtree_ptr[lk].destCnt_>0){

			// BcTree_forwardMessageSimple(UBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(UBtree_ptr[lk],'d')*nrhs+XK_H,'d');
			C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], recvbuf0, UBtree_ptr[lk].msgSize_*nrhs+XK_H);
			// nfrecvx_buf++;
		    }

		    /*
		     * Perform local block modifications: lsum[i] -= U_i,k * X[k]
		     */

		    lk = LBj( k, grid ); /* Local block number, column-wise. */
		    zlsum_bmod_inv_master(lsum, x, &recvbuf0[XK_H], rtemp, nrhs, k, bmod, Urbs,
				Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
				stat_loc, sizelsum,sizertemp,thread_id,num_thread);
		}else if(status.MPI_TAG==RD_U){

		    lk = LBi( k, grid ); /* Local block number, row-wise. */

		    knsupc = SuperSize( k );
		    tempv = &recvbuf0[LSUM_H];
		    il = LSUM_BLK( lk );
		    RHS_ITERATE(j) {
			for (i = 0; i < knsupc; ++i)
			    z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum],
				  &lsum[i + il + j*knsupc + thread_id*sizelsum],
				  &tempv[i + j*knsupc]);
		    }
		// #ifdef _OPENMP
		// #pragma omp atomic capture
		// #endif
		    bmod_tmp=--bmod[lk*aln_i];
		    thread_id = 0;
		    rtemp_loc = &rtemp[sizertemp* thread_id];
		    if ( bmod_tmp==0 ) {
			//if(RdTree_IsRoot(URtree_ptr[lk],'z')==YES){
			if(C_RdTree_IsRoot(&URtree_ptr[lk])==YES){

			    knsupc = SuperSize( k );
			    for (ii=1;ii<num_thread;ii++)
				for (jj=0;jj<knsupc*nrhs;jj++)
					z_add(&lsum[il+ jj ],
					  	  &lsum[il+ jj ],
						  &lsum[il + jj + ii*sizelsum]);

			    ii = X_BLK( lk );
			    RHS_ITERATE(j)
				for (i = 0; i < knsupc; ++i)
				    z_add(&x[i + ii + j*knsupc],
						  &x[i + ii + j*knsupc],
						  &lsum[i + il + j*knsupc] );

			    lk = LBj( k, grid ); /* Local block number, column-wise. */
			    lsub = Lrowind_bc_ptr[lk];
			    lusup = Lnzval_bc_ptr[lk];
			    nsupr = lsub[1];

			    if(Llu->inv == 1){

				Uinv = Uinv_bc_ptr[lk];

#ifdef _CRAY
				CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#endif

			#ifdef _OPENMP
			#pragma omp simd
			#endif
				for (i=0 ; i<knsupc*nrhs ; i++){
					z_copy(&x[ii+i],&rtemp_loc[i]);
				}
			    }else{ /* Llu->inv == 0 */
#ifdef _CRAY
				CTRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc);
#endif
			    }

#if ( PROFlevel>=1 )
			    TOC(t2, t1);
			    stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
			    stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc + 1) * nrhs
					+ 10 * knsupc * nrhs; /* complex division */

#if ( DEBUGlevel>=2 )
			    printf("(%2d) Solve X[%2d]\n", iam, k);
#endif
			    /*
			     * Send Xk to process column Pc[k].
			     */
			    if(UBtree_ptr[lk].empty_==NO){
				//BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z');
				C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
			    }

			    /*
			     * Perform local block modifications:
			     *         lsum[i] -= U_i,k * X[k]
			     */
			    if ( Urbs[lk] )
				zlsum_bmod_inv_master(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
					Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
					stat_loc, sizelsum,sizertemp,thread_id,num_thread);

			    }else{
				il = LSUM_BLK( lk );
				knsupc = SuperSize( k );

				for (ii=1;ii<num_thread;ii++)
				    for (jj=0;jj<knsupc*nrhs;jj++)
					z_add(&lsum[il+ jj ], &lsum[il+ jj ],
						  &lsum[il + jj + ii*sizelsum]);

				//RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
				C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
			    } /* end if-else */
			} /* end if bmod_tmp==0 */
		    } /* end if MPI_TAG==RD_U */
		} /* while not finished ... */
	} /* end parallel region */

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	stat->utime[SOL_TOT] += t;
	if ( !iam ) printf(".. U-solve time\t%8.4f\n", t);
	MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid->comm);
	if ( !iam ) {
		printf(".. U-solve time (MAX) \t%8.4f\n", tmax);
		fflush(stdout);
	}
	t = SuperLU_timer_();
#endif

#if ( DEBUGlevel>=2 )
        {
	    doublecomplex *x_col;
	    int diag;
	    printf("\n(%d) .. After U-solve: x (ON DIAG PROCS) = \n", iam);
	    ii = 0;
	    for (k = 0; k < nsupers; ++k) {
	    	knsupc = SuperSize( k );
		krow = PROW( k, grid );
		kcol = PCOL( k, grid );
		diag = PNUM( krow, kcol, grid);
		if ( iam == diag ) { /* Diagonal process. */
		   lk = LBi( k, grid );
		   jj = X_BLK( lk );
		   x_col = &x[jj];
		   RHS_ITERATE(j) {
		       for (i = 0; i < knsupc; ++i) { /* X stored in blocks */
			   printf("\t(%d)\t%4d\t%.10f\n", iam, xsup[k]+i, x_col[i]);
		       }      
		       x_col += knsupc;
		   }
		}
		ii += knsupc;
	    } /* for k ... */
	}
#endif

	pzReDistribute_X_to_B(n, B, m_loc, ldb, fst_row, nrhs, x, ilsum,
				ScalePermstruct, Glu_persist, grid, SOLVEstruct);

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. X to B redistribute time\t%8.4f\n", t);
	t = SuperLU_timer_();
#endif

	double tmp1=0;
	double tmp2=0;
	double tmp3=0;
	double tmp4=0;
	for(i=0;i<num_thread;i++){
		tmp1 = SUPERLU_MAX(tmp1,stat_loc[i]->utime[SOL_TRSM]);
		tmp2 = SUPERLU_MAX(tmp2,stat_loc[i]->utime[SOL_GEMM]);
		tmp3 = SUPERLU_MAX(tmp3,stat_loc[i]->utime[SOL_COMM]);
		tmp4 += stat_loc[i]->ops[SOLVE];
#if ( PRNTlevel>=2 )
		if(iam==0)printf("thread %5d gemm %9.5f\n",i,stat_loc[i]->utime[SOL_GEMM]);
#endif
	}

	stat->utime[SOL_TRSM] += tmp1;
	stat->utime[SOL_GEMM] += tmp2;
	stat->utime[SOL_COMM] += tmp3;
	stat->ops[SOLVE]+= tmp4;

	/* Deallocate storage. */
	for(i=0;i<num_thread;i++){
		PStatFree(stat_loc[i]);
		SUPERLU_FREE(stat_loc[i]);
	}
	SUPERLU_FREE(stat_loc);
	SUPERLU_FREE(rtemp);
	SUPERLU_FREE(lsum);
	SUPERLU_FREE(x);

	SUPERLU_FREE(bmod);
	SUPERLU_FREE(brecv);
	SUPERLU_FREE(root_send);

	SUPERLU_FREE(rootsups);
	SUPERLU_FREE(recvbuf_BC_fwd);

	log_memory(-nlb*aln_i*iword-nlb*iword - nsupers_i*iword - (CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword - maxrecvsz*(nbrecvx+1)*dword*2.0 - sizelsum*num_thread * dword*2.0 - (ldalsum * nrhs + nlb * XK_H) *dword*2.0 - (sizertemp*num_thread + 1)*dword*2.0, stat);	//account for bmod, brecv, root_send, rootsups, recvbuf_BC_fwd,rtemp,lsum,x

	for (lk=0;lk<nsupers_j;++lk){
		if(UBtree_ptr[lk].empty_==NO){
			// if(BcTree_IsRoot(LBtree_ptr[lk],'z')==YES){
			C_BcTree_waitSendRequest(&UBtree_ptr[lk]);
			// }
			// deallocate requests here
		}
	}

	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			C_RdTree_waitSendRequest(&URtree_ptr[lk]);
			// deallocate requests here
		}
	}
	MPI_Barrier( grid->comm );


#if ( PROFlevel>=2 )
	{
		float msg_vol_max, msg_vol_sum, msg_cnt_max, msg_cnt_sum;
		MPI_Reduce (&msg_cnt, &msg_cnt_sum,
				1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
		MPI_Reduce (&msg_cnt, &msg_cnt_max,
				1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
		MPI_Reduce (&msg_vol, &msg_vol_sum,
				1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
		MPI_Reduce (&msg_vol, &msg_vol_max,
				1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
		if (!iam) {
			printf ("\tPZGSTRS comm stat:"
				"\tAvg\tMax\t\tAvg\tMax\n"
				"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
				msg_cnt_sum / Pr / Pc, msg_cnt_max,
				msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
		}
	}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pzgstrs()");
#endif

#if ( PRNTlevel>=2 )
    float for_lu, total, max, avg, temp;
    superlu_dist_mem_usage_t num_mem_usage;

    zQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
    temp = num_mem_usage.total;

    MPI_Reduce( &temp, &max, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm );
    MPI_Reduce( &temp, &avg, 1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
    if (!iam) {
	printf("\n** Memory Usage **********************************\n");
        printf("** Total highmark (MB):\n"
	       "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
	       avg * 1e-6,
	       avg / grid->nprow / grid->npcol * 1e-6,
	       max * 1e-6);
	printf("**************************************************\n");
	fflush(stdout);
    }
#endif

    return;
} /* PZGSTRS */

