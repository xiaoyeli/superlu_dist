/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Several matrix utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 * Last modified:
 * 	December 28, 2022
 * </pre>
 */

#include <math.h>
#include "superlu_zdefs.h"
#ifdef GPU_ACC
#include "gpu_api_utils.h"
#endif

/*! \brief Gather A from the distributed compressed row format to global A in compressed column format.
 */
int pzCompRow_loc_to_CompCol_global
(
 int_t need_value, /* Input. Whether need to gather numerical values */
 SuperMatrix *A,   /* Input. Distributed matrix in NRformat_loc format. */
 gridinfo_t *grid, /* Input */
 SuperMatrix *GA   /* Output */
)
{
    NRformat_loc *Astore;
    NCformat *GAstore;
    doublecomplex *a, *a_loc;
    int_t *colind, *rowptr;
    int_t *colptr_loc, *rowind_loc;
    int_t m_loc, n, i, j, k, l;
    int_t colnnz, fst_row, nnz_loc, nnz;
    doublecomplex *a_recv;  /* Buffer to receive the blocks of values. */
    doublecomplex *a_buf;   /* Buffer to merge blocks into block columns. */
    int_t *itemp;
    int_t *colptr_send; /* Buffer to redistribute the column pointers of the
			   local block rows.
			   Use n_loc+1 pointers for each block. */
    int_t *colptr_blk;  /* The column pointers for each block, after
			   redistribution to the local block columns.
			   Use n_loc+1 pointers for each block. */
    int_t *rowind_recv; /* Buffer to receive the blocks of row indices. */
    int_t *rowind_buf;  /* Buffer to merge blocks into block columns. */
    int_t *fst_rows, *n_locs;
    int   *sendcnts, *sdispls, *recvcnts, *rdispls, *itemp_32;
    int   it, n_loc, procs;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Enter pzCompRow_loc_to_CompCol_global");
#endif

    /* Initialization. */
    n = A->ncol;
    Astore = (NRformat_loc *) A->Store;
    nnz_loc = Astore->nnz_loc;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    a = Astore->nzval;
    rowptr = Astore->rowptr;
    colind = Astore->colind;
    n_loc = m_loc; /* NOTE: CURRENTLY ONLY WORK FOR SQUARE MATRIX */

    /* ------------------------------------------------------------
       FIRST PHASE: TRANSFORM A INTO DISTRIBUTED COMPRESSED COLUMN.
       ------------------------------------------------------------*/
    zCompRow_to_CompCol_dist(m_loc, n, nnz_loc, a, colind, rowptr, &a_loc,
                             &rowind_loc, &colptr_loc);
    /* Change local row index numbers to global numbers. */
    for (i = 0; i < nnz_loc; ++i) rowind_loc[i] += fst_row;

#if ( DEBUGlevel>=2 )
    printf("Proc %d\n", grid->iam);
    PrintInt10("rowind_loc", nnz_loc, rowind_loc);
    PrintInt10("colptr_loc", n+1, colptr_loc);
#endif

    procs = grid->nprow * grid->npcol;
    if ( !(fst_rows = (int_t *) intMalloc_dist(2*procs)) )
	  ABORT("Malloc fails for fst_rows[]");
    n_locs = fst_rows + procs;
    MPI_Allgather(&fst_row, 1, mpi_int_t, fst_rows, 1, mpi_int_t,
		  grid->comm);
    for (i = 0; i < procs-1; ++i) n_locs[i] = fst_rows[i+1] - fst_rows[i];
    n_locs[procs-1] = n - fst_rows[procs-1];
    if ( !(recvcnts = SUPERLU_MALLOC(5*procs * sizeof(int))) )
	  ABORT("Malloc fails for recvcnts[]");
    sendcnts = recvcnts + procs;
    rdispls = sendcnts + procs;
    sdispls = rdispls + procs;
    itemp_32 = sdispls + procs;

    /* All-to-all transfer column pointers of each block.
       Now the matrix view is P-by-P block-partition. */
    /* n column starts for each column, and procs column ends for each block */
    if ( !(colptr_send = intMalloc_dist(n + procs)) )
	   ABORT("Malloc fails for colptr_send[]");
    if ( !(colptr_blk = intMalloc_dist( (((size_t) n_loc)+1)*procs)) )
	   ABORT("Malloc fails for colptr_blk[]");
    for (i = 0, j = 0; i < procs; ++i) {
        for (k = j; k < j + n_locs[i]; ++k) colptr_send[i+k] = colptr_loc[k];
	colptr_send[i+k] = colptr_loc[k]; /* Add an END marker */
	sendcnts[i] = n_locs[i] + 1;
#if ( DEBUGlevel>=1 )
	assert(j == fst_rows[i]);
#endif
	sdispls[i] = j + i;
	recvcnts[i] = n_loc + 1;
	rdispls[i] = i * (n_loc + 1);
	j += n_locs[i]; /* First column of next block in colptr_loc[] */
    }
    MPI_Alltoallv(colptr_send, sendcnts, sdispls, mpi_int_t,
		  colptr_blk, recvcnts, rdispls, mpi_int_t, grid->comm);

    /* Adjust colptr_blk[] so that they contain the local indices of the
       column pointers in the receive buffer. */
    nnz = 0; /* The running sum of the nonzeros counted by far */
    k = 0;
    for (i = 0; i < procs; ++i) {
	for (j = rdispls[i]; j < rdispls[i] + n_loc; ++j) {
	    colnnz = colptr_blk[j+1] - colptr_blk[j];
	    /*assert(k<=j);*/
	    colptr_blk[k] = nnz;
	    nnz += colnnz; /* Start of the next column */
	    ++k;
	}
	colptr_blk[k++] = nnz; /* Add an END marker for each block */
    }
    /*assert(k == (n_loc+1)*procs);*/

    /* Now prepare to transfer row indices and values. */
    sdispls[0] = 0;
    for (i = 0; i < procs-1; ++i) {
        sendcnts[i] = colptr_loc[fst_rows[i+1]] - colptr_loc[fst_rows[i]];
	sdispls[i+1] = sdispls[i] + sendcnts[i];
    }
    sendcnts[procs-1] = colptr_loc[n] - colptr_loc[fst_rows[procs-1]];
    for (i = 0; i < procs; ++i) {
        j = rdispls[i]; /* Point to this block in colptr_blk[]. */
	recvcnts[i] = colptr_blk[j+n_loc] - colptr_blk[j];
    }
    rdispls[0] = 0; /* Recompute rdispls[] for row indices. */
    for (i = 0; i < procs-1; ++i) rdispls[i+1] = rdispls[i] + recvcnts[i];

    k = rdispls[procs-1] + recvcnts[procs-1]; /* Total received */
    if ( !(rowind_recv = (int_t *) intMalloc_dist(2*k)) )
        ABORT("Malloc fails for rowind_recv[]");
    rowind_buf = rowind_recv + k;
    MPI_Alltoallv(rowind_loc, sendcnts, sdispls, mpi_int_t,
		  rowind_recv, recvcnts, rdispls, mpi_int_t, grid->comm);
    if ( need_value ) {
        if ( !(a_recv = (doublecomplex *) doublecomplexMalloc_dist(2*k)) )
	    ABORT("Malloc fails for rowind_recv[]");
	a_buf = a_recv + k;
	MPI_Alltoallv(a_loc, sendcnts, sdispls, SuperLU_MPI_DOUBLE_COMPLEX,
                      a_recv, recvcnts, rdispls, SuperLU_MPI_DOUBLE_COMPLEX,
                      grid->comm);
    }

    /* Reset colptr_loc[] to point to the n_loc global columns. */
    colptr_loc[0] = 0;
    itemp = colptr_send;
    for (j = 0; j < n_loc; ++j) {
        colnnz = 0;
	for (i = 0; i < procs; ++i) {
	    k = i * (n_loc + 1) + j; /* j-th column in i-th block */
	    colnnz += colptr_blk[k+1] - colptr_blk[k];
	}
	colptr_loc[j+1] = colptr_loc[j] + colnnz;
	itemp[j] = colptr_loc[j]; /* Save a copy of the column starts */
    }
    itemp[n_loc] = colptr_loc[n_loc];

    /* Merge blocks of row indices into columns of row indices. */
    for (i = 0; i < procs; ++i) {
        k = i * (n_loc + 1);
	for (j = 0; j < n_loc; ++j) { /* i-th block */
	    for (l = colptr_blk[k+j]; l < colptr_blk[k+j+1]; ++l) {
	        rowind_buf[itemp[j]] = rowind_recv[l];
		++itemp[j];
	    }
	}
    }

    if ( need_value ) {
        for (j = 0; j < n_loc+1; ++j) itemp[j] = colptr_loc[j];
        for (i = 0; i < procs; ++i) {
	    k = i * (n_loc + 1);
	    for (j = 0; j < n_loc; ++j) { /* i-th block */
	        for (l = colptr_blk[k+j]; l < colptr_blk[k+j+1]; ++l) {
		    a_buf[itemp[j]] = a_recv[l];
		    ++itemp[j];
		}
	    }
	}
    }

    /* ------------------------------------------------------------
       SECOND PHASE: GATHER TO GLOBAL A IN COMPRESSED COLUMN FORMAT.
       ------------------------------------------------------------*/
    GA->nrow  = A->nrow;
    GA->ncol  = A->ncol;
    GA->Stype = SLU_NC;
    GA->Dtype = A->Dtype;
    GA->Mtype = A->Mtype;
    GAstore = GA->Store = (NCformat *) SUPERLU_MALLOC ( sizeof(NCformat) );
    if ( !GAstore ) ABORT ("SUPERLU_MALLOC fails for GAstore");

    /* First gather the size of each piece. */
    nnz_loc = colptr_loc[n_loc];
    MPI_Allgather(&nnz_loc, 1, mpi_int_t, itemp, 1, mpi_int_t, grid->comm);
    for (i = 0, nnz = 0; i < procs; ++i) nnz += itemp[i];
    GAstore->nnz = nnz;

    if ( !(GAstore->rowind = (int_t *) intMalloc_dist (nnz)) )
        ABORT ("SUPERLU_MALLOC fails for GAstore->rowind[]");
    if ( !(GAstore->colptr = (int_t *) intMalloc_dist (n+1)) )
        ABORT ("SUPERLU_MALLOC fails for GAstore->colptr[]");

    /* Allgatherv for row indices. */
    rdispls[0] = 0;
    for (i = 0; i < procs-1; ++i) {
        rdispls[i+1] = rdispls[i] + itemp[i];
        itemp_32[i] = itemp[i];
    }
    itemp_32[procs-1] = itemp[procs-1];
    it = nnz_loc;
    MPI_Allgatherv(rowind_buf, it, mpi_int_t, GAstore->rowind,
		   itemp_32, rdispls, mpi_int_t, grid->comm);
    if ( need_value ) {
      if ( !(GAstore->nzval = (doublecomplex *) doublecomplexMalloc_dist (nnz)) )
          ABORT ("SUPERLU_MALLOC fails for GAstore->rnzval[]");
      MPI_Allgatherv(a_buf, it, SuperLU_MPI_DOUBLE_COMPLEX, GAstore->nzval,
		     itemp_32, rdispls, SuperLU_MPI_DOUBLE_COMPLEX, grid->comm);
    } else GAstore->nzval = NULL;

    /* Now gather the column pointers. */
    rdispls[0] = 0;
    for (i = 0; i < procs-1; ++i) {
        rdispls[i+1] = rdispls[i] + n_locs[i];
        itemp_32[i] = n_locs[i];
    }
    itemp_32[procs-1] = n_locs[procs-1];
    MPI_Allgatherv(colptr_loc, n_loc, mpi_int_t, GAstore->colptr,
		   itemp_32, rdispls, mpi_int_t, grid->comm);

    /* Recompute column pointers. */
    for (i = 1; i < procs; ++i) {
        k = rdispls[i];
	for (j = 0; j < n_locs[i]; ++j) GAstore->colptr[k++] += itemp[i-1];
	itemp[i] += itemp[i-1]; /* prefix sum */
    }
    GAstore->colptr[n] = nnz;

#if ( DEBUGlevel>=2 )
    if ( !grid->iam ) {
        printf("After pdCompRow_loc_to_CompCol_global()\n");
	zPrint_CompCol_Matrix_dist(GA);
    }
#endif

    SUPERLU_FREE(a_loc);
    SUPERLU_FREE(rowind_loc);
    SUPERLU_FREE(colptr_loc);
    SUPERLU_FREE(fst_rows);
    SUPERLU_FREE(recvcnts);
    SUPERLU_FREE(colptr_send);
    SUPERLU_FREE(colptr_blk);
    SUPERLU_FREE(rowind_recv);
    if ( need_value) SUPERLU_FREE(a_recv);
#if ( DEBUGlevel>=1 )
    if ( !grid->iam ) printf("sizeof(NCformat) %lu\n", sizeof(NCformat));
    CHECK_MALLOC(grid->iam, "Exit pzCompRow_loc_to_CompCol_global");
#endif
    return 0;
} /* pzCompRow_loc_to_CompCol_global */


/*! \brief Permute the distributed dense matrix: B <= perm(X). perm[i] = j means the i-th row of X is in the j-th row of B.
 */
int pzPermute_Dense_Matrix
(
 int_t fst_row,
 int_t m_loc,
 int_t row_to_proc[],
 int_t perm[],
 doublecomplex X[], int ldx,
 doublecomplex B[], int ldb,
 int nrhs,
 gridinfo_t *grid
)
{
    int_t i, j, k, l;
    int p, procs;
    int *sendcnts, *sendcnts_nrhs, *recvcnts, *recvcnts_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *send_ibuf, *recv_ibuf;
    doublecomplex *send_dbuf, *recv_dbuf;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Enter pzPermute_Dense_Matrix()");
#endif

    procs = grid->nprow * grid->npcol;
    if ( !(sendcnts = SUPERLU_MALLOC(10*procs * sizeof(int))) )
        ABORT("Malloc fails for sendcnts[].");
    sendcnts_nrhs = sendcnts + procs;
    recvcnts = sendcnts_nrhs + procs;
    recvcnts_nrhs = recvcnts + procs;
    sdispls = recvcnts_nrhs + procs;
    sdispls_nrhs = sdispls + procs;
    rdispls = sdispls_nrhs + procs;
    rdispls_nrhs = rdispls + procs;
    ptr_to_ibuf = rdispls_nrhs + procs;
    ptr_to_dbuf = ptr_to_ibuf + procs;

    for (i = 0; i < procs; ++i) sendcnts[i] = 0;

    /* Count the number of X entries to be sent to each process.*/
    for (i = fst_row; i < fst_row + m_loc; ++i) {
        p = row_to_proc[perm[i]];
	++sendcnts[p];
    }
    MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, grid->comm);
    sdispls[0] = rdispls[0] = 0;
    sdispls_nrhs[0] = rdispls_nrhs[0] = 0;
    sendcnts_nrhs[0] = sendcnts[0] * nrhs;
    recvcnts_nrhs[0] = recvcnts[0] * nrhs;
    for (i = 1; i < procs; ++i) {
        sdispls[i] = sdispls[i-1] + sendcnts[i-1];
	sdispls_nrhs[i] = sdispls[i] * nrhs;
	rdispls[i] = rdispls[i-1] + recvcnts[i-1];
	rdispls_nrhs[i] = rdispls[i] * nrhs;
	sendcnts_nrhs[i] = sendcnts[i] * nrhs;
	recvcnts_nrhs[i] = recvcnts[i] * nrhs;
    }
    k = sdispls[procs-1] + sendcnts[procs-1];/* Total number of sends */
    l = rdispls[procs-1] + recvcnts[procs-1];/* Total number of recvs */
    /*assert(k == m_loc);*/
    /*assert(l == m_loc);*/
    if ( !(send_ibuf = intMalloc_dist(k + l)) )
        ABORT("Malloc fails for send_ibuf[].");
    recv_ibuf = send_ibuf + k;
    if ( !(send_dbuf = doublecomplexMalloc_dist((k + l)*nrhs)) )
        ABORT("Malloc fails for send_dbuf[].");
    recv_dbuf = send_dbuf + k * nrhs;

    for (i = 0; i < procs; ++i) {
        ptr_to_ibuf[i] = sdispls[i];
	ptr_to_dbuf[i] = sdispls_nrhs[i];
    }

    /* Fill in the send buffers: send_ibuf[] and send_dbuf[]. */
    for (i = fst_row; i < fst_row + m_loc; ++i) {
        j = perm[i];
	p = row_to_proc[j];
	send_ibuf[ptr_to_ibuf[p]] = j;
	j = ptr_to_dbuf[p];
	RHS_ITERATE(k) { /* RHS stored in row major in the buffer */
	    send_dbuf[j++] = X[i-fst_row + k*ldx];
	}
	++ptr_to_ibuf[p];
	ptr_to_dbuf[p] += nrhs;
    }

    /* Transfer the (permuted) row indices and numerical values. */
    MPI_Alltoallv(send_ibuf, sendcnts, sdispls, mpi_int_t,
		  recv_ibuf, recvcnts, rdispls, mpi_int_t, grid->comm);
    MPI_Alltoallv(send_dbuf, sendcnts_nrhs, sdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
		  recv_dbuf, recvcnts_nrhs, rdispls_nrhs, SuperLU_MPI_DOUBLE_COMPLEX,
		  grid->comm);

    /* Copy the buffer into b. */
    for (i = 0, l = 0; i < m_loc; ++i) {
        j = recv_ibuf[i] - fst_row; /* Relative row number */
	RHS_ITERATE(k) { /* RHS stored in row major in the buffer */
	    B[j + k*ldb] = recv_dbuf[l++];
	}
    }

    SUPERLU_FREE(sendcnts);
    SUPERLU_FREE(send_ibuf);
    SUPERLU_FREE(send_dbuf);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid->iam, "Exit pzPermute_Dense_Matrix()");
#endif
    return 0;
} /* pzPermute_Dense_Matrix */


/*! \brief Allocate storage in LUstruct */
void zLUstructInit(const int_t n, zLUstruct_t *LUstruct)
{
    if ( !(LUstruct->etree = intMalloc_dist(n)) )
	ABORT("Malloc fails for etree[].");
    if ( !(LUstruct->Glu_persist = (Glu_persist_t *)
	   SUPERLU_MALLOC(sizeof(Glu_persist_t))) )
	ABORT("Malloc fails for Glu_persist_t.");
    if ( !(LUstruct->Llu = (zLocalLU_t *)
	   SUPERLU_MALLOC(sizeof(zLocalLU_t))) )
	ABORT("Malloc fails for LocalLU_t.");
	LUstruct->Llu->inv = 0;
}

/*! \brief Deallocate LUstruct */
void zLUstructFree(zLUstruct_t *LUstruct)
{
#if ( DEBUGlevel>=1 )
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    CHECK_MALLOC(iam, "Enter zLUstructFree()");
#endif

    SUPERLU_FREE(LUstruct->etree);
    SUPERLU_FREE(LUstruct->Glu_persist);
    SUPERLU_FREE(LUstruct->Llu);
    zDestroy_trf3Dpartition(LUstruct->trf3Dpart);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit zLUstructFree()");
#endif
}

/*! \brief Destroy distributed L & U matrices. */
void
zDestroy_LU(int_t n, gridinfo_t *grid, zLUstruct_t *LUstruct)
{
    int_t i, nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;

#if ( DEBUGlevel>=1 )
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    CHECK_MALLOC(iam, "Enter zDestroy_LU()");
#endif

    zDestroy_Tree(n, grid, LUstruct);

    nsupers = Glu_persist->supno[n-1] + 1;

    /* Following are free'd in distribution routines */
    // nb = CEILING(nsupers, grid->npcol);
    // for (i = 0; i < nb; ++i)
    //	if ( Llu->Lrowind_bc_ptr[i] ) {
    //	    SUPERLU_FREE (Llu->Lrowind_bc_ptr[i]);
#if 0 // Sherry: the following is not allocated with cudaHostAlloc
    //#ifdef GPU_ACC
	    checkGPU(gpuFreeHost(Llu->Lnzval_bc_ptr[i]));
#endif
    //	    SUPERLU_FREE (Llu->Lnzval_bc_ptr[i]);
    //	}

    SUPERLU_FREE (Llu->Lrowind_bc_ptr);
    SUPERLU_FREE (Llu->Lrowind_bc_dat);
    SUPERLU_FREE (Llu->Lrowind_bc_offset);
    SUPERLU_FREE (Llu->Lnzval_bc_ptr);
    SUPERLU_FREE (Llu->Lnzval_bc_dat);
    SUPERLU_FREE (Llu->Lnzval_bc_offset);

    /* Following are free'd in distribution routines */
    nb = CEILING(nsupers, grid->nprow);
    for (i = 0; i < nb; ++i)
    	if ( Llu->Ufstnz_br_ptr[i] ) {
    	    SUPERLU_FREE (Llu->Ufstnz_br_ptr[i]);
    	    SUPERLU_FREE (Llu->Unzval_br_ptr[i]);
    	}
    SUPERLU_FREE (Llu->Ufstnz_br_ptr);
    // SUPERLU_FREE (Llu->Ufstnz_br_dat);
    // SUPERLU_FREE (Llu->Ufstnz_br_offset);
    SUPERLU_FREE (Llu->Unzval_br_ptr);
    // SUPERLU_FREE (Llu->Unzval_br_dat);
    // SUPERLU_FREE (Llu->Unzval_br_offset);

    /* The following can be freed after factorization. */
    SUPERLU_FREE(Llu->ToRecv);
    SUPERLU_FREE(Llu->ToSendD);
    SUPERLU_FREE(Llu->ToSendR[0]);
    SUPERLU_FREE(Llu->ToSendR);

    /* The following can be freed only after iterative refinement. */
    SUPERLU_FREE(Llu->ilsum);
    SUPERLU_FREE(Llu->fmod);
    SUPERLU_FREE((Llu->fsendx_plist)[0]);
    SUPERLU_FREE(Llu->fsendx_plist);
    SUPERLU_FREE(Llu->bmod);
    SUPERLU_FREE((Llu->bsendx_plist)[0]);
    SUPERLU_FREE(Llu->bsendx_plist);
    SUPERLU_FREE(Llu->mod_bit);

    /* Following are free'd in distribution routines */
    // nb = CEILING(nsupers, grid->npcol);
    // for (i = 0; i < nb; ++i)
    //	if ( Llu->Lindval_loc_bc_ptr[i]!=NULL) {
    //	    SUPERLU_FREE (Llu->Lindval_loc_bc_ptr[i]);
    //	}
    SUPERLU_FREE(Llu->Lindval_loc_bc_ptr);
    SUPERLU_FREE(Llu->Lindval_loc_bc_dat);
    SUPERLU_FREE(Llu->Lindval_loc_bc_offset);

    /* Following are free'd in distribution routines */
    // nb = CEILING(nsupers, grid->npcol);
    // for (i=0; i<nb; ++i) {
    //	if(Llu->Linv_bc_ptr[i]!=NULL) {
    //	    SUPERLU_FREE(Llu->Linv_bc_ptr[i]);
    //	}
    //	if(Llu->Uinv_bc_ptr[i]!=NULL){
    //	    SUPERLU_FREE(Llu->Uinv_bc_ptr[i]);
    //	}
    // }
    SUPERLU_FREE(Llu->Linv_bc_ptr);
    SUPERLU_FREE(Llu->Linv_bc_dat);
    SUPERLU_FREE(Llu->Linv_bc_offset);
    SUPERLU_FREE(Llu->Uinv_bc_ptr);
    SUPERLU_FREE(Llu->Uinv_bc_dat);
    SUPERLU_FREE(Llu->Uinv_bc_offset);
    SUPERLU_FREE(Llu->Unnz);

    /* Following are free'd in distribution routines */
    nb = CEILING(nsupers, grid->npcol);
    for (i = 0; i < nb; ++i)
    	if ( Llu->Urbs[i] ) {
    	    SUPERLU_FREE(Llu->Ucb_indptr[i]);
    	    SUPERLU_FREE(Llu->Ucb_valptr[i]);
    }
    SUPERLU_FREE(Llu->Ucb_indptr);
    // SUPERLU_FREE(Llu->Ucb_inddat);
    // SUPERLU_FREE(Llu->Ucb_indoffset);
    SUPERLU_FREE(Llu->Ucb_valptr);
    // SUPERLU_FREE(Llu->Ucb_valdat);
    // SUPERLU_FREE(Llu->Ucb_valoffset);
    SUPERLU_FREE(Llu->Urbs);

    SUPERLU_FREE(Glu_persist->xsup);
    SUPERLU_FREE(Glu_persist->supno);
    SUPERLU_FREE(Llu->bcols_masked);

#ifdef GPU_ACC
if (get_acc_solve()){
    checkGPU (gpuFree (Llu->d_xsup));
    checkGPU (gpuFree (Llu->d_bcols_masked));
    checkGPU (gpuFree (Llu->d_LRtree_ptr));
    checkGPU (gpuFree (Llu->d_LBtree_ptr));
    checkGPU (gpuFree (Llu->d_URtree_ptr));
    checkGPU (gpuFree (Llu->d_UBtree_ptr));
    checkGPU (gpuFree (Llu->d_ilsum));
    checkGPU (gpuFree (Llu->d_Lrowind_bc_dat));
    checkGPU (gpuFree (Llu->d_Lrowind_bc_offset));
    checkGPU (gpuFree (Llu->d_Lnzval_bc_dat));
    checkGPU (gpuFree (Llu->d_Lnzval_bc_offset));
    checkGPU (gpuFree (Llu->d_Linv_bc_dat));
    checkGPU (gpuFree (Llu->d_Uinv_bc_dat));
    checkGPU (gpuFree (Llu->d_Linv_bc_offset));
    checkGPU (gpuFree (Llu->d_Uinv_bc_offset));
    checkGPU (gpuFree (Llu->d_Lindval_loc_bc_dat));
    checkGPU (gpuFree (Llu->d_Lindval_loc_bc_offset));

    checkGPU (gpuFree (Llu->d_Ucolind_bc_dat));
    checkGPU (gpuFree (Llu->d_Ucolind_bc_offset));
    checkGPU (gpuFree (Llu->d_Uind_br_dat));
    checkGPU (gpuFree (Llu->d_Uind_br_offset));
    checkGPU (gpuFree (Llu->d_Unzval_bc_dat));
    checkGPU (gpuFree (Llu->d_Unzval_bc_offset));
    checkGPU (gpuFree (Llu->d_Uindval_loc_bc_dat));
    checkGPU (gpuFree (Llu->d_Uindval_loc_bc_offset));
#ifdef U_BLOCK_PER_ROW_ROWDATA
    checkGPU (gpuFree (Llu->d_Ucolind_br_dat));
    checkGPU (gpuFree (Llu->d_Ucolind_br_offset));
    checkGPU (gpuFree (Llu->d_Unzval_br_new_dat));
    checkGPU (gpuFree (Llu->d_Unzval_br_new_offset));
#endif
}


    #ifdef HAVE_NVSHMEM
    /* nvshmem related*/
    if (get_acc_solve()){
        zdelete_multiGPU_buffers();
    }

    SUPERLU_FREE(mystatus);
    SUPERLU_FREE(h_nfrecv);
    SUPERLU_FREE(h_nfrecvmod);
    SUPERLU_FREE(mystatusmod);
    SUPERLU_FREE(mystatus_u);
    SUPERLU_FREE(h_nfrecv_u);
    SUPERLU_FREE(mystatusmod_u);

    checkGPU (gpuFree (d_recv_cnt));
    checkGPU (gpuFree (d_recv_cnt_u));
    #endif

#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit zDestroy_LU()");
#endif
}

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Set up the communication pattern for redistribution between B and X
 *   in the triangular solution.
 *
 * Arguments
 * =========
 *
 * n      (input) int (global)
 *        The dimension of the linear system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of the distributed input matrix.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * fst_row (input) int (global)
 *        The row number of matrix B's first row in the global matrix.
 *
 * perm_r (input) int* (global)
 *        The row permutation vector.
 *
 * perm_c (input) int* (global)
 *        The column permutation vector.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 * </pre>
 */
int_t
pzgstrs_init(int_t n, int_t m_loc, int_t nrhs, int_t fst_row,
	     int_t perm_r[], int_t perm_c[], gridinfo_t *grid,
	     Glu_persist_t *Glu_persist, zSOLVEstruct_t *SOLVEstruct)
{

    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *itemp, *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *row_to_proc;
    int_t i, gbi, k, l, num_diag_procs, *diag_procs;
    int_t irow, q, knsupc, nsupers, *xsup, *supno;
    int   iam, p, pkk, procs;
    pxgstrs_comm_t *gstrs_comm;

    procs = grid->nprow * grid->npcol;
    iam = grid->iam;
    gstrs_comm = SOLVEstruct->gstrs_comm;
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n-1] + 1;
    row_to_proc = SOLVEstruct->row_to_proc;

    /* ------------------------------------------------------------
       SET UP COMMUNICATION PATTERN FOR ReDistribute_B_to_X.
       ------------------------------------------------------------*/
    if ( !(itemp = SUPERLU_MALLOC(8*procs * sizeof(int))) )
        ABORT("Malloc fails for B_to_X_itemp[].");
    SendCnt      = itemp;
    SendCnt_nrhs = itemp +   procs;
    RecvCnt      = itemp + 2*procs;
    RecvCnt_nrhs = itemp + 3*procs;
    sdispls      = itemp + 4*procs;
    sdispls_nrhs = itemp + 5*procs;
    rdispls      = itemp + 6*procs;
    rdispls_nrhs = itemp + 7*procs;

    /* Count the number of elements to be sent to each diagonal process.*/
    for (p = 0; p < procs; ++p) SendCnt[p] = 0;
    for (i = 0, l = fst_row; i < m_loc; ++i, ++l) {
        irow = perm_c[perm_r[l]]; /* Row number in Pc*Pr*B */
	gbi = BlockNum( irow );
	p = PNUM( PROW(gbi,grid), PCOL(gbi,grid), grid ); /* Diagonal process */
	++SendCnt[p];
    }

    /* Set up the displacements for alltoall. */
    MPI_Alltoall(SendCnt, 1, MPI_INT, RecvCnt, 1, MPI_INT, grid->comm);
    sdispls[0] = rdispls[0] = 0;
    for (p = 1; p < procs; ++p) {
        sdispls[p] = sdispls[p-1] + SendCnt[p-1];
        rdispls[p] = rdispls[p-1] + RecvCnt[p-1];
    }
    for (p = 0; p < procs; ++p) {
        SendCnt_nrhs[p] = SendCnt[p] * nrhs;
	sdispls_nrhs[p] = sdispls[p] * nrhs;
        RecvCnt_nrhs[p] = RecvCnt[p] * nrhs;
	rdispls_nrhs[p] = rdispls[p] * nrhs;
    }

    /* This is saved for repeated solves, and is freed in pxgstrs_finalize().*/
    gstrs_comm->B_to_X_SendCnt = SendCnt;

    /* ------------------------------------------------------------
       SET UP COMMUNICATION PATTERN FOR ReDistribute_X_to_B.
       ------------------------------------------------------------*/
    /* This is freed in pxgstrs_finalize(). */
    if ( !(itemp = SUPERLU_MALLOC(8*procs * sizeof(int))) )
        ABORT("Malloc fails for X_to_B_itemp[].");
    SendCnt      = itemp;
    SendCnt_nrhs = itemp +   procs;
    RecvCnt      = itemp + 2*procs;
    RecvCnt_nrhs = itemp + 3*procs;
    sdispls      = itemp + 4*procs;
    sdispls_nrhs = itemp + 5*procs;
    rdispls      = itemp + 6*procs;
    rdispls_nrhs = itemp + 7*procs;

    /* Count the number of X entries to be sent to each process.*/
    for (p = 0; p < procs; ++p) SendCnt[p] = 0;
    num_diag_procs = SOLVEstruct->num_diag_procs;
    diag_procs = SOLVEstruct->diag_procs;

    for (p = 0; p < num_diag_procs; ++p) { /* for all diagonal processes */
	pkk = diag_procs[p];
	if ( iam == pkk ) {
	    for (k = p; k < nsupers; k += num_diag_procs) {
		knsupc = SuperSize( k );
		irow = FstBlockC( k );
		for (i = 0; i < knsupc; ++i) {
#if 0
		    q = row_to_proc[inv_perm_c[irow]];
#else
		    q = row_to_proc[irow];
#endif
		    ++SendCnt[q];
		    ++irow;
		}
	    }
	}
    }

    MPI_Alltoall(SendCnt, 1, MPI_INT, RecvCnt, 1, MPI_INT, grid->comm);
    sdispls[0] = rdispls[0] = 0;
    sdispls_nrhs[0] = rdispls_nrhs[0] = 0;
    SendCnt_nrhs[0] = SendCnt[0] * nrhs;
    RecvCnt_nrhs[0] = RecvCnt[0] * nrhs;
    for (p = 1; p < procs; ++p) {
        sdispls[p] = sdispls[p-1] + SendCnt[p-1];
        rdispls[p] = rdispls[p-1] + RecvCnt[p-1];
        sdispls_nrhs[p] = sdispls[p] * nrhs;
        rdispls_nrhs[p] = rdispls[p] * nrhs;
	SendCnt_nrhs[p] = SendCnt[p] * nrhs;
	RecvCnt_nrhs[p] = RecvCnt[p] * nrhs;
    }

    /* This is saved for repeated solves, and is freed in pxgstrs_finalize().*/
    gstrs_comm->X_to_B_SendCnt = SendCnt;

    if ( !(ptr_to_ibuf = SUPERLU_MALLOC(2*procs * sizeof(int))) )
        ABORT("Malloc fails for ptr_to_ibuf[].");
    gstrs_comm->ptr_to_ibuf = ptr_to_ibuf;
    gstrs_comm->ptr_to_dbuf = ptr_to_ibuf + procs;

    return 0;
} /* PZGSTRS_INIT */



int_t
pzgstrs_init_device_lsum_x(superlu_dist_options_t *options, int_t n, int_t m_loc, int_t nrhs, gridinfo_t *grid,
	     zLUstruct_t *LUstruct, zSOLVEstruct_t *SOLVEstruct, int* supernodeMask)
{
#if ( defined(GPU_ACC) )
    doublecomplex zero = {0.0, 0.0};
    int  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int  nbrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t i, gbi, k, l, gb;
    int_t irow, q, knsupc, nsupers, *xsup, *supno;
    int   iam, p, pkk, procs;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t ldalsum = LUstruct->Llu->ldalsum;
    int_t* ilsum = LUstruct->Llu->ilsum;
    int *frecv, *brecv;
    int_t  *Urbs = LUstruct->Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = LUstruct->Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */

    procs = grid->nprow * grid->npcol;
    iam = grid->iam;
    int_t myrow = MYROW (iam, grid);

    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n-1] + 1;

    /* Allocate working storage. */
    int_t nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */
    int_t nc = CEILING (nsupers, Pc);    /* Number of local block cols. */
    int_t sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    checkGPU(gpuMalloc( (void**)&(SOLVEstruct->d_lsum), sizelsum * sizeof(doublecomplex)));
    checkGPU(gpuMalloc( (void**)&(SOLVEstruct->d_x), (ldalsum * nrhs + nlb * XK_H) * sizeof(doublecomplex)));
    checkGPU(gpuMemset( SOLVEstruct->d_lsum, 0, sizelsum * sizeof(doublecomplex)));
    checkGPU(gpuMemset( SOLVEstruct->d_x, 0, (ldalsum * nrhs + nlb * XK_H) * sizeof(doublecomplex)));

    doublecomplex* lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum * sizeof(doublecomplex));
    for (int_t ii=0; ii < sizelsum; ii++ )
	lsum[ii]=zero;
    int_t ii = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t knsupc = SuperSize (k);
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* Local block number. */
            int_t il = LSUM_BLK (lk);
            lsum[il - LSUM_H].r = k;/* Block number prepended in the header. */
            lsum[il - LSUM_H].i = 0;
        }
        ii += knsupc;
    }
    checkGPU(gpuMalloc( (void**)&(SOLVEstruct->d_lsum_save), sizelsum * sizeof(doublecomplex)));
    checkGPU(gpuMemcpy(SOLVEstruct->d_lsum_save, lsum, sizelsum * sizeof(doublecomplex), gpuMemcpyHostToDevice));
    SUPERLU_FREE(lsum);

    int* fmod = getfmod_newsolve(nlb, nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
    int  nfrecvx = getNfrecvx_newsolve(nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
    if ( !(frecv = int32Calloc_dist(nlb)) )
	ABORT("Calloc fails for frecv[].");
    C_Tree  *LRtree_ptr = LUstruct->Llu->LRtree_ptr;
	nfrecvmod=0;
    for (int_t lk=0;lk<nlb;++lk){
		if(LRtree_ptr[lk].empty_==NO){
            gb = myrow+lk*grid->nprow;  /* not sure */
            if (supernodeMask[gb]==1){
                frecv[lk] = LRtree_ptr[lk].destCnt_;
                nfrecvmod += frecv[lk];
            }
		}
	}
	for (i = 0; i < nlb; ++i) fmod[i] += frecv[i];

    checkGPU(gpuMalloc( (void**)&SOLVEstruct->d_fmod, nlb * sizeof(int)));
    checkGPU(gpuMemset( SOLVEstruct->d_fmod, 0, nlb * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&SOLVEstruct->d_fmod_save, nlb * sizeof(int)));
	checkGPU(gpuMemcpy(SOLVEstruct->d_fmod_save, fmod, nlb * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(fmod);
    SUPERLU_FREE(frecv);



    int* bmod=  getBmod3d_newsolve(nlb, nsupers, supernodeMask, xsup, LUstruct->Llu->Ufstnz_br_ptr, grid);
    int nbrecvx= getNbrecvX_newsolve(nsupers, supernodeMask, Urbs, Ucb_indptr, grid);
    if ( !(brecv = int32Calloc_dist(nlb)) )
        ABORT("Calloc fails for brecv[].");
    C_Tree  *URtree_ptr = LUstruct->Llu->URtree_ptr;
    nbrecvmod=0;
    for (int_t lk=0;lk<nlb;++lk){
		if(URtree_ptr[lk].empty_==NO){
			brecv[lk] = URtree_ptr[lk].destCnt_;
            nbrecvmod+=brecv[lk];
		}
	}
	for (i = 0; i < nlb; ++i) bmod[i] += brecv[i];

    checkGPU(gpuMalloc( (void**)&SOLVEstruct->d_bmod, nlb * sizeof(int)));
    checkGPU(gpuMemset( SOLVEstruct->d_bmod, 0, nlb * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&SOLVEstruct->d_bmod_save, nlb * sizeof(int)));
	checkGPU(gpuMemcpy(SOLVEstruct->d_bmod_save, bmod, nlb * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(bmod);
    SUPERLU_FREE(brecv);

	// /* Compute ldaspa and ilsum[]. */
	// ldaspa = 0;
	// ilsum[0] = 0;
	// for (gb = 0; gb < nsupers; ++gb) {
	//     if ( myrow == PROW( gb, grid ) ) {
	// 	i = SuperSize( gb );
	// 	ldaspa += i;
	// 	lb = LBi( gb, grid );
	// 	ilsum[lb + 1] = ilsum[lb] + i;
	//     }
	// }

    /* nvshmem related. */
    #ifdef HAVE_NVSHMEM
    /////* for L solve *////
    int *my_colnum;
    if ( !(my_colnum = (int*)SUPERLU_MALLOC((nfrecvx+1) * sizeof(int))) )
        ABORT("Malloc fails for my_colnum[].");
    checkGPU(gpuMalloc( (void**)&d_colnum,  (nfrecvx+1) * sizeof(int)));

    //printf("(%d),CEILING( nsupers, grid->npcol)=%d\n",iam,CEILING( nsupers, grid->npcol));
    //fflush(stdout);
    int tmp_idx=0;
    for(int i=0; i<CEILING( nsupers, grid->npcol);i++){
        if(mystatus[i]==0) {
            my_colnum[tmp_idx]=i;
            //printf("(%d),nfrecvx=%d,i=%d,my_column[%d]=%d\n",iam,nfrecvx,i,tmp_idx,my_colnum[tmp_idx]);
            //fflush(stdout);
            tmp_idx += 1;
        }
    }
    checkGPU(gpuMemcpy(d_colnum, my_colnum,  (nfrecvx+1) * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(my_colnum);
    //printf("(%d) nfrecvx=%d,nfrecvmod=%d,maxrecvsz=%d\n",iam,nfrecvx,nfrecvmod,maxrecvsz);
    //fflush(stdout);

    h_nfrecv[0]=nfrecvx;
#ifdef _USE_SUMMIT
    h_nfrecv[1]=32;
    h_nfrecv[2]=8;
#else
    //printf("I'm here------- %d\n",iam);
    //fflush(stdout);
    h_nfrecv[1]=1024;
    h_nfrecv[2]=2;
#endif

    checkGPU(gpuMalloc( (void**)&d_mynum, h_nfrecv[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymaskstart, h_nfrecv[1] * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymasklength, h_nfrecv[1]  * sizeof(int)));

    //printf("(%d), wait=%d,%d\n",iam,h_nfrecv[2],h_nfrecv[1]);
    //fflush(stdout);


    checkGPU(gpuMalloc( (void**)&d_status,  CEILING( nsupers, grid->npcol) * sizeof(int)));
    checkGPU(gpuMemcpy(d_status, mystatus, CEILING( nsupers, grid->npcol) * sizeof(int), gpuMemcpyHostToDevice));

    checkGPU(gpuMalloc( (void**)&d_nfrecv,  3 * sizeof(int)));
    checkGPU(gpuMemcpy(d_nfrecv, h_nfrecv, 3 * sizeof(int), gpuMemcpyHostToDevice));

    int *my_colnummod;
    if ( !(my_colnummod = (int*)SUPERLU_MALLOC((nfrecvmod+1) * sizeof(int))) )
        ABORT("Malloc fails for my_colnum[].");
    checkGPU(gpuMalloc( (void**)&d_colnummod,  (nfrecvmod+1) * sizeof(int)));

    tmp_idx=0;
    for(int i=0; i<CEILING( nsupers, grid->nprow);i++){
        //printf("(%d),nfrecvmod=%d,i=%d,recv_cnt=%d\n",iam,nfrecvmod,i,h_recv_cnt[i]);
        if(mystatusmod[i*2]==0) {
            my_colnummod[tmp_idx]=i;
            //printf("(%d),nfrecvmod=%d,i=%d,my_colnummod[%d]=%d\n",iam,nfrecvmod,i,tmp_idx,my_colnummod[tmp_idx]);
            //fflush(stdout);
            tmp_idx += 1;
        }
    }
    h_nfrecvmod[0]=nfrecvmod;
    h_nfrecvmod[1]=tmp_idx;
    h_nfrecvmod[2]=h_nfrecv[2];
    checkGPU(gpuMalloc( (void**)&d_nfrecvmod, 4 * sizeof(int)));
    checkGPU(gpuMemcpy(d_nfrecvmod, h_nfrecvmod, 4 * sizeof(int), gpuMemcpyHostToDevice));
    checkGPU(gpuMalloc( (void**)&d_statusmod, 2*CEILING(nsupers, grid->nprow) * sizeof(int)));

    checkGPU(gpuMemcpy(d_colnummod, my_colnummod,  (nfrecvmod+1) * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(my_colnummod);
    checkGPU(gpuMalloc( (void**)&d_mynummod, h_nfrecv[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymaskstartmod, h_nfrecv[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymasklengthmod,   h_nfrecv[1] * sizeof(int)));

    checkGPU(gpuMalloc( (void**)&d_msgnum,  h_nfrecv[1] * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_flag_mod,  (h_nfrecvmod[3]+1) * sizeof(int)));
    int* tmp_val;
    if ( !(tmp_val = (int*)SUPERLU_MALLOC((h_nfrecvmod[3]+1) * sizeof(int))) )
        ABORT("Malloc fails for tmp_val[].");
    tmp_val[0]=0;
    for(int i=1; i<h_nfrecvmod[3]+1;i++) tmp_val[i]=-1;
    checkGPU(gpuMemcpy(d_flag_mod, tmp_val,  (h_nfrecvmod[3]+1) * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(tmp_val);
    //printf("(%d) nfrecvx=%d, nfrecvmod=%d,nfsendmod=%d\n",iam,nfrecvx, nfrecvmod,h_nfrecvmod[3]);
    //fflush(stdout);

        /////* for U solve *////
    checkGPU(gpuMalloc( (void**)&d_nfrecv_u,  3 * sizeof(int)));

    int *my_colnum_u;
    if ( !(my_colnum_u = (int*)SUPERLU_MALLOC((nbrecvx+1) * sizeof(int))) )
        ABORT("Malloc fails for my_colnum_u[].");
    checkGPU(gpuMalloc( (void**)&d_colnum_u,  (nbrecvx+1) * sizeof(int)));

    //printf("(%d),CEILING( nsupers, grid->npcol)=%d\n",iam,CEILING( nsupers, grid->npcol));
    //fflush(stdout);
    tmp_idx=0;
    for(int i=0; i<CEILING( nsupers, grid->npcol);i++){
        if(mystatus_u[i]==0) {
            my_colnum_u[tmp_idx]=i;
            //printf("(%d),nbrecvx=%d,i=%d,my_column_u[%d]=%d\n",iam,nbrecvx,i,tmp_idx,my_colnum_u[tmp_idx]);
            //fflush(stdout);
            tmp_idx += 1;
        }
    }
    h_nfrecv_u[0]=nbrecvx;
#ifdef _USE_SUMMIT
    h_nfrecv_u[1]=32;
    h_nfrecv_u[2]=8;
#else
    h_nfrecv_u[1]=1024;
    h_nfrecv_u[2]=2;
#endif
    //printf("(%d), wait=%d,%d\n",iam,h_nfrecv[2],h_nfrecv[1]);
    //fflush(stdout);

    checkGPU(gpuMemcpy(d_colnum_u, my_colnum_u,  (nbrecvx+1) * sizeof(int), gpuMemcpyHostToDevice));
    //printf("(%d) nbrecvx=%d,nbrecvmod=%d\n",iam,nbrecvx,nbrecvmod);
    //fflush(stdout);
    checkGPU(gpuMalloc( (void**)&d_mynum_u, h_nfrecv_u[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymaskstart_u, h_nfrecv_u[1] * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymasklength_u, h_nfrecv_u[1]  * sizeof(int)));
    checkGPU(gpuMemcpy(d_nfrecv_u, h_nfrecv_u, 3 * sizeof(int), gpuMemcpyHostToDevice));

    int *my_colnummod_u;
    if ( !(my_colnummod_u = (int*)SUPERLU_MALLOC((nbrecvmod+1) * sizeof(int))) )
        ABORT("Malloc fails for my_colnummod_u[].");
    checkGPU(gpuMalloc( (void**)&d_colnummod_u,  (nbrecvmod+1) * sizeof(int)));

    tmp_idx=0;
    for(int i=0; i<CEILING( nsupers, grid->nprow);i++){
        //printf("(%d),nfrecvmod=%d,i=%d,recv_cnt=%d\n",iam,nfrecvmod,i,h_recv_cnt[i]);
        if(mystatusmod_u[i*2]==0) {
            my_colnummod_u[tmp_idx]=i;
            //printf("(%d),nbrecvmod=%d,i=%d,my_colnummod_u[%d]=%d\n",iam,nbrecvmod,i,tmp_idx,my_colnummod_u[tmp_idx]);
            //fflush(stdout);
            tmp_idx += 1;
        }
    }
    h_nfrecvmod_u[0]=nbrecvmod;
    h_nfrecvmod_u[1]=tmp_idx;
    h_nfrecvmod_u[2]=h_nfrecv[2];
    //printf("(%d) nbrecvmod=%d,%d,nbrecvx=%d\n",iam,nbrecvmod, tmp_idx,nbrecvx);
    //fflush(stdout);
    checkGPU(gpuMalloc( (void**)&d_nfrecvmod_u,  4 * sizeof(int)));
    checkGPU(gpuMemcpy(d_nfrecvmod_u, h_nfrecvmod_u, 4 * sizeof(int), gpuMemcpyHostToDevice));

    checkGPU(gpuMemcpy(d_colnummod_u, my_colnummod_u,  (nbrecvmod+1) * sizeof(int), gpuMemcpyHostToDevice));
    checkGPU(gpuMalloc( (void**)&d_mynummod_u, h_nfrecv_u[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymaskstartmod_u, h_nfrecv_u[1]  * sizeof(int)));
    checkGPU(gpuMalloc( (void**)&d_mymasklengthmod_u,   h_nfrecv_u[1] * sizeof(int)));

    checkGPU(gpuMalloc( (void**)&d_flag_mod_u,  (h_nfrecvmod_u[3]+1) * sizeof(int)));
    int* tmp_val_u;
    if ( !(tmp_val_u = (int*)SUPERLU_MALLOC((h_nfrecvmod_u[3]+1) * sizeof(int))) )
    ABORT("Malloc fails for tmp_val[].");
    tmp_val_u[0]=0;
    for(int i=1; i<h_nfrecvmod_u[3]+1;i++) tmp_val_u[i]=-1;
    checkGPU(gpuMemcpy(d_flag_mod_u, tmp_val_u,  (h_nfrecvmod_u[3]+1) * sizeof(int), gpuMemcpyHostToDevice));
    SUPERLU_FREE(tmp_val_u);
    #endif


#endif

    return 0;
} /* pzgstrs_init_device_lsum_x */


int_t
pzgstrs_delete_device_lsum_x(zSOLVEstruct_t *SOLVEstruct)
{
#if ( defined(GPU_ACC) )
    checkGPU (gpuFree (SOLVEstruct->d_x));
    checkGPU (gpuFree (SOLVEstruct->d_lsum));
    checkGPU (gpuFree (SOLVEstruct->d_lsum_save));
    checkGPU (gpuFree (SOLVEstruct->d_fmod));
    checkGPU (gpuFree (SOLVEstruct->d_fmod_save));
    checkGPU (gpuFree (SOLVEstruct->d_bmod));
    checkGPU (gpuFree (SOLVEstruct->d_bmod_save));


/* nvshmem related*/

    #ifdef HAVE_NVSHMEM
    // zdelete_multiGPU_buffers();

    checkGPU(gpuFree(d_colnum));
    checkGPU(gpuFree(d_mynum));
    checkGPU(gpuFree(d_mymaskstart));
    checkGPU(gpuFree(d_mymasklength));
    checkGPU(gpuFree(d_status));
    checkGPU(gpuFree(d_nfrecv));

    checkGPU(gpuFree(d_nfrecvmod));
    checkGPU(gpuFree(d_statusmod));
    checkGPU(gpuFree(d_mynummod));
    checkGPU(gpuFree(d_mymaskstartmod));
    checkGPU(gpuFree(d_mymasklengthmod));
    checkGPU(gpuFree(d_msgnum));
    checkGPU(gpuFree(d_flag_mod));
    #endif

#endif
    return 0;
} /* pzgstrs_delete_device_lsum_x */



/*! \brief Initialize the data structure for the solution phase.
 */
int zSolveInit(superlu_dist_options_t *options, SuperMatrix *A,
	       int_t perm_r[], int_t perm_c[], int_t nrhs,
	       zLUstruct_t *LUstruct, gridinfo_t *grid,
	       zSOLVEstruct_t *SOLVEstruct)
{
    int_t *row_to_proc, *inv_perm_c, *itemp;
    NRformat_loc *Astore;
    int_t        i, fst_row, m_loc, p;
    int          procs;

    Astore = (NRformat_loc *) A->Store;
    fst_row = Astore->fst_row;
    m_loc = Astore->m_loc;
    procs = grid->nprow * grid->npcol;

    if ( !(row_to_proc = intMalloc_dist(A->nrow)) )
	ABORT("Malloc fails for row_to_proc[]");
    SOLVEstruct->row_to_proc = row_to_proc;
    if ( !(inv_perm_c = intMalloc_dist(A->ncol)) )
        ABORT("Malloc fails for inv_perm_c[].");
    for (i = 0; i < A->ncol; ++i) inv_perm_c[perm_c[i]] = i;
    SOLVEstruct->inv_perm_c = inv_perm_c;

    /* ------------------------------------------------------------
       EVERY PROCESS NEEDS TO KNOW GLOBAL PARTITION.
       SET UP THE MAPPING BETWEEN ROWS AND PROCESSES.

       NOTE: For those processes that do not own any row, it must
             must be set so that fst_row == A->nrow.
       ------------------------------------------------------------*/
    if ( !(itemp = intMalloc_dist(procs+1)) )
        ABORT("Malloc fails for itemp[]");
    MPI_Allgather(&fst_row, 1, mpi_int_t, itemp, 1, mpi_int_t,
		  grid->comm);
    itemp[procs] = A->nrow;
    for (p = 0; p < procs; ++p) {
        for (i = itemp[p] ; i < itemp[p+1]; ++i) row_to_proc[i] = p;
    }
#if ( DEBUGlevel>=2 )
    if ( !grid->iam ) {
      printf("fst_row = %d\n", fst_row);
      PrintInt10("row_to_proc", A->nrow, row_to_proc);
      PrintInt10("inv_perm_c", A->ncol, inv_perm_c);
    }
#endif
    SUPERLU_FREE(itemp);

#if 0
    /* Compute the mapping between rows and processes. */
    /* XSL NOTE: What happens if # of mapped processes is smaller
       than total Procs?  For the processes without any row, let
       fst_row be EMPTY (-1). Make sure this case works! */
    MPI_Allgather(&fst_row, 1, mpi_int_t, itemp, 1, mpi_int_t,
		  grid->comm);
    itemp[procs] = n;
    for (p = 0; p < procs; ++p) {
        j = itemp[p];
	if ( j != SLU_EMPTY ) {
	    k = itemp[p+1];
	    if ( k == SLU_EMPTY ) k = n;
	    for (i = j ; i < k; ++i) row_to_proc[i] = p;
	}
    }
#endif

    get_diag_procs(A->ncol, LUstruct->Glu_persist, grid,
		   &SOLVEstruct->num_diag_procs,
		   &SOLVEstruct->diag_procs,
		   &SOLVEstruct->diag_len);

    /* Setup communication pattern for redistribution of B and X. */
    if ( !(SOLVEstruct->gstrs_comm = (pxgstrs_comm_t *)
	   SUPERLU_MALLOC(sizeof(pxgstrs_comm_t))) )
        ABORT("Malloc fails for gstrs_comm[]");
    pzgstrs_init(A->ncol, m_loc, nrhs, fst_row, perm_r, perm_c, grid,
		 LUstruct->Glu_persist, SOLVEstruct);

    if ( !(SOLVEstruct->gsmv_comm = (pzgsmv_comm_t *)
           SUPERLU_MALLOC(sizeof(pzgsmv_comm_t))) )
        ABORT("Malloc fails for gsmv_comm[]");
    SOLVEstruct->A_colind_gsmv = NULL;

    options->SolveInitialized = YES;
    return 0;
} /* zSolveInit */

/*! \brief Release the resources used for the solution phase.
 */
void zSolveFinalize(superlu_dist_options_t *options, zSOLVEstruct_t *SOLVEstruct)
{
    if ( options->SolveInitialized ) {
        pxgstrs_finalize(SOLVEstruct->gstrs_comm);

        if ( options->RefineInitialized ) {
            pzgsmv_finalize(SOLVEstruct->gsmv_comm);
	    options->RefineInitialized = NO;
        }
        SUPERLU_FREE(SOLVEstruct->gsmv_comm);
        SUPERLU_FREE(SOLVEstruct->row_to_proc);
        SUPERLU_FREE(SOLVEstruct->inv_perm_c);
        SUPERLU_FREE(SOLVEstruct->diag_procs);
        SUPERLU_FREE(SOLVEstruct->diag_len);
        if ( SOLVEstruct->A_colind_gsmv )
	    SUPERLU_FREE(SOLVEstruct->A_colind_gsmv);
        options->SolveInitialized = NO;
    }
} /* zSolveFinalize */

#if 0
void zDestroy_A3d_gathered_on_2d(zSOLVEstruct_t *SOLVEstruct, gridinfo3d_t *grid3d)
{
    /* free A2d and B2d, which are allocated only in 2D layer grid-0 */
    NRformat_loc3d *A3d = SOLVEstruct->A3d;
    NRformat_loc *A2d = A3d->A_nfmt;
    if (grid3d->zscp.Iam == 0) {
	SUPERLU_FREE( A2d->rowptr );
	SUPERLU_FREE( A2d->colind );
	SUPERLU_FREE( A2d->nzval );
    }
    SUPERLU_FREE(A3d->row_counts_int);  // free displacements and counts
    SUPERLU_FREE(A3d->row_disp);
    SUPERLU_FREE(A3d->nnz_counts_int);
    SUPERLU_FREE(A3d->nnz_disp);
    SUPERLU_FREE(A3d->b_counts_int);
    SUPERLU_FREE(A3d->b_disp);
    int rankorder = grid3d->rankorder;
    if ( rankorder == 0 ) { /* Z-major in 3D grid */
        SUPERLU_FREE(A3d->procs_to_send_list);
        SUPERLU_FREE(A3d->send_count_list);
        SUPERLU_FREE(A3d->procs_recv_from_list);
        SUPERLU_FREE(A3d->recv_count_list);
    }
    SUPERLU_FREE( A2d );         // free 2D structure
    SUPERLU_FREE( A3d );         // free 3D structure
} /* zDestroy_A3d_gathered_on_2d */
#else
void zDestroy_A3d_gathered_on_2d(zSOLVEstruct_t *SOLVEstruct, gridinfo3d_t *grid3d)
{
    /* free A2d and B2d, which are allocated on all 2D layers*/
    NRformat_loc3d *A3d = SOLVEstruct->A3d;
    NRformat_loc *A2d = A3d->A_nfmt;
    // if (grid3d->zscp.Iam == 0) {
	SUPERLU_FREE( A2d->rowptr );
	SUPERLU_FREE( A2d->colind );
	SUPERLU_FREE( A2d->nzval );
    // }
    SUPERLU_FREE(A3d->row_counts_int);  // free displacements and counts
    SUPERLU_FREE(A3d->row_disp);
    SUPERLU_FREE(A3d->nnz_counts_int);
    SUPERLU_FREE(A3d->nnz_disp);
    SUPERLU_FREE(A3d->b_counts_int);
    SUPERLU_FREE(A3d->b_disp);
    int rankorder = grid3d->rankorder;
    if ( rankorder == 0 ) { /* Z-major in 3D grid */
        SUPERLU_FREE(A3d->procs_to_send_list);
        SUPERLU_FREE(A3d->send_count_list);
        SUPERLU_FREE(A3d->procs_recv_from_list);
        SUPERLU_FREE(A3d->recv_count_list);
    }
    SUPERLU_FREE( A2d );         // free 2D structure
    SUPERLU_FREE( A3d );         // free 3D structure
} /* zDestroy_A3d_gathered_on_2d_allgrid */
#endif

/*! \brief Check the inf-norm of the error vector
 */
void pzinf_norm_error(int iam, int_t n, int_t nrhs, doublecomplex x[], int_t ldx,
		      doublecomplex xtrue[], int_t ldxtrue, MPI_Comm slucomm)
{
    double err, xnorm, temperr, tempxnorm;
    doublecomplex *x_work, *xtrue_work;
    doublecomplex temp;
    int i, j;
    double errcomp;  // componentwise error
    double derr;

    for (j = 0; j < nrhs; j++) {
      x_work = &x[j*ldx];
      xtrue_work = &xtrue[j*ldxtrue];
      err = xnorm = errcomp = 0.0;
      for (i = 0; i < n; i++) {
        z_sub(&temp, &x_work[i], &xtrue_work[i]);
	err = SUPERLU_MAX(err, slud_z_abs(&temp));
	xnorm = SUPERLU_MAX(xnorm, slud_z_abs(&x_work[i]));
        errcomp = SUPERLU_MAX(errcomp, slud_z_abs(&temp) / slud_z_abs(&x_work[i]) );
      }

      /* get the golbal max err & xnrom */
      temperr = err;
      MPI_Allreduce( &temperr, &err, 1, MPI_DOUBLE, MPI_MAX, slucomm);
      tempxnorm = xnorm;
      MPI_Allreduce( &tempxnorm, &xnorm, 1, MPI_DOUBLE, MPI_MAX, slucomm);
      temperr = errcomp;
      MPI_Allreduce( &temperr, &errcomp, 1, MPI_FLOAT, MPI_MAX, slucomm);

      err = err / xnorm;
      if ( !iam ) {
	printf(".. Sol %2d: ||X - Xtrue|| / ||X|| = %e\t max_i |x - xtrue|_i / |x|_i = %e\n", j, err, errcomp);
	fflush(stdout);
      }
    }
}

/*! \brief Destroy broadcast and reduction trees used in triangular solve */
void
zDestroy_Tree(int_t n, gridinfo_t *grid, zLUstruct_t *LUstruct)
{
    int i, nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
#if ( DEBUGlevel>=1 )
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    CHECK_MALLOC(iam, "Enter zDestroy_Tree()");
#endif

    nsupers = Glu_persist->supno[n-1] + 1;

    nb = CEILING(nsupers, grid->npcol);
    for (i=0;i<nb;++i){
        if(Llu->LBtree_ptr[i].empty_==NO){
			// BcTree_Destroy(Llu->LBtree_ptr[i],LUstruct->dt);
            C_BcTree_Nullify(&Llu->LBtree_ptr[i]);
	}
        if(Llu->UBtree_ptr[i].empty_==NO){
			// BcTree_Destroy(Llu->UBtree_ptr[i],LUstruct->dt);
            C_BcTree_Nullify(&Llu->UBtree_ptr[i]);
	}
    }
    SUPERLU_FREE(Llu->LBtree_ptr);
    SUPERLU_FREE(Llu->UBtree_ptr);

    nb = CEILING(nsupers, grid->nprow);
    for (i=0;i<nb;++i){
        if(Llu->LRtree_ptr[i].empty_==NO){
			// RdTree_Destroy(Llu->LRtree_ptr[i],LUstruct->dt);
            C_RdTree_Nullify(&Llu->LRtree_ptr[i]);
	}
        if(Llu->URtree_ptr[i].empty_==NO){
			// RdTree_Destroy(Llu->URtree_ptr[i],LUstruct->dt);
            C_RdTree_Nullify(&Llu->URtree_ptr[i]);
	}
    }
    SUPERLU_FREE(Llu->LRtree_ptr);
    SUPERLU_FREE(Llu->URtree_ptr);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit zDestroy_Tree()");
#endif
}


