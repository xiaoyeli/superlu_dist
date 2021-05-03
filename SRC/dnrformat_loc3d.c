/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



/*! @file
 * \brief Preprocessing routines for the 3D factorization/solve codes:
 *        - Gather {A,B} from 3D grid to 2D process layer 0
 *        - Scatter B (solution) from 2D process layer 0 to 3D grid
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Oak Ridge National Lab.
 * October 22, 2020
 */

#include "superlu_ddefs.h"

/* Dst <- BlockByBlock (Src), reshape the block storage. */
static void matCopy(int n, int m, double *Dst, int lddst, double *Src, int ldsrc)
{
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; ++i)
        {
            Dst[i + lddst * j] = Src[i + ldsrc * j];
        }

    return;
}

/*
 * Gather {A,B} from 3D grid to 2D process layer 0
 *     Input:  {A, B, ldb} are distributed on 3D process grid
 *     Output: {A2d, B2d} are distributed on layer 0 2D process grid
 *             output is in the returned A3d->{} structure.
 *             see supermatrix.h for nrformat_loc3d{} structure.
 */
NRformat_loc3d *dGatherNRformat_loc3d(NRformat_loc *A, // input, on 3D grid
                                      double *B,       // input
				      int ldb, int nrhs, // input
                                      gridinfo3d_t *grid3d)
{
    NRformat_loc3d *A3d = SUPERLU_MALLOC(sizeof(NRformat_loc3d));
    NRformat_loc *A2d = SUPERLU_MALLOC(sizeof(NRformat_loc));
    A3d->m_loc = A->m_loc;
    A3d->B3d = (double *) B; // on 3D process grid
    A3d->ldb = ldb;
    A3d->nrhs = nrhs;

    // find number of nnzs
    int_t *nnz_counts; // number of local nonzeros relative to all processes
    int_t *row_counts; // number of local rows relative to all processes
    int *nnz_counts_int, *row_counts_int; // 32-bit
    int *nnz_disp, *row_disp; // displacement
    int *b_counts_int; // number of local B entries relative to all processes 
    int *b_disp;       // including 'nrhs'

    nnz_counts = SUPERLU_MALLOC(grid3d->npdep * sizeof(int_t));
    row_counts = SUPERLU_MALLOC(grid3d->npdep * sizeof(int_t));
    nnz_counts_int = SUPERLU_MALLOC(grid3d->npdep * sizeof(int));
    row_counts_int = SUPERLU_MALLOC(grid3d->npdep * sizeof(int));
    b_counts_int = SUPERLU_MALLOC(grid3d->npdep * sizeof(int));
    MPI_Gather(&A->nnz_loc, 1, mpi_int_t, nnz_counts,
               1, mpi_int_t, 0, grid3d->zscp.comm);
    MPI_Gather(&A->m_loc, 1, mpi_int_t, row_counts,
               1, mpi_int_t, 0, grid3d->zscp.comm);
    nnz_disp = SUPERLU_MALLOC((grid3d->npdep + 1) * sizeof(int));
    row_disp = SUPERLU_MALLOC((grid3d->npdep + 1) * sizeof(int));
    b_disp = SUPERLU_MALLOC((grid3d->npdep + 1) * sizeof(int));

    nnz_disp[0] = 0;
    row_disp[0] = 0;
    b_disp[0] = 0;
    for (int i = 0; i < grid3d->npdep; i++)
    {
        nnz_disp[i + 1] = nnz_disp[i] + nnz_counts[i];
        row_disp[i + 1] = row_disp[i] + row_counts[i];
        b_disp[i + 1] = nrhs * row_disp[i + 1];
        nnz_counts_int[i] = nnz_counts[i];
        row_counts_int[i] = row_counts[i];
        b_counts_int[i] = nrhs * row_counts[i];
    }

    if (grid3d->zscp.Iam == 0)
    {
        A2d->colind = SUPERLU_MALLOC(nnz_disp[grid3d->npdep] * sizeof(int_t));
        A2d->nzval = SUPERLU_MALLOC(nnz_disp[grid3d->npdep] * sizeof(double));
        A2d->rowptr = SUPERLU_MALLOC((row_disp[grid3d->npdep] + 1) * sizeof(int_t));
        A2d->rowptr[0] = 0;
    }

    MPI_Gatherv(A->nzval, A->nnz_loc, MPI_DOUBLE, A2d->nzval,
                nnz_counts_int, nnz_disp,
                MPI_DOUBLE, 0, grid3d->zscp.comm);
    MPI_Gatherv(A->colind, A->nnz_loc, mpi_int_t, A2d->colind,
                nnz_counts_int, nnz_disp,
                mpi_int_t, 0, grid3d->zscp.comm);
    MPI_Gatherv(&A->rowptr[1], A->m_loc, mpi_int_t, &A2d->rowptr[1],
                row_counts_int, row_disp,
                mpi_int_t, 0, grid3d->zscp.comm);

    if (grid3d->zscp.Iam == 0)
    {
        for (int i = 0; i < grid3d->npdep; i++)
        {
            for (int j = row_disp[i] + 1; j < row_disp[i + 1] + 1; j++)
            {
                // A2d->rowptr[j] += row_disp[i];
                A2d->rowptr[j] += nnz_disp[i];
            }
        }
        A2d->nnz_loc = nnz_disp[grid3d->npdep];
        A2d->m_loc = row_disp[grid3d->npdep];

        if (grid3d->rankorder == 1) { // XY-major
     	    A2d->fst_row = A->fst_row;
	} else { // Z-major
	    gridinfo_t *grid2d = &(grid3d->grid2d);
            int procs2d = grid2d->nprow * grid2d->npcol;
            int m_loc_2d = A2d->m_loc;
            int *m_loc_2d_counts = SUPERLU_MALLOC(procs2d * sizeof(int));

            MPI_Allgather(&m_loc_2d, 1, MPI_INT, m_loc_2d_counts, 1, 
	                  MPI_INT, grid2d->comm);

            int fst_row = 0;
            for (int p = 0; p < procs2d; ++p)
            {
		if (grid2d->iam == p)
                   A2d->fst_row = fst_row;
            	fst_row += m_loc_2d_counts[p];
            }

            SUPERLU_FREE(m_loc_2d_counts);
        }
    }

    // Btmp <- compact(B)
    // compacting B
    double *Btmp;
    Btmp = SUPERLU_MALLOC(A->m_loc * nrhs * sizeof(double));
    matCopy(A->m_loc, nrhs, Btmp, A->m_loc, B, ldb);

    double *B1;
    if (grid3d->zscp.Iam == 0)
    {
        B1 = SUPERLU_MALLOC(A2d->m_loc * nrhs * sizeof(double));
        A3d->B2d = (double *) SUPERLU_MALLOC(A2d->m_loc * nrhs * sizeof(double));
    }

    // B1 <- gatherv(Btmp)
    MPI_Gatherv(Btmp, nrhs * A->m_loc, MPI_DOUBLE, B1,
                b_counts_int, b_disp,
                MPI_DOUBLE, 0, grid3d->zscp.comm);

    // B2d <- colMajor(B1)
    if (grid3d->zscp.Iam == 0)
    {
        for (int i = 0; i < grid3d->npdep; ++i)
        {
            /* code */
            matCopy(row_counts_int[i], nrhs, ((double*)A3d->B2d) + row_disp[i],
		    A2d->m_loc, B1 + nrhs * row_disp[i], row_counts_int[i]);
        }

        SUPERLU_FREE(B1);
    }

    A3d->A_nfmt = A2d;
    A3d->b_counts_int = b_counts_int;
    A3d->b_disp = b_disp;
    A3d->row_counts_int = row_counts_int;
    A3d->row_disp = row_disp;

    /* free storage */
    SUPERLU_FREE(nnz_counts);
    SUPERLU_FREE(nnz_counts_int);
    SUPERLU_FREE(row_counts);
    SUPERLU_FREE(nnz_disp);
    SUPERLU_FREE(Btmp);

    return A3d;

} /* dGatherNRformat_loc3d */

/*
 * Scatter B (solution) from 2D process layer 0 to 3D grid
 *   Output: X3d <- A^{-1} B2d
 */
int dScatter_B3d(NRformat_loc3d *A3d,  // modified
		 gridinfo3d_t *grid3d)
{
    double *B = (double *) A3d->B3d; // on 3D grid
    int ldb = A3d->ldb;
    int nrhs = A3d->nrhs;
    double *B2d = (double *) A3d->B2d; // on 2D layer 0 
    NRformat_loc A2d = *(A3d->A_nfmt);

    /* The following are the number of local rows relative to all processes */
    int m_loc = A3d->m_loc;
    int *b_counts_int = A3d->b_counts_int;
    int *b_disp = A3d->b_disp;
    int *row_counts_int = A3d->row_counts_int;
    int *row_disp = A3d->row_disp;
    int i, p;
    int iam = grid3d->iam;
    int rankorder = grid3d->rankorder;
    gridinfo_t *grid2d = &(grid3d->grid2d);

    double *B1;  // on 2D layer 0
    if (grid3d->zscp.Iam == 0)
    {
        B1 = SUPERLU_MALLOC(A2d.m_loc * nrhs * sizeof(double));
    }

    // B1 <- BlockByBlock(B2d)
    if (grid3d->zscp.Iam == 0)
    {
        for (i = 0; i < grid3d->npdep; ++i)
        {
            /* code */
            matCopy(row_counts_int[i], nrhs, B1 + nrhs * row_disp[i], row_counts_int[i],
                    B2d + row_disp[i], A2d.m_loc);
        }
    }

    double *Btmp; // on 3D grid
    Btmp = SUPERLU_MALLOC(A3d->m_loc * nrhs * sizeof(double));

    // Btmp <- scatterv(B1), block-by-block
    if ( rankorder == 1 ) { /* XY-major in 3D grid */
        /*    e.g. 1x3x4 grid: layer0 layer1 layer2 layer3
	 *                     0      1      2      3
	 *                     4      5      6      7
	 *                     8      9      10     11
	 */
        MPI_Scatterv(B1, b_counts_int, b_disp, MPI_DOUBLE,
		     Btmp, nrhs * A3d->m_loc, MPI_DOUBLE,
		     0, grid3d->zscp.comm);

    } else { /* Z-major in 3D grid */
        /*    e.g. 1x3x4 grid: layer0 layer1 layer2 layer3
	                       0      3      6      9
 	                       1      4      7      10      
	                       2      5      8      11
	  GATHER:  {A, B} in A * X = B
	  layer-0:
    	       B (row space)  X (column space)  SCATTER
	       ----           ----        ---->>
           P0  0              0
(equations     3              1      Proc 0 -> Procs {0, 1, 2, 3}
 reordered     6              2
 after gather) 9              3
	       ----           ----
	   P1  1              4      Proc 1 -> Procs {4, 5, 6, 7}
	       4              5
               7              6
               10             7
	       ----           ----
	   P2  2              8      Proc 2 -> Procs {8, 9, 10, 11}
	       5              9
	       8             10
	       11            11
	       ----         ----
	*/
        MPI_Request recv_req;
	MPI_Status recv_status;
	int pxy = grid2d->nprow * grid2d->npcol;
	int npdep = grid3d->npdep, dest, src, tag;
	int nprocs = pxy * npdep;

	/* Everyone receives one block (post non-blocking irecv) */
	src = grid3d->iam / npdep;  // Z-major
	tag = iam;
	MPI_Irecv(Btmp, nrhs * A3d->m_loc, MPI_DOUBLE,
		 src, tag, grid3d->comm, &recv_req);

	/* Layer 0 sends to npdep procs */
	if (grid3d->zscp.Iam == 0) {
	    int dest, tag;
	    for (p = 0; p < npdep; ++p) { // send to npdep procs
	        dest = p + grid2d->iam * npdep; // Z-major order
		tag = dest;

		MPI_Send(B1 + b_disp[p], b_counts_int[p], 
			 MPI_DOUBLE, dest, tag, grid3d->comm);
	    }
	}  /* end layer 0 send */
    
	/* Wait for Irecv to complete */
	MPI_Wait(&recv_req, &recv_status);

    } /* else Z-major */

    // B <- colMajor(Btmp)
    matCopy(A3d->m_loc, nrhs, B, ldb, Btmp, A3d->m_loc);

    /* free storage */
    SUPERLU_FREE(A3d->b_counts_int);
    SUPERLU_FREE(A3d->b_disp);
    SUPERLU_FREE(A3d->row_counts_int);
    SUPERLU_FREE(A3d->row_disp);
    SUPERLU_FREE(Btmp);
    if (grid3d->zscp.Iam == 0) SUPERLU_FREE(B1);

    return 0;
} /* dScatter_B3d */
