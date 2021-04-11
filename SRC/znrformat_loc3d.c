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

#include "superlu_zdefs.h"

/* Dst <- BlockByBlock (Src), reshape the block storage. */
static void matCopy(int n, int m, doublecomplex *Dst, int lddst, doublecomplex *Src, int ldsrc)
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
NRformat_loc3d *zGatherNRformat_loc3d(NRformat_loc *A, // input, on 3D grid
                                      doublecomplex *B,       // input
				      int ldb, int nrhs, // input
                                      gridinfo3d_t *grid3d)
{
    NRformat_loc3d *A3d = SUPERLU_MALLOC(sizeof(NRformat_loc3d));
    NRformat_loc *A2d = SUPERLU_MALLOC(sizeof(NRformat_loc));
    A3d->m_loc = A->m_loc;
    A3d->B = (doublecomplex *) B; // on 3D process grid
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
        A2d->nzval = SUPERLU_MALLOC(nnz_disp[grid3d->npdep] * sizeof(doublecomplex));
        A2d->rowptr = SUPERLU_MALLOC((row_disp[grid3d->npdep] + 1) * sizeof(int_t));
        A2d->rowptr[0] = 0;
    }

    MPI_Gatherv(A->nzval, A->nnz_loc, SuperLU_MPI_DOUBLE_COMPLEX, A2d->nzval,
                nnz_counts_int, nnz_disp,
                SuperLU_MPI_DOUBLE_COMPLEX, 0, grid3d->zscp.comm);
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
#if 0	
        A2d->fst_row = A->fst_row; // This is a bug
#else
        gridinfo_t *grid2d = &(grid3d->grid2d);
        int procs2d = grid2d->nprow * grid2d->npcol;
        int m_loc_2d = A2d->m_loc;
        int *m_loc_2d_counts = SUPERLU_MALLOC(procs2d * sizeof(int));

        MPI_Allgather(&m_loc_2d, 1, MPI_INT, m_loc_2d_counts, 1, MPI_INT, grid2d->comm);

        int fst_row = 0;
        for (int p = 0; p < procs2d; ++p)
        {
            if (grid2d->iam == p)
                A2d->fst_row = fst_row;
            fst_row += m_loc_2d_counts[p];
        }

        SUPERLU_FREE(m_loc_2d_counts);
#endif
    }
    // Btmp <- compact(B)
    // compacting B
    doublecomplex *Btmp;
    Btmp = SUPERLU_MALLOC(A->m_loc * nrhs * sizeof(doublecomplex));
    matCopy(A->m_loc, nrhs, Btmp, A->m_loc, B, ldb);

    doublecomplex *B1;
    if (grid3d->zscp.Iam == 0)
    {
        B1 = SUPERLU_MALLOC(A2d->m_loc * nrhs * sizeof(doublecomplex));
        A3d->B2d = (doublecomplex *) SUPERLU_MALLOC(A2d->m_loc * nrhs * sizeof(doublecomplex));
    }

    // B1 <- gatherv(Btmp)
    MPI_Gatherv(Btmp, nrhs * A->m_loc, SuperLU_MPI_DOUBLE_COMPLEX, B1,
                b_counts_int, b_disp,
                SuperLU_MPI_DOUBLE_COMPLEX, 0, grid3d->zscp.comm);

    // B2d <- colMajor(B1)
    if (grid3d->zscp.Iam == 0)
    {
        for (int i = 0; i < grid3d->npdep; ++i)
        {
            /* code */
            matCopy(row_counts_int[i], nrhs, ((doublecomplex*)A3d->B2d) + row_disp[i],
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

} /* zGatherNRformat_loc3d */

/*
 * Scatter B (solution) from 2D process layer 0 to 3D grid
 *   Output: X2d <- A^{-1} B2d
 */
int zScatter_B3d(NRformat_loc3d *A3d,  // modified
		 gridinfo3d_t *grid3d)
{
    doublecomplex *B = (doublecomplex *) A3d->B; // on 3D grid
    int ldb = A3d->ldb;
    int nrhs = A3d->nrhs;
    doublecomplex *B2d = (doublecomplex *) A3d->B2d; // on 2D layer 0 
    NRformat_loc A2d = *(A3d->A_nfmt);

    /* The following are the number of local rows relative to all processes */
    int m_loc = A3d->m_loc;
    int *b_counts_int = A3d->b_counts_int;
    int *b_disp = A3d->b_disp;
    int *row_counts_int = A3d->row_counts_int;
    int *row_disp = A3d->row_disp;

    gridinfo_t *grid2d = &(grid3d->grid2d);
    int iam = grid3d->iam;

    doublecomplex *B1;  // on 2D layer 0
    if (grid3d->zscp.Iam == 0)
    {
        B1 = SUPERLU_MALLOC(A2d.m_loc * nrhs * sizeof(doublecomplex));
    }

    // B1 <- blockByBlock(b2d)
    if (grid3d->zscp.Iam == 0)
    {
        for (int i = 0; i < grid3d->npdep; ++i)
        {
            /* code */
            matCopy(row_counts_int[i], nrhs, B1 + nrhs * row_disp[i], row_counts_int[i],
                    B2d + row_disp[i], A2d.m_loc);
        }
    }

    doublecomplex *Btmp; // on 3D grid
    Btmp = SUPERLU_MALLOC(A3d->m_loc * nrhs * sizeof(doublecomplex));

#if 0 // This is a bug: the result of this scatter is a "permuted" distribution
    // Btmp <- scatterv(B1) 
    MPI_Scatterv(B1, b_counts_int, b_disp, SuperLU_MPI_DOUBLE_COMPLEX,
                 Btmp, nrhs * A3d->m_loc, SuperLU_MPI_DOUBLE_COMPLEX, 0, grid3d->zscp.comm);
#else
    /* For example, in 1x3x4 grid, layer 0 has procs:{0,1,2}, the process 
       scattering pattern is:
           0 -> {0,1,2,3}, 1 -> {4,5,6,7}, 2 -> {8,9,10,11}
       This is different from the scattering pattern along Z-dimension.
     */
    if (grid3d->zscp.Iam == 0) // processes on layer 0
    {
      MPI_Request send_req;
      for (int p = 0; p < grid3d->npdep; ++p) { // send to npdep procs
	int dest = p + grid2d->iam * grid3d->npdep;
	int tag = dest;

	MPI_Isend(B1 + b_disp[p], b_counts_int[p], SuperLU_MPI_DOUBLE_COMPLEX,
		  dest, tag, grid3d->comm, &send_req);
      }
    } 
    
    /* Everyone receives one block */
    MPI_Status status;
    int src = grid3d->iam / grid3d->npdep;  // which proc the data should come from
    MPI_Recv(Btmp, nrhs * A3d->m_loc, SuperLU_MPI_DOUBLE_COMPLEX,
	     src, grid3d->iam, grid3d->comm, &status);
#endif

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
} /* zScatter_B3d */
