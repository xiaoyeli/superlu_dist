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
 * -- Distributed SuperLU routine (version 7.1.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab.
 * May 12, 2021
 * October 5, 2021
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
void dGatherNRformat_loc3d
(
 fact_t Fact,     // how matrix A will be factorized
 NRformat_loc *A, // input, on 3D grid
 double *B,       // input
 int ldb, int nrhs, // input
 gridinfo3d_t *grid3d, 
 NRformat_loc3d **A3d_addr /* If Fact == DOFACT, it is an input;
 		              Else it is both input and may be modified */
 )
{
    NRformat_loc3d *A3d = (NRformat_loc3d *) *A3d_addr;
    NRformat_loc *A2d;
    int *row_counts_int; // 32-bit, number of local rows relative to all processes
    int *row_disp;       // displacement
    int *nnz_counts_int; // number of local nnz relative to all processes
    int *nnz_disp;       // displacement
    int *b_counts_int;   // number of local B entries relative to all processes 
    int *b_disp;         // including 'nrhs'
	
    /********* Gather A2d *********/
    if ( Fact == DOFACT ) { /* Factorize from scratch */
	/* A3d is output. Compute counts from scratch */
	A3d = SUPERLU_MALLOC(sizeof(NRformat_loc3d));
	A3d->num_procs_to_send = EMPTY; // No X(2d) -> X(3d) comm. schedule yet
	A2d = SUPERLU_MALLOC(sizeof(NRformat_loc));
    
	// find number of nnzs
	int_t *nnz_counts; // number of local nonzeros relative to all processes
	int_t *row_counts; // number of local rows relative to all processes
	int *nnz_counts_int; // 32-bit
	int *nnz_disp; // displacement

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
	int nrhs1 = nrhs; // input 
	if ( nrhs <= 0 ) nrhs1 = 1; /* Make sure to compute offsets and
	                               counts for future use.   */
	for (int i = 0; i < grid3d->npdep; i++)
	    {
		nnz_disp[i + 1] = nnz_disp[i] + nnz_counts[i];
		row_disp[i + 1] = row_disp[i] + row_counts[i];
		b_disp[i + 1] = nrhs1 * row_disp[i + 1];
		nnz_counts_int[i] = nnz_counts[i];
		row_counts_int[i] = row_counts[i];
		b_counts_int[i] = nrhs1 * row_counts[i];
	    }

	if (grid3d->zscp.Iam == 0)
	    {
		A2d->colind = intMalloc_dist(nnz_disp[grid3d->npdep]);
		A2d->nzval = doubleMalloc_dist(nnz_disp[grid3d->npdep]);
		A2d->rowptr = intMalloc_dist((row_disp[grid3d->npdep] + 1));
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

	if (grid3d->zscp.Iam == 0) /* Set up rowptr[] relative to 2D grid-0 */
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
	    } /* end 2D layer grid-0 */

	A3d->A_nfmt         = A2d;
	A3d->row_counts_int = row_counts_int;
	A3d->row_disp       = row_disp;
	A3d->nnz_counts_int = nnz_counts_int;
	A3d->nnz_disp       = nnz_disp;
	A3d->b_counts_int   = b_counts_int;
	A3d->b_disp         = b_disp;

	/* free storage */
	SUPERLU_FREE(nnz_counts);
	SUPERLU_FREE(row_counts);
	
	*A3d_addr = (NRformat_loc3d *) A3d; // return pointer to A3d struct
	
    } else if ( Fact == SamePattern || Fact == SamePattern_SameRowPerm ) {
	/* A3d is input. No need to recompute count.
	   Only need to gather A2d matrix; the previous 2D matrix
	   was overwritten by equilibration, perm_r and perm_c.  */
	NRformat_loc *A2d = A3d->A_nfmt;
	row_counts_int = A3d->row_counts_int;
	row_disp       = A3d->row_disp;
	nnz_counts_int = A3d->nnz_counts_int;
	nnz_disp       = A3d->nnz_disp;

	MPI_Gatherv(A->nzval, A->nnz_loc, MPI_DOUBLE, A2d->nzval,
		    nnz_counts_int, nnz_disp,
		    MPI_DOUBLE, 0, grid3d->zscp.comm);
	MPI_Gatherv(A->colind, A->nnz_loc, mpi_int_t, A2d->colind,
		    nnz_counts_int, nnz_disp,
		    mpi_int_t, 0, grid3d->zscp.comm);
	MPI_Gatherv(&A->rowptr[1], A->m_loc, mpi_int_t, &A2d->rowptr[1],
		    row_counts_int, row_disp,
		    mpi_int_t, 0, grid3d->zscp.comm);
		    
	if (grid3d->zscp.Iam == 0) { /* Set up rowptr[] relative to 2D grid-0 */
	    A2d->rowptr[0] = 0;
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
	} /* end 2D layer grid-0 */
    } /* SamePattern or SamePattern_SameRowPerm */

    A3d->m_loc = A->m_loc;
    A3d->B3d = (double *) B; /* save the pointer to the original B
				    stored on 3D process grid.  */
    A3d->ldb = ldb;
    A3d->nrhs = nrhs; // record the input 
	
    /********* Gather B2d **********/
    if ( nrhs > 0 ) {
	
	A2d = (NRformat_loc *) A3d->A_nfmt; // matrix A gathered on 2D grid-0
	row_counts_int = A3d->row_counts_int;
	row_disp       = A3d->row_disp;
	b_counts_int   = A3d->b_counts_int;
	b_disp         = A3d->b_disp;;
	
	/* Btmp <- compact(B), compacting B */
	double *Btmp;
	Btmp = SUPERLU_MALLOC(A->m_loc * nrhs * sizeof(double));
	matCopy(A->m_loc, nrhs, Btmp, A->m_loc, B, ldb);

	double *B1;
	if (grid3d->zscp.Iam == 0)
	    {
		B1 = doubleMalloc_dist(A2d->m_loc * nrhs);
		A3d->B2d = doubleMalloc_dist(A2d->m_loc * nrhs);
	    }

	// B1 <- gatherv(Btmp)
	MPI_Gatherv(Btmp, nrhs * A->m_loc, MPI_DOUBLE, B1,
		    b_counts_int, b_disp,
		    MPI_DOUBLE, 0, grid3d->zscp.comm);
	SUPERLU_FREE(Btmp);

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

    } /* end gather B2d */

} /* dGatherNRformat_loc3d */

/*
 * Scatter B (solution) from 2D process layer 0 to 3D grid
 *   Output: X3d <- A^{-1} B2d
 */
int dScatter_B3d(NRformat_loc3d *A3d,  // modified
		 gridinfo3d_t *grid3d)
{
    double *B = (double *) A3d->B3d; // retrieve original pointer on 3D grid
    int ldb = A3d->ldb;
    int nrhs = A3d->nrhs;
    double *B2d = (double *) A3d->B2d; // only on 2D layer grid_0 
    NRformat_loc *A2d = A3d->A_nfmt;

    /* The following are the number of local rows relative to Z-dimension */
    int m_loc           = A3d->m_loc;
    int *b_counts_int   = A3d->b_counts_int;
    int *b_disp         = A3d->b_disp;
    int *row_counts_int = A3d->row_counts_int;
    int *row_disp       = A3d->row_disp;
    int i, j, k, p;
    int num_procs_to_send, num_procs_to_recv; // persistent across multiple solves
    int iam = grid3d->iam;
    int rankorder = grid3d->rankorder;
    gridinfo_t *grid2d = &(grid3d->grid2d);

    double *B1;  // on 2D layer 0
    if (grid3d->zscp.Iam == 0)
    {
        B1 = doubleMalloc_dist(A2d->m_loc * nrhs);
    }

    // B1 <- BlockByBlock(B2d)
    if (grid3d->zscp.Iam == 0)
    {
        for (i = 0; i < grid3d->npdep; ++i)
        {
            /* code */
            matCopy(row_counts_int[i], nrhs, B1 + nrhs * row_disp[i], row_counts_int[i],
                    B2d + row_disp[i], A2d->m_loc);
        }
    }

    double *Btmp; // on 3D grid
    Btmp = doubleMalloc_dist(A3d->m_loc * nrhs);

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

    } else { /* Z-major in 3D grid (default) */
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
         In the most general case, block rows of B are not of even size, then the
	 Layer 0 partition may overlap with 3D partition in an arbitrary manner.
	 For example:
	                  P0        P1        P2       P3
             X on grid-0: |___________|__________|_________|________|

	     X on 3D:     |___|____|_____|____|__|______|_____|_____|
	                  P0  P1   P2    P3   P4   P5     P6   P7  
	*/
	MPI_Status recv_status;
	int pxy = grid2d->nprow * grid2d->npcol;
	int npdep = grid3d->npdep, dest, src, tag;
	int nprocs = pxy * npdep; // all procs in 3D grid 
	MPI_Request *recv_reqs = (MPI_Request*) SUPERLU_MALLOC(npdep * sizeof(MPI_Request));
	int num_procs_to_send;
	int *procs_to_send_list;
	int *send_count_list;
	int num_procs_to_recv;
	int *procs_recv_from_list;
	int *recv_count_list;

	if ( A3d->num_procs_to_send == -1 ) { /* First time: set up communication schedule */
	    /* 1. Set up the destination processes from each source process,
	       and the send counts.	
	       - Only grid-0 processes need to send.
	       - row_disp[] recorded the prefix sum of the block rows of RHS
	       	 	    along the processes Z-dimension.
	         row_disp[npdep] is the total number of X entries on my proc.
	       	     (equals A2d->m_loc.)
	         A2d->fst_row records the boundary of the partition on grid-0.
	       - Need to compute the prefix sum of the block rows of X
	       	 among all the processes.
	       	 A->fst_row has this info, but is available only locally.
	    */
	
	    int *m_loc_3d_counts = SUPERLU_MALLOC(nprocs * sizeof(int));
	
	    /* related to m_loc in 3D partition */
	    int *x_send_counts = SUPERLU_MALLOC(nprocs * sizeof(int));
	    int *x_recv_counts = SUPERLU_MALLOC(nprocs * sizeof(int));
	
	    /* The following should be persistent across multiple solves.
	       These lists avoid All-to-All communication. */
	    procs_to_send_list = SUPERLU_MALLOC(nprocs * sizeof(int));
	    send_count_list = SUPERLU_MALLOC(nprocs * sizeof(int));
	    procs_recv_from_list = SUPERLU_MALLOC(nprocs * sizeof(int));
	    recv_count_list = SUPERLU_MALLOC(nprocs * sizeof(int));

	    for (p = 0; p < nprocs; ++p) {
		x_send_counts[p] = 0;
		x_recv_counts[p] = 0;
		procs_to_send_list[p] = EMPTY; // (-1)
		procs_recv_from_list[p] = EMPTY;
	    }
	    
	    /* All procs participate */
	    MPI_Allgather(&(A3d->m_loc), 1, MPI_INT, m_loc_3d_counts, 1,
			  MPI_INT, grid3d->comm);
	    
	    /* Layer 0 set up sends info. The other layers have 0 send counts. */
	    if (grid3d->zscp.Iam == 0) {
		int x_fst_row = A2d->fst_row; // start from a layer 0 boundary
		int x_end_row = A2d->fst_row + A2d->m_loc; // end of boundary + 1
		int sum_m_loc; // prefix sum of m_loc among all processes
		
		/* Loop through all processes.
		   Search for 1st X-interval in grid-0's B-interval */
		num_procs_to_send = sum_m_loc = 0;
		for (p = 0; p < nprocs; ++p) {
		    
		    sum_m_loc += m_loc_3d_counts[p];
		    
		    if (sum_m_loc > x_end_row) { // reach the 2D block boundary
			x_send_counts[p] = x_end_row - x_fst_row;
			procs_to_send_list[num_procs_to_send] = p;
			send_count_list[num_procs_to_send] = x_send_counts[p];
			num_procs_to_send++;
			break;
		    } else if (x_fst_row < sum_m_loc) {
			x_send_counts[p] = sum_m_loc - x_fst_row;
			procs_to_send_list[num_procs_to_send] = p;
			send_count_list[num_procs_to_send] = x_send_counts[p];
			num_procs_to_send++;
			x_fst_row = sum_m_loc; //+= m_loc_3d_counts[p];
			if (x_fst_row >= x_end_row) break;
		    }
		    
		    //sum_m_loc += m_loc_3d_counts[p+1];
		} /* end for p ... */
	    } else { /* end layer 0 */
		num_procs_to_send = 0;
	    }
	    
	    /* 2. Set up the source processes from each destination process,
	       and the recv counts.
	       All processes may need to receive something from grid-0. */
	    /* The following transposes x_send_counts matrix to
	       x_recv_counts matrix */
	    MPI_Alltoall(x_send_counts, 1, MPI_INT, x_recv_counts, 1, MPI_INT,
			 grid3d->comm);
	    
	    j = 0; // tracking number procs to receive from
	    for (p = 0; p < nprocs; ++p) {
		if (x_recv_counts[p]) {
		    procs_recv_from_list[j] = p;
		    recv_count_list[j] = x_recv_counts[p];
		    src = p;  tag = iam;
		    ++j;
#if 0		    
		    printf("RECV: src %d -> iam %d, x_recv_counts[p] %d, tag %d\n",
			   src, iam, x_recv_counts[p], tag);
		    fflush(stdout);
#endif		    
		}
	    }
	    num_procs_to_recv = j;

	    /* Persist in A3d structure */
	    A3d->num_procs_to_send = num_procs_to_send;
	    A3d->procs_to_send_list = procs_to_send_list;
	    A3d->send_count_list = send_count_list;
	    A3d->num_procs_to_recv = num_procs_to_recv;
	    A3d->procs_recv_from_list = procs_recv_from_list;
	    A3d->recv_count_list = recv_count_list;

	    SUPERLU_FREE(m_loc_3d_counts);
	    SUPERLU_FREE(x_send_counts);
	    SUPERLU_FREE(x_recv_counts);
	} else { /* Reuse the communication schedule */
	    num_procs_to_send = A3d->num_procs_to_send;
	    procs_to_send_list = A3d->procs_to_send_list;
	    send_count_list = A3d->send_count_list;
	    num_procs_to_recv = A3d->num_procs_to_recv;
	    procs_recv_from_list = A3d->procs_recv_from_list;
	    recv_count_list = A3d->recv_count_list;
	}
	
	/* 3. Perform the acutal communication */
	    
	/* Post irecv first */
	i = 0; // tracking offset in the recv buffer Btmp[]
	for (j = 0; j < num_procs_to_recv; ++j) {
	    src = procs_recv_from_list[j];
	    tag = iam;
	    k = nrhs * recv_count_list[j]; // recv count
	    MPI_Irecv( Btmp + i, k, MPI_DOUBLE,
		       src, tag, grid3d->comm, &recv_reqs[j] );
	    i += k;
	}
	    
	/* Send */
	/* Layer 0 sends to *num_procs_to_send* procs */
	if (grid3d->zscp.Iam == 0) {
	    int dest, tag;
	    for (i = 0, p = 0; p < num_procs_to_send; ++p) { 
		dest = procs_to_send_list[p]; //p + grid2d->iam * npdep;
		tag = dest;
		/*printf("SEND: iam %d -> %d, send_count_list[p] %d, tag %d\n",
		  iam,dest, send_count_list[p], tag);
		  fflush(stdout); */
		    
		MPI_Send(B1 + i, nrhs * send_count_list[p], 
			 MPI_DOUBLE, dest, tag, grid3d->comm);
		i += nrhs * send_count_list[p];
	    }
	}  /* end layer 0 send */
	    
	/* Wait for all Irecv's to complete */
	for (i = 0; i < num_procs_to_recv; ++i)
	    MPI_Wait(&recv_reqs[i], &recv_status);

        SUPERLU_FREE(recv_reqs);

	///////////	
#if 0 // The following code works only with even block distribution of RHS 
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
#endif
	///////////
	
    } /* else Z-major */

    // B <- colMajor(Btmp)
    matCopy(A3d->m_loc, nrhs, B, ldb, Btmp, A3d->m_loc);

    /* free storage */
    SUPERLU_FREE(Btmp);
    if (grid3d->zscp.Iam == 0) {
	SUPERLU_FREE(B1);
	SUPERLU_FREE(B2d);
    }

    return 0;
} /* dScatter_B3d */
