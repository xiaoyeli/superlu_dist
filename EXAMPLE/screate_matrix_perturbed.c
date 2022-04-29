/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Read the matrix from data file
 *
 * <pre>
 * -- Distributed SuperLU routine (version 5.1.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * December 31, 2016
 * </pre>
 */
#include <math.h>
#include "superlu_sdefs.h"

/* \brief
 *
 * <pre>
 * Purpose
 * =======
 * 
 * SCREATE_MATRIX_PERTURBED read the matrix from data file in
 * Harwell-Boeing format, and distribute it to processors in a distributed
 * compressed row format. It also generate the distributed true solution X
 * and the right-hand side RHS.
 *
 * Arguments   
 * =========      
 *
 * A     (output) SuperMatrix*
 *       Local matrix A in NR_loc format. 
 *
 * NRHS  (input) int_t
 *       Number of right-hand sides.
 *
 * RHS   (output) float**
 *       The right-hand side matrix.
 *
 * LDB   (output) int*
 *       Leading dimension of the right-hand side matrix.
 *
 * X     (output) float**
 *       The true solution matrix.
 *
 * LDX   (output) int*
 *       The leading dimension of the true solution matrix.
 *
 * FP    (input) FILE*
 *       The matrix file pointer.
 *
 * GRID  (input) gridinof_t*
 *       The 2D process mesh.
 * </pre>
 */

int screate_matrix_perturbed(SuperMatrix *A, int nrhs, float **rhs,
                   int *ldb, float **x, int *ldx,
                   FILE *fp, gridinfo_t *grid)
{
    SuperMatrix GA;              /* global A */
    float   *b_global, *xtrue_global;  /* replicated on all processes */
    int_t    *rowind, *colptr;	 /* global */
    float   *nzval;             /* global */
    float   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;	 /* local */
    int_t    m, n, nnz;
    int_t    m_loc, fst_row, nnz_loc;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    int_t    row, col, i, j, relpos;
    int      iam;
    char     trans[1];
    int_t      *marker;

    iam = grid->iam;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter screate_matrix()");
#endif

    if ( !iam ) {
        /* Read the matrix stored on disk in Harwell-Boeing format. */
        sreadhb_dist(iam, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);

	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( nzval,  nnz, MPI_FLOAT, 0, grid->comm );
	MPI_Bcast( rowind, nnz, mpi_int_t,  0, grid->comm );
	MPI_Bcast( colptr, n+1, mpi_int_t,  0, grid->comm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid->comm );

	/* Allocate storage for compressed column representation. */
	sallocateA_dist(n, nnz, &nzval, &rowind, &colptr);

	MPI_Bcast( nzval,   nnz, MPI_FLOAT, 0, grid->comm );
	MPI_Bcast( rowind,  nnz, mpi_int_t,  0, grid->comm );
	MPI_Bcast( colptr,  n+1, mpi_int_t,  0, grid->comm );
    }

    /* Perturbed the 1st and last diagonal of the matrix to lower
       values. Intention is to change perm_r[].   */
    nzval[0] *= 0.01;
    nzval[nnz-1] *= 0.0001; 

    /* Compute the number of rows to be distributed to local process */
    m_loc = m / (grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * grid->nprow * grid->npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid->nprow * grid->npcol - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * (grid->nprow * grid->npcol - 1);
    }

    /* Create compressed column matrix for GA. */
    sCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval, rowind, colptr,
				SLU_NC, SLU_S, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    if ( !(b_global = floatMalloc_dist(m*nrhs)) )
        ABORT("Malloc fails for b[]");
    if ( !(xtrue_global = floatMalloc_dist(n*nrhs)) )
        ABORT("Malloc fails for xtrue[]");
    *trans = 'N';

    sGenXtrue_dist(n, nrhs, xtrue_global, n);
    sFillRHS_dist(trans, nrhs, xtrue_global, n, &GA, b_global, m);

    /*************************************************
     * Change GA to a local A with NR_loc format     *
     *************************************************/

    rowptr = (int_t *) intMalloc_dist(m_loc+1);
    marker = (int_t *) intCalloc_dist(n);

    /* Get counts of each row of GA */
    for (i = 0; i < n; ++i)
      for (j = colptr[i]; j < colptr[i+1]; ++j) ++marker[rowind[j]];
    /* Set up row pointers */
    rowptr[0] = 0;
    fst_row = iam * m_loc_fst;
    nnz_loc = 0;
    for (j = 0; j < m_loc; ++j) {
      row = fst_row + j;
      rowptr[j+1] = rowptr[j] + marker[row];
      marker[j] = rowptr[j];
    }
    nnz_loc = rowptr[m_loc];

    nzval_loc = (float *) floatMalloc_dist(nnz_loc);
    colind = (int_t *) intMalloc_dist(nnz_loc);

    /* Transfer the matrix into the compressed row storage */
    for (i = 0; i < n; ++i) {
      for (j = colptr[i]; j < colptr[i+1]; ++j) {
	row = rowind[j];
	if ( (row>=fst_row) && (row<fst_row+m_loc) ) {
	  row = row - fst_row;
	  relpos = marker[row];
	  colind[relpos] = i;
	  nzval_loc[relpos] = nzval[j];
	  ++marker[row];
	}
      }
    }

#if ( DEBUGlevel>=2 )
    if ( !iam ) sPrint_CompCol_Matrix_dist(&GA);
#endif   

    /* Destroy GA */
    Destroy_CompCol_Matrix_dist(&GA);

    /******************************************************/
    /* Change GA to a local A with NR_loc format */
    /******************************************************/

    /* Set up the local A in NR_loc format */
    sCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				   nzval_loc, colind, rowptr,
				   SLU_NR_loc, SLU_S, SLU_GE);
    
    /* Get the local B */
    if ( !((*rhs) = floatMalloc_dist(m_loc*nrhs)) )
        ABORT("Malloc fails for rhs[]");
    for (j =0; j < nrhs; ++j) {
	for (i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    (*rhs)[j*m_loc+i] = b_global[j*n+row];
	}
    }
    *ldb = m_loc;

    /* Set the true X */    
    *ldx = m_loc;
    if ( !((*x) = floatMalloc_dist(*ldx * nrhs)) )
        ABORT("Malloc fails for x_loc[]");

    /* Get the local part of xtrue_global */
    for (j = 0; j < nrhs; ++j) {
      for (i = 0; i < m_loc; ++i)
	(*x)[i + j*(*ldx)] = xtrue_global[i + fst_row + j*n];
    }

    SUPERLU_FREE(b_global);
    SUPERLU_FREE(xtrue_global);
    SUPERLU_FREE(marker);

#if ( DEBUGlevel>=1 )
    printf("sizeof(NRforamt_loc) %lu\n", sizeof(NRformat_loc));
    CHECK_MALLOC(iam, "Exit screate_matrix()");
#endif
    return 0;
}



int screate_matrix_perturbed_postfix(SuperMatrix *A, int nrhs, float **rhs,
                   int *ldb, float **x, int *ldx,
                   FILE *fp, char *postfix, gridinfo_t *grid)
{
    SuperMatrix GA;              /* global A */
    float   *b_global, *xtrue_global;  /* replicated on all processes */
    int_t    *rowind, *colptr;	 /* global */
    float   *nzval;             /* global */
    float   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;	 /* local */
    int_t    m, n, nnz;
    int_t    m_loc, fst_row, nnz_loc;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    int_t    row, col, i, j, relpos;
    int      iam;
    char     trans[1];
    int_t      *marker;

    iam = grid->iam;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter screate_matrix()");
#endif

    if ( !iam ) {
    double t = SuperLU_timer_();       
    if(!strcmp(postfix,"rua")){
		/* Read the matrix stored on disk in Harwell-Boeing format. */
		sreadhb_dist(iam, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"mtx")){
		/* Read the matrix stored on disk in Matrix Market format. */
		sreadMM_dist(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"rb")){
		/* Read the matrix stored on disk in Rutherford-Boeing format. */
		sreadrb_dist(iam, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);		
	}else if(!strcmp(postfix,"dat")){
		/* Read the matrix stored on disk in triplet format. */
		sreadtriple_dist(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"bin")){
		/* Read the matrix stored on disk in binary format. */
		sread_binary(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);		
	}else {
		ABORT("File format not known");
	}

	printf("Time to read and distribute matrix %.2f\n", 
	        SuperLU_timer_() - t);  fflush(stdout);
			
	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( nzval,  nnz, MPI_FLOAT, 0, grid->comm );
	MPI_Bcast( rowind, nnz, mpi_int_t,  0, grid->comm );
	MPI_Bcast( colptr, n+1, mpi_int_t,  0, grid->comm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid->comm );

	/* Allocate storage for compressed column representation. */
	sallocateA_dist(n, nnz, &nzval, &rowind, &colptr);

	MPI_Bcast( nzval,   nnz, MPI_FLOAT, 0, grid->comm );
	MPI_Bcast( rowind,  nnz, mpi_int_t,  0, grid->comm );
	MPI_Bcast( colptr,  n+1, mpi_int_t,  0, grid->comm );
    }

    /* Perturbed the 1st and last diagonal of the matrix to lower
       values. Intention is to change perm_r[].   */
    nzval[0] *= 0.01;
    nzval[nnz-1] *= 0.0001; 

    /* Compute the number of rows to be distributed to local process */
    m_loc = m / (grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * grid->nprow * grid->npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid->nprow * grid->npcol - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * (grid->nprow * grid->npcol - 1);
    }

    /* Create compressed column matrix for GA. */
    sCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval, rowind, colptr,
				SLU_NC, SLU_S, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    if ( !(b_global = floatMalloc_dist(m*nrhs)) )
        ABORT("Malloc fails for b[]");
    if ( !(xtrue_global = floatMalloc_dist(n*nrhs)) )
        ABORT("Malloc fails for xtrue[]");
    *trans = 'N';

    sGenXtrue_dist(n, nrhs, xtrue_global, n);
    sFillRHS_dist(trans, nrhs, xtrue_global, n, &GA, b_global, m);

    /*************************************************
     * Change GA to a local A with NR_loc format     *
     *************************************************/

    rowptr = (int_t *) intMalloc_dist(m_loc+1);
    marker = (int_t *) intCalloc_dist(n);

    /* Get counts of each row of GA */
    for (i = 0; i < n; ++i)
      for (j = colptr[i]; j < colptr[i+1]; ++j) ++marker[rowind[j]];
    /* Set up row pointers */
    rowptr[0] = 0;
    fst_row = iam * m_loc_fst;
    nnz_loc = 0;
    for (j = 0; j < m_loc; ++j) {
      row = fst_row + j;
      rowptr[j+1] = rowptr[j] + marker[row];
      marker[j] = rowptr[j];
    }
    nnz_loc = rowptr[m_loc];

    nzval_loc = (float *) floatMalloc_dist(nnz_loc);
    colind = (int_t *) intMalloc_dist(nnz_loc);

    /* Transfer the matrix into the compressed row storage */
    for (i = 0; i < n; ++i) {
      for (j = colptr[i]; j < colptr[i+1]; ++j) {
	row = rowind[j];
	if ( (row>=fst_row) && (row<fst_row+m_loc) ) {
	  row = row - fst_row;
	  relpos = marker[row];
	  colind[relpos] = i;
	  nzval_loc[relpos] = nzval[j];
	  ++marker[row];
	}
      }
    }

#if ( DEBUGlevel>=2 )
    if ( !iam ) sPrint_CompCol_Matrix_dist(&GA);
#endif   

    /* Destroy GA */
    Destroy_CompCol_Matrix_dist(&GA);

    /******************************************************/
    /* Change GA to a local A with NR_loc format */
    /******************************************************/

    /* Set up the local A in NR_loc format */
    sCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				   nzval_loc, colind, rowptr,
				   SLU_NR_loc, SLU_S, SLU_GE);
    
    /* Get the local B */
    if ( !((*rhs) = floatMalloc_dist(m_loc*nrhs)) )
        ABORT("Malloc fails for rhs[]");
    for (j =0; j < nrhs; ++j) {
	for (i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    (*rhs)[j*m_loc+i] = b_global[j*n+row];
	}
    }
    *ldb = m_loc;

    /* Set the true X */    
    *ldx = m_loc;
    if ( !((*x) = floatMalloc_dist(*ldx * nrhs)) )
        ABORT("Malloc fails for x_loc[]");

    /* Get the local part of xtrue_global */
    for (j = 0; j < nrhs; ++j) {
      for (i = 0; i < m_loc; ++i)
	(*x)[i + j*(*ldx)] = xtrue_global[i + fst_row + j*n];
    }

    SUPERLU_FREE(b_global);
    SUPERLU_FREE(xtrue_global);
    SUPERLU_FREE(marker);

#if ( DEBUGlevel>=1 )
    printf("sizeof(NRforamt_loc) %lu\n", sizeof(NRformat_loc));
    CHECK_MALLOC(iam, "Exit screate_matrix()");
#endif
    return 0;
}
