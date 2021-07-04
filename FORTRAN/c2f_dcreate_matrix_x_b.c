

/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief Read the matrix from data file, then distribute it in a
 * distributed CSR format.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 * Last update: December 31, 2020
 * </pre>
 */
#include <math.h>
#include "superlu_ddefs.h"

/* \brief
 *
 * <pre>
 * Purpose
 * =======
 * 
 * C2F_DCREATE_MATRIX_X_B read the matrix from data file in various formats,
 * and distribute it to processors in a distributed compressed row format.
 * It also generate the distributed true solution X and the right-hand
 * side RHS.
 *
 * Arguments   
 * =========      
 *
 * fname (input) char*
 *       File name as a character string.
 *
 * nrhs  (input) int
 *       Number of right-hand sides.
 *
 * nprocs (input) int*
 *       Total number of MPI processes.
 *
 * slucomm (input) MPI_Comm
 *       SuperLU's communicator
 *
 * A     (output) SuperMatrix*
 *       Local matrix A in NR_loc format.
 *
 * m_g   (output) int*
 *       Global matrix row dimension
 *
 * n_g   (output) int*
 *       Global matrix column dimension
 *
 * nnz_g (output) int_t*
 *       Number of nonzeros in global matrix
 *
 * rhs   (output) double*
 *       The right-hand side matrix.
 *
 * ldb   (output) int*
 *       Leading dimension of the right-hand side matrix.
 *
 * x     (output) double*
 *       The true solution matrix.
 *
 * ldx   (output) int*
 *       The leading dimension of the true solution matrix.
 *
 * </pre>
 */

int c2f_dcreate_matrix_x_b(char *fname, int nrhs, int nprocs,
                           MPI_Comm slucomm, SuperMatrix *A,
			   int *m_g, int *n_g, int_t *nnz_g,
			   double *rhs, int *ldb, double *x, int *ldx)
{
    SuperMatrix GA;              /* global A */
    double   *b_global, *xtrue_global;  /* replicated on all processes */
    int_t    *rowind, *colptr;	 /* global */
    double   *nzval;             /* global */
    double   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;	 /* local */
    int_t     *marker;
    int_t  nnz, nnz_loc, m, n;
    int    m_loc, fst_row; 
    int    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    int      row, col, i, j, relpos;
    int      iam;
    char     trans[1];

    char     **cpp, c, *postfix;;
    FILE     *fp, *fopen();
    
    MPI_Comm_rank(slucomm, &iam);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter c2f_dreate_matrix_x_b()");
#endif

    if ( iam==0 ) {
	double t = SuperLU_timer_(); 

	if ( !(fp = fopen(fname, "r")) ) {
	    ABORT("File does not exist");
	}
	for (i = 0; i < strlen(fname); i++) {
	    if (fname[i]=='.') {
		postfix = &(fname[i+1]);
	    }
	}
	if(!strcmp(postfix,"rua")){
		/* Read the matrix stored on disk in Harwell-Boeing format. */
		dreadhb_dist(iam, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"mtx")){
		/* Read the matrix stored on disk in Matrix Market format. */
		dreadMM_dist(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"rb")){
		/* Read the matrix stored on disk in Rutherford-Boeing format. */
		dreadrb_dist(iam, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);		
	}else if(!strcmp(postfix,"dat")){
		/* Read the matrix stored on disk in triplet format. */
		dreadtriple_dist(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
	}else if(!strcmp(postfix,"datnh")){
		/* Read the matrix stored on disk in triplet format (without header). */
		dreadtriple_noheader(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);		
	}else if(!strcmp(postfix,"bin")){
		/* Read the matrix stored on disk in binary format. */
		dread_binary(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);		
	}else {
		ABORT("File format not known");
	}

        fclose(fp);
	printf("Time to read and distribute matrix %.2f\n", 
	        SuperLU_timer_() - t);  fflush(stdout);
			
	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,     1,   mpi_int_t,  0, slucomm );
	MPI_Bcast( &n,     1,   mpi_int_t,  0, slucomm );
	MPI_Bcast( &nnz,   1,   mpi_int_t,  0, slucomm );
	MPI_Bcast( nzval,  nnz, MPI_DOUBLE, 0, slucomm );
	MPI_Bcast( rowind, nnz, mpi_int_t,  0, slucomm );
	MPI_Bcast( colptr, n+1, mpi_int_t,  0, slucomm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, slucomm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, slucomm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, slucomm );

	/* Allocate storage for compressed column representation. */
	dallocateA_dist(n, nnz, &nzval, &rowind, &colptr);

	MPI_Bcast( nzval,   nnz, MPI_DOUBLE, 0, slucomm );
	MPI_Bcast( rowind,  nnz, mpi_int_t,  0, slucomm );
	MPI_Bcast( colptr,  n+1, mpi_int_t,  0, slucomm );
    }

#if 0
    nzval[0]=0.1;
#endif

    /* Compute the number of rows to be distributed to local process */
    m_loc = m / nprocs;         //(grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * nprocs) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (nprocs - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * (nprocs - 1);
    }

    /* Create compressed column matrix for GA. */
    dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval, rowind, colptr,
				SLU_NC, SLU_D, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    if ( !(b_global = doubleMalloc_dist(m*nrhs)) )
        ABORT("Malloc fails for b[]");
    if ( !(xtrue_global = doubleMalloc_dist(n*nrhs)) )
        ABORT("Malloc fails for xtrue[]");
    *trans = 'N';

    dGenXtrue_dist(n, nrhs, xtrue_global, n);
    dFillRHS_dist(trans, nrhs, xtrue_global, n, &GA, b_global, m);

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

    nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
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
    if ( !iam ) dPrint_CompCol_Matrix_dist(&GA);
#endif   

    /* Destroy GA */
    Destroy_CompCol_Matrix_dist(&GA);

    /******************************************************/
    /* Change GA to a local A with NR_loc format */
    /******************************************************/

    /* Set up the local A in NR_loc format */
    dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				   nzval_loc, colind, rowptr,
				   SLU_NR_loc, SLU_D, SLU_GE);
    
    /* Get the local B */
    for (j =0; j < nrhs; ++j) {
	for (i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    rhs[j*m_loc+i] = b_global[j*n+row];
	}
    }
    *ldb = m_loc;
    *ldx = m_loc;

    /* Set the true X */    
    /* Get the local part of xtrue_global */
    for (j = 0; j < nrhs; ++j) {
      for (i = 0; i < m_loc; ++i)
	x[i + j*(*ldx)] = xtrue_global[i + fst_row + j*n];
    }

    SUPERLU_FREE(b_global);
    SUPERLU_FREE(xtrue_global);
    SUPERLU_FREE(marker);

#if ( DEBUGlevel>=1 )
    printf("sizeof(NRforamt_loc) %lu\n", sizeof(NRformat_loc));
    CHECK_MALLOC(iam, "Exit c2f_dreate_matrix_x_b()");
#endif

    *m_g = m;
    *n_g = n;
    *nnz_g = nnz;
    return 0;
}
