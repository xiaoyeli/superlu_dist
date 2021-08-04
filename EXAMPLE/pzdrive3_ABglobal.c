/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief Driver program for pzgssvx_ABglobal example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * April 5, 2015
 * </pre>
 */

#include <math.h>
#include "superlu_zdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program pzdrive3A_ABglobal.
 *
 * This example illustrates how to use pzgssvx_ABglobal to solve
 * systems repeatedly with the same sparsity pattern and similar
 * numerical values of matrix A.
 * In this case, the column permutation vector and symbolic factorization are
 * computed only once. The following data structures will be reused in the
 * subsequent call to pzgssvx_ABglobal:
 *        ScalePermstruct : DiagScale, R, C, perm_r, perm_c
 *        LUstruct        : etree, Glu_persist, Llu
 *
 * NOTE:
 * The distributed nonzero structures of L and U remain the same,
 * although the numerical values are different. So 'Llu' is set up once
 * in the first call to pzgssvx_ABglobal, and reused in the subsequent call.
 *
 * On an IBM SP, the program may be run by typing:
 *    poe pzdrive3_ABglobal -r <proc rows> -c <proc columns> <input_matrix>  -procs <p>
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    gridinfo_t grid;
    double   *berr;
    doublecomplex   *a, *a1, *b, *b1, *xtrue;
    int_t    *asub, *asub1, *xa, *xa1;
    int_t    i, j, m, n, nnz;
    int_t    nprow, npcol;
    int      iam, info, ldb, ldx, nrhs;
    char     trans[1];
    char     **cpp, c;
    FILE *fp, *fopen();
    extern int cpp_defs();

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init( &argc, &argv );

    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default " IFMT ")\n", nprow);
		  printf("\t-c <int>: process columns (default " IFMT ")\n", npcol);
		  exit(0);
		  break;
	      case 'r': nprow = atoi(*cpp);
		        break;
	      case 'c': npcol = atoi(*cpp);
		        break;
	    }
	} else { /* Last arg is considered a filename */
	    if ( !(fp = fopen(*cpp, "r")) ) {
                ABORT("File does not exist");
            }
	    break;
	}
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID. 
       ------------------------------------------------------------*/
    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);

    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( iam == -1 ) goto out;
    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       PROCESS 0 READS THE MATRIX A, AND THEN BROADCASTS IT TO ALL
       THE OTHER PROCESSES.
       ------------------------------------------------------------*/
    if ( !iam ) {
	/* Print the CPP definitions. */
	cpp_defs();
	
	/* Read the matrix stored on disk in Harwell-Boeing format. */
	zreadhb_dist(iam, fp, &m, &n, &nnz, &a, &asub, &xa);
	
	printf("Input matrix file: %s\n", *cpp);
	printf("\tDimension\t" IFMT "x" IFMT "\t # nonzeros " IFMT "\n", m, n, nnz);
	printf("\tProcess grid\t%d X %d\n", (int) grid.nprow, (int) grid.npcol);

	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( a,    nnz, SuperLU_MPI_DOUBLE_COMPLEX, 0, grid.comm );
	MPI_Bcast( asub, nnz, mpi_int_t,  0, grid.comm );
	MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid.comm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid.comm );

	/* Allocate storage for compressed column representation. */
	zallocateA_dist(n, nnz, &a, &asub, &xa);

	MPI_Bcast( a,    nnz, SuperLU_MPI_DOUBLE_COMPLEX, 0, grid.comm );
	MPI_Bcast( asub, nnz, mpi_int_t,  0, grid.comm );
	MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid.comm );
    }
	
    /* Create compressed column matrix for A. */
    zCreate_CompCol_Matrix_dist(&A, m, n, nnz, a, asub, xa,
				SLU_NC, SLU_Z, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    if (!(b=doublecomplexMalloc_dist(m*nrhs))) ABORT("Malloc fails for b[]");
    if (!(xtrue=doublecomplexMalloc_dist(n*nrhs))) ABORT("Malloc fails for xtrue[]");
    *trans = 'N';
    ldx = n;
    ldb = m;
    zGenXtrue_dist(n, nrhs, xtrue, ldx);
    zFillRHS_dist(trans, nrhs, xtrue, ldx, &A, b, ldb);

    /* Save a copy of the right-hand side. */  
    if ( !(b1 = doublecomplexMalloc_dist(m * nrhs)) ) ABORT("Malloc fails for b1[]");
    for (j = 0; j < nrhs; ++j)
	for (i = 0; i < m; ++i) b1[i+j*ldb] = b[i+j*ldb];
    
    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* Save a copy of the matrix A. */
    zallocateA_dist(n, nnz, &a1, &asub1, &xa1);
    for (i = 0; i < nnz; ++i) { a1[i] = a[i]; asub1[i] = asub[i]; }
    for (i = 0; i < n+1; ++i) xa1[i] = xa[i];


    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME.
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = METIS_AT_PLUS_A;
        options.RowPerm = LargeDiag_MC64;
        options.ReplaceTinyPivot = YES;
        options.Trans = NOTRANS;
        options.IterRefine = DOUBLE;
        options.SolveInitialized = NO;
        options.RefineInitialized = NO;
        options.PrintStat = YES;
     */
    set_default_options_dist(&options);

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
    }

    /* Initialize ScalePermstruct and LUstruct. */
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    /* Call the linear equation solver: factorize and solve. */
    pzgssvx_ABglobal(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
		     &LUstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    if ( !iam ) {
	zinf_norm_error_dist(n, nrhs, b, ldb, xtrue, ldx, &grid);
    }
    
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    PStatFree(&stat);
    Destroy_CompCol_Matrix_dist(&A); /* Deallocate storage of matrix A.     */
    SUPERLU_FREE(b);                 /* Free storage of right-hand side.    */


    /* ------------------------------------------------------------
       NOW WE SOLVE ANOTHER LINEAR SYSTEM.
       THE MATRIX A HAS THE SAME SPARSITY PATTERN AND THE SIMILAR
       NUMERICAL VALUES AS THAT IN A PREVIOUS SYSTEM.
       ------------------------------------------------------------*/
    options.Fact = SamePattern_SameRowPerm;
    PStatInit(&stat); /* Initialize the statistics variables. */

    /* Create compressed column matrix for A. */
    zCreate_CompCol_Matrix_dist(&A, m, n, nnz, a1, asub1, xa1,
				SLU_NC, SLU_Z, SLU_GE);

    /* Solve the linear system. */
    pzgssvx_ABglobal(&options, &A, &ScalePermstruct, b1, ldb, nrhs, &grid,
		     &LUstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    if ( !iam ) {
	printf("Solve a system with the same pattern and similar values.\n");
	zinf_norm_error_dist(n, nrhs, b1, ldb, xtrue, ldx, &grid);
    }

    /* Print the statistics. */
    PStatPrint(&options, &stat, &grid);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    PStatFree(&stat);
    Destroy_CompCol_Matrix_dist(&A); /* Deallocate storage of matrix A.     */
    zDestroy_LU(n, &grid, &LUstruct); /* Deallocate storage associated with    
					the L and U matrices.               */
    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);         /* Deallocate the structure of L and U.*/
    SUPERLU_FREE(b1);	             /* Free storage of right-hand side.    */
    SUPERLU_FREE(xtrue);             /* Free storage of the exact solution. */
    SUPERLU_FREE(berr);
    fclose(fp);


    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();


#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

}


int cpp_defs()
{
    printf(".. CPP definitions:\n");
#if ( PRNTlevel>=1 )
    printf("\tPRNTlevel = %d\n", PRNTlevel);
#endif
#if ( DEBUGlevel>=1 )
    printf("\tDEBUGlevel = %d\n", DEBUGlevel);
#endif
#if ( PROFlevel>=1 )
    printf("\tPROFlevel = %d\n", PROFlevel);
#endif
#if ( StaticPivot>=1 )
    printf("\tStaticPivot = %d\n", StaticPivot);
#endif
    printf("....\n");
    return 0;
}
