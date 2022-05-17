/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief This example illustrates how to divide up the processes into subgroups
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * April 5, 2015
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program psdrive4_ABglobal.
 *
 * This example illustrates how to divide up the processes into
 * subgroups (multiple grids) such that each subgroup solves a linear
 * system independently from the other.
 *
 * In this example, there are 2 subgroups:
 *  1. subgroup 1 consists of processes 0 to 5 arranged as
 *     a 2-by-3 process grid.
 *  2. subgroup 2 consists of processes 6 to 9 arranged as
 *     a 2-by-2 process grid.
 *
 * On an IBM SP, the program may be run by typing
 *    poe psdrive4_ABglobal <input_file> -procs 10
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    gridinfo_t grid1, grid2;
    float   *berr;
    float   *a, *b, *xtrue;
    int_t    *asub, *xa;
    int_t    i, j, m, n, nnz;
    int_t    nprow, npcol, ldumap, p;
    int    usermap[6];
    int      iam, info, ldb, ldx, nprocs;
    int      nrhs = 1;   /* Number of right-hand side. */
    char     trans[1];
    char     **cpp, c;
    FILE *fp, *fopen();

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    if ( nprocs < 10 ) {
	fprintf(stderr, "Requires at least 10 processes\n");
	exit(-1);
    }

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
       INITIALIZE THE SUPERLU PROCESS GRID 1. 
       ------------------------------------------------------------*/
    nprow = 2;
    npcol = 3;
    ldumap = 2;
    p = 0;    /* Grid 1 starts from process 0. */
    for (i = 0; i < nprow; ++i)
	for (j = 0; j < npcol; ++j) usermap[i+j*ldumap] = p++;
    superlu_gridmap(MPI_COMM_WORLD, nprow, npcol, usermap, ldumap, &grid1);

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID 2. 
       ------------------------------------------------------------*/
    nprow = 2;
    npcol = 2;
    ldumap = 2;
    p = 6;   /* Grid 2 starts from process 6. */
    for (i = 0; i < nprow; ++i)
	for (j = 0; j < npcol; ++j) usermap[i+j*ldumap] = p++;
    superlu_gridmap(MPI_COMM_WORLD, nprow, npcol, usermap, ldumap, &grid2);

    /* Bail out if I do not belong in any of the 2 grids. */
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    if ( iam == -1 ) goto out;
    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    if ( iam >= 0 && iam < 6 ) { /* I am in grid 1. */
	iam = grid1.iam;  /* Get the logical number in the new grid. */

	/* ------------------------------------------------------------
	   PROCESS 0 READS THE MATRIX A, AND THEN BROADCASTS IT TO ALL
	   THE OTHER PROCESSES.
	   ------------------------------------------------------------*/
	if ( !iam ) {
	    /* Read the matrix stored on disk in Harwell-Boeing format. */
	    sreadhb_dist(iam, fp, &m, &n, &nnz, &a, &asub, &xa);
	
	    printf("\tDimension\t" IFMT "x" IFMT "\t # nonzeros " IFMT "\n", m, n, nnz);
	    printf("\tProcess grid\t%d X %d\n", (int) grid1.nprow, (int) grid1.npcol);

	    /* Broadcast matrix A to the other PEs. */
	    MPI_Bcast( &m,   1,   mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( &n,   1,   mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid1.comm );
	    MPI_Bcast( asub, nnz, mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid1.comm );
	} else {
	    /* Receive matrix A from PE 0. */
	    MPI_Bcast( &m,   1,   mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( &n,   1,   mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid1.comm );

	    /* Allocate storage for compressed column representation. */
	    sallocateA_dist(n, nnz, &a, &asub, &xa);
	    
	    MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid1.comm );
	    MPI_Bcast( asub, nnz, mpi_int_t,  0, grid1.comm );
	    MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid1.comm );
	}
	
	/* Create compressed column matrix for A. */
	sCreate_CompCol_Matrix_dist(&A, m, n, nnz, a, asub, xa,
				    SLU_NC, SLU_S, SLU_GE);

	/* Generate the exact solution and compute the right-hand side. */
	if (!(b=floatMalloc_dist(m*nrhs))) ABORT("Malloc fails for b[]");
	if (!(xtrue=floatMalloc_dist(n*nrhs))) ABORT("Malloc fails for xtrue[]");
	*trans = 'N';
	ldx = n;
	ldb = m;
	sGenXtrue_dist(n, nrhs, xtrue, ldx);
	sFillRHS_dist(trans, nrhs, xtrue, ldx, &A, b, ldb);

	if ( !(berr = floatMalloc_dist(nrhs)) )
	    ABORT("Malloc fails for berr[].");

	/* ------------------------------------------------------------
	   NOW WE SOLVE THE LINEAR SYSTEM.
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
	    print_options_dist(&options);
        }

	/* Initialize ScalePermstruct and LUstruct. */
	sScalePermstructInit(m, n, &ScalePermstruct);
	sLUstructInit(n, &LUstruct);

	/* Initialize the statistics variables. */
	PStatInit(&stat);
	
	/* Call the linear equation solver: factorize and solve. */
	psgssvx_ABglobal(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid1,
			 &LUstruct, berr, &stat, &info);

	/* Check the accuracy of the solution. */
	if ( !iam ) {
	    sinf_norm_error_dist(n, nrhs, b, ldb, xtrue, ldx, &grid1);
	}
    
    
	/* Print the statistics. */
	PStatPrint(&options, &stat, &grid1);

	/* ------------------------------------------------------------
	   DEALLOCATE STORAGE.
	   ------------------------------------------------------------*/
	PStatFree(&stat);
	Destroy_CompCol_Matrix_dist(&A); 
	sDestroy_LU(n, &grid1, &LUstruct);
	sScalePermstructFree(&ScalePermstruct);
	sLUstructFree(&LUstruct);
	SUPERLU_FREE(b);
	SUPERLU_FREE(xtrue);
	SUPERLU_FREE(berr);

    } else { /* I am in grid 2. */
	iam = grid2.iam;  /* Get the logical number in the new grid. */

	/* ------------------------------------------------------------
	   PROCESS 0 READS THE MATRIX A, AND THEN BROADCASTS IT TO ALL
	   THE OTHER PROCESSES.
	   ------------------------------------------------------------*/
	if ( !iam ) {
	    /* Read the matrix stored on disk in Harwell-Boeing format. */
	    sreadhb_dist(iam, fp, &m, &n, &nnz, &a, &asub, &xa);
	
	    printf("\tDimension\t" IFMT "x" IFMT "\t # nonzeros " IFMT "\n", m, n, nnz);
	    printf("\tProcess grid\t%d X %d\n", (int) grid2.nprow, (int) grid2.npcol);

	    /* Broadcast matrix A to the other PEs. */
	    MPI_Bcast( &m,   1,   mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( &n,   1,   mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid2.comm );
	    MPI_Bcast( asub, nnz, mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid2.comm );
	} else {
	    /* Receive matrix A from PE 0. */
	    MPI_Bcast( &m,   1,   mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( &n,   1,   mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid2.comm );

	    /* Allocate storage for compressed column representation. */
	    sallocateA_dist(n, nnz, &a, &asub, &xa);
	    
	    MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid2.comm );
	    MPI_Bcast( asub, nnz, mpi_int_t,  0, grid2.comm );
	    MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid2.comm );
	}
	
	/* Create compressed column matrix for A. */
	sCreate_CompCol_Matrix_dist(&A, m, n, nnz, a, asub, xa,
				    SLU_NC, SLU_S, SLU_GE);

	/* Generate the exact solution and compute the right-hand side. */
	if (!(b=floatMalloc_dist(m*nrhs))) ABORT("Malloc fails for b[]");
	if (!(xtrue=floatMalloc_dist(n*nrhs))) ABORT("Malloc fails for xtrue[]");
	*trans = 'N';
	ldx = n;
	ldb = m;
	sGenXtrue_dist(n, nrhs, xtrue, ldx);
	sFillRHS_dist(trans, nrhs, xtrue, ldx, &A, b, ldb);

	if ( !(berr = floatMalloc_dist(nrhs)) )
	    ABORT("Malloc fails for berr[].");

	/* ------------------------------------------------------------
	   NOW WE SOLVE THE LINEAR SYSTEM.
	   ------------------------------------------------------------*/
	
        /* Set the default input options:
            options.Fact = DOFACT;
            options.Equil = YES;
            options.ColPerm = MMD_AT_PLUS_A;
            options.RowPerm = LargeDiag_MC64;
            options.ReplaceTinyPivot = YES;
            options.Trans = NOTRANS;
            options.IterRefine = DOUBLE;
            options.SolveInitialized = NO;
            options.RefineInitialized = NO;
            options.PrintStat = YES;
         */
	set_default_options_dist(&options);
	
	/* Initialize ScalePermstruct and LUstruct. */
	sScalePermstructInit(m, n, &ScalePermstruct);
	sLUstructInit(n, &LUstruct);

	/* Initialize the statistics variables. */
	PStatInit(&stat);
	
	/* Call the linear equation solver: factorize and solve. */
	psgssvx_ABglobal(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid2,
			 &LUstruct, berr, &stat, &info);

	/* Check the accuracy of the solution. */
	if ( !iam ) {
	    sinf_norm_error_dist(n, nrhs, b, ldb, xtrue, ldx, &grid2);
	}
    
    
	/* Print the statistics. */
	PStatPrint(&options, &stat, &grid2);

	/* ------------------------------------------------------------
	   DEALLOCATE STORAGE.
	   ------------------------------------------------------------*/
	PStatFree(&stat);
	Destroy_CompCol_Matrix_dist(&A); 
	sDestroy_LU(n, &grid2, &LUstruct);
	sScalePermstructFree(&ScalePermstruct);
	sLUstructFree(&LUstruct);
	SUPERLU_FREE(b);
	SUPERLU_FREE(xtrue);
	SUPERLU_FREE(berr);
    }

    fclose(fp);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRIDS.
       ------------------------------------------------------------*/
    superlu_gridexit(&grid1);
    superlu_gridexit(&grid2);

out:
    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

}
