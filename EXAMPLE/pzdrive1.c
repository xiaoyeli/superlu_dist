/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief Driver program for PZGSSVX example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 * April 5, 2015
 * January 4 2020
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
 * The driver program PZDRIVE1.
 *
 * This example illustrates how to use PZGSSVX to
 * solve systems with the same A but different right-hand side,
 * possibly with different number of right-hand sides.
 * In this case, we factorize A only once in the first call to
 * PZGSSVX, and reuse the following data structures
 * in the subsequent call to PZGSSVX:
 *        ScalePermstruct  : DiagScale, R, C, perm_r, perm_c
 *        LUstruct         : Glu_persist, Llu
 * 
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pzdrive1 -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */
int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
    doublecomplex   *b, *xtrue, *b1, *b2;
    int    i, j, m, n, m_loc;
    int    nprow, npcol;
    int    iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;
    int ii, omp_mpi_level;
    FILE *fp, *fopen();
    int cpp_defs();

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs  = 3;  /* Max. number of right-hand sides. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level); 

    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default %d)\n", nprow);
		  printf("\t-c <int>: process columns (default %d)\n", npcol);
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
    if ( iam == -1 )	goto out;
    if ( !iam ) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid:\t\t%d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    for(ii = 0;ii<strlen(*cpp);ii++){
	if((*cpp)[ii]=='.'){
		postfix = &((*cpp)[ii+1]);
	}
    }	
    // printf("%s\n", postfix);

    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    zcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);
    if ( !(b1 = doublecomplexMalloc_dist(ldb * nrhs)) )
        ABORT("Malloc fails for b1[]");
    if ( !(b2 = doublecomplexMalloc_dist(ldb * nrhs)) )
        ABORT("Malloc fails for b1[]");
    for (j = 0; j < nrhs; ++j) {
        for (i = 0; i < ldb; ++i) {
	    b1[i+j*ldb] = b[i+j*ldb];
	    b2[i+j*ldb] = b[i+j*ldb];
        }
    }	    

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    m = A.nrow;
    n = A.ncol;
    m_loc = ((NRformat_loc *)A.Store)->m_loc;
    
    /* ------------------------------------------------------------
       1. SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME, WITH 1 RHS.
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = METIS_AT_PLUS_A;
        options.RowPerm = LargeDiag_MC64;
        options.ReplaceTinyPivot = NO;
        options.Trans = NOTRANS;
        options.IterRefine = SLU_DOUBLE;
        options.SolveInitialized = NO;
        options.RefineInitialized = NO;
        options.PrintStat = YES;
     */
    set_default_options_dist(&options);

    if (!iam) {
	print_options_dist(&options);
	fflush(stdout);
    }

    /* Initialize ScalePermstruct and LUstruct. */
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    /* Call the linear equation solver. */
    nrhs = 1;
    pzgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if ( info ) {  /* Something is wrong */
        if ( iam==0 ) {
	    printf("ERROR: INFO = %d returned from pzgssvx()\n", info);
	    fflush(stdout);
	}
    } else {
        /* Check the accuracy of the solution. */
        if ( !iam ) printf("\tSolve the first system:\n");
        pzinf_norm_error(iam, m_loc, nrhs, b, ldb, xtrue, ldx, grid.comm);
    }
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    PStatFree(&stat);

    /* ------------------------------------------------------------
       2. NOW SOLVE ANOTHER SYSTEM WITH THE SAME A BUT DIFFERENT
       RIGHT-HAND SIDE,  WE WILL USE THE EXISTING L AND U FACTORS IN
       LUSTRUCT OBTAINED FROM A PREVIOUS FATORIZATION.
       ------------------------------------------------------------*/
    options.Fact = FACTORED; /* Indicate the factored form of A is supplied. */
    PStatInit(&stat); /* Initialize the statistics variables. */

    nrhs = 1;
    pzgssvx(&options, &A, &ScalePermstruct, b1, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if ( info ) {  /* Something is wrong */
        if ( iam==0 ) {
	    printf("ERROR: INFO = %d returned from pzgssvx()\n", info);
	    fflush(stdout);
	}
    } else {    
        /* Check the accuracy of the solution. */
        if ( !iam ) printf("\tSolve the system with a different B:\n");
        pzinf_norm_error(iam, m_loc, nrhs, b1, ldb, xtrue, ldx, grid.comm);
    }
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    PStatFree(&stat);

    /* ------------------------------------------------------------
       3. SOLVE ANOTHER SYSTEM WITH THE SAME A BUT DIFFERENT
       NUMBER OF RIGHT-HAND SIDES,  WE WILL USE THE EXISTING L AND U
       FACTORS IN LUSTRUCT OBTAINED FROM A PREVIOUS FATORIZATION.
       ------------------------------------------------------------*/
    options.Fact = FACTORED; /* Indicate the factored form of A is supplied. */
    PStatInit(&stat); /* Initialize the statistics variables. */

    nrhs = 3;
    
    /* When changing the number of RHS's, the following counters 
       for communication messages must be reset. */
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct.gstrs_comm;
    SUPERLU_FREE(gstrs_comm->B_to_X_SendCnt);
    SUPERLU_FREE(gstrs_comm->X_to_B_SendCnt);
    SUPERLU_FREE(gstrs_comm->ptr_to_ibuf);
    pzgstrs_init(n, m_loc, nrhs, ((NRformat_loc *)A.Store)->fst_row,
		 ScalePermstruct.perm_r, ScalePermstruct.perm_c, &grid,
		 LUstruct.Glu_persist, &SOLVEstruct);
    
    pzgssvx(&options, &A, &ScalePermstruct, b2, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if ( info ) {  /* Something is wrong */
        if ( iam==0 ) {
	    printf("ERROR: INFO = %d returned from pzgssvx()\n", info);
	    fflush(stdout);
	}
    } else {
        /* Check the accuracy of the solution. */
        if ( !iam ) printf("\tSolve the system with 3 RHS's:\n");
        pzinf_norm_error(iam, m_loc, nrhs, b2, ldb, xtrue, ldx, grid.comm);
    }
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    PStatFree(&stat);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    Destroy_CompRowLoc_Matrix_dist(&A);
    zScalePermstructFree(&ScalePermstruct);   
    zDestroy_LU(n, &grid, &LUstruct);
    zLUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        zSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(b);
    SUPERLU_FREE(b1);
    SUPERLU_FREE(b2);
    SUPERLU_FREE(xtrue);
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
