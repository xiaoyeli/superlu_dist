/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for PSGSSVX example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"
#include "superlu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PSDRIVE.
 *
 * This example illustrates how to use PSGSSVX with the full
 * (default) options to solve a linear system.
 * 
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call psgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> psdrive -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    float   *err_bounds, *berr;
    float   *b, *xtrue;
    int    m, n;
    int      nprow, npcol, lookahead, colperm, rowperm, ir, use_tensorcore;
    int      iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();

    int ii, i, omp_mpi_level;
    extern void psgssvx_d2(superlu_dist_options_t *options, SuperMatrix *A,
			sScalePermstruct_t *ScalePermstruct,
			float B[], int ldb, int nrhs, gridinfo_t *grid,
			sLUstruct_t *LUstruct, sSOLVEstruct_t *SOLVEstruct,
			float *err_bounds, SuperLUStat_t *stat, int *info,
			float *xtrue);

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */
    lookahead = -1;
    colperm = -1;
    rowperm = -1;
    ir = -1;
    use_tensorcore = -1;

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    //MPI_Init( &argc, &argv );
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level); 
	

#if ( VAMPIR>=1 )
    VT_traceoff(); 
#endif

#if ( VTUNE>=1 )
	__itt_pause();
#endif
	
    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default %4d)\n", nprow);
		  printf("\t-c <int>: process columns (default %4d)\n", npcol);
		  exit(0);
		  break;
	      case 'r': nprow = atoi(*cpp);
		        break;
	      case 'c': npcol = atoi(*cpp);
		        break;
	      case 'l': lookahead = atoi(*cpp);
		        break;
	      case 'p': rowperm = atoi(*cpp);
		        break;
	      case 'q': colperm = atoi(*cpp);
		        break;
	      case 'i': ir = atoi(*cpp);
		        break;
	      case 't': use_tensorcore = atoi(*cpp);
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
	
    if(grid.iam==0){
	MPI_Query_thread(&omp_mpi_level);
    switch (omp_mpi_level) {
      case MPI_THREAD_SINGLE:
		printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
		fflush(stdout);
	break;
      case MPI_THREAD_FUNNELED:
		printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_SERIALIZED:
		printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_MULTIPLE:
		printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
		fflush(stdout);
	break;
    }
	}
	
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
    //printf("%s\n", postfix);
	
    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    fp = fopen(*cpp, "r");
    screate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);
    fclose(fp);
    
#if 1
    /* Generate a good RHS in double precision, then rounded to single */
    {
        double *db, *dxtrue, *dberr;
	SuperMatrix dA;
	dScalePermstruct_t dScalePermstruct;
	dLUstruct_t dLUstruct;
	dSOLVEstruct_t dSOLVEstruct;
	//extern int dcreate_matrix_postfix();

	dcreate_matrix_postfix(&dA, nrhs, &db, &ldb, &dxtrue, &ldx, fp, postfix, &grid);
	fclose(fp);

	NRformat_loc *Astore = dA.Store;
	int m_loc = Astore->m_loc;

	/* Rounded to single */
	for (i = 0; i < m_loc; ++i) {
	  b[i] = (float) db[i];
	}

	/* Now, compute dxtrue via double-precision solver */
	/* Why? dA is the double precision version of A, etc.
	   If db = dA*dXtrue to double, then rounding db to b and dA to A introduce
	   a perturbation of eps_single in A and b, so a perturbation of 
	   cond(A)*eps_single in x vs dXtrue. 
	   So need to use a computed Xtrue from a double code: dXtrue <- dA \ db,
	   this computed Xtrue would be comparable to x, i.e., having the same cond(A)
	   factor in the perturbation error.   See bullet 7, page 20, LAWN 165.*/
	if ( !(dberr = doubleMalloc_dist(nrhs)) )
	  ABORT("Malloc fails for dberr[].");

	m = dA.nrow;
	n = dA.ncol;

	set_default_options_dist(&options);
	dScalePermstructInit(m, n, &dScalePermstruct);
	dLUstructInit(n, &dLUstruct);
	PStatInit(&stat);

	pdgssvx(&options, &dA, &dScalePermstruct, db, ldb, nrhs, &grid,
		&dLUstruct, &dSOLVEstruct, dberr, &stat, &info);

	pdinf_norm_error(iam, m_loc, nrhs, db, ldb, dxtrue, ldx, grid.comm);

	/* Rounded to single */
	for (i = 0; i < m_loc; ++i) {
	  xtrue[i] = (float) db[i];
	}

	if ( iam==0 ) { //(nprow*npcol-1) ) {
	  printf("\ndouble generated dxtrue[%d]: %.16e\n", m_loc-1, dxtrue[m_loc-1]);
	  for (i = 0; i < 5; ++i) printf("%.16e\t", dxtrue[i]);
	  printf("\ndouble computed dxtrue (stored in db):\n");
	  for (i = 0; i < 5; ++i) printf("%.16e\t", db[i]);
	  printf("\nsingle xtrue:\n");
	  for (i = 0; i < 5; ++i) printf("%.16e\t", xtrue[i]);
	  printf("\n"); fflush(stdout);
	}

	PStatPrint(&options, &stat, &grid); /* Print the statistics. */

	PStatFree(&stat);
	Destroy_CompRowLoc_Matrix_dist(&dA);
	dScalePermstructFree(&dScalePermstruct);
	dDestroy_LU(n, &grid, &dLUstruct);
	dLUstructFree(&dLUstruct);
	if ( options.SolveInitialized ) {
	  dSolveFinalize(&options, &dSOLVEstruct);
	}
	SUPERLU_FREE(db);
	SUPERLU_FREE(dxtrue);
	SUPERLU_FREE(dberr);
    }
#endif

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM in single precision
       ------------------------------------------------------------*/
    if ( !(err_bounds = floatCalloc_dist(nrhs*3)) )
	ABORT("Malloc fails for err_bounds[].");
    if ( !(berr = floatMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    if ( iam==0 ) {
      printf("\n Now single LU with double ItRef... N = %d\n", (int) A.nrow);
      fflush(stdout);
    }

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES;
	options.DiagInv           = NO;
     */
    set_default_options_dist(&options);

    options.IterRefine = SLU_DOUBLE;
#if 0
    options.IterRefine = SLU_SINGLE;
    options.Equil = NO; 
    options.ReplaceTinyPivot  = YES;
    options.RowPerm = NOROWPERM;
    options.ColPerm = NATURAL;
    options.ReplaceTinyPivot = YES;
#endif

    if (rowperm != -1) options.RowPerm = rowperm;
    if (colperm != -1) options.ColPerm = colperm;
    if (lookahead != -1) options.num_lookaheads = lookahead;
    if (ir != -1) options.IterRefine = ir;
    if (use_tensorcore != -1) options.Use_TensorCore = use_tensorcore;

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    m = A.nrow;
    n = A.ncol;

    /* Initialize ScalePermstruct and LUstruct. */
    sScalePermstructInit(m, n, &ScalePermstruct);
    sLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    if (iam==0) {
        Printfloat5("before psgssvx xtrue[]", 5, xtrue);
        Printfloat5("before psgssvx b[]", 5, b);
	printf("\n"); fflush(stdout);
    }

    if ( options.IterRefine == SLU_DOUBLE || options.IterRefine == SLU_EXTRA ) { 
        /* Call the linear equation solver with extra-precise iterative refinement */
        psgssvx_d2(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
		   &LUstruct, &SOLVEstruct, err_bounds, &stat, &info, xtrue);
    } else {
        /* Call the linear equation solver */
        psgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
		&LUstruct, &SOLVEstruct, berr, &stat, &info);
    }

    /* Check the accuracy of the solution. */
    psinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
		     nrhs, b, ldb, xtrue, ldx, grid.comm);
    if (iam == 0) {
        printf(" Normwise error bound: %e\n", err_bounds[0]);
	printf(" Componentwise error bound: %e\n", err_bounds[1*nrhs]);
	printf(" Componentwise backword error: %e\n", err_bounds[2*nrhs]);
    }

    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A);
    sScalePermstructFree(&ScalePermstruct);
    sDestroy_LU(n, &grid, &LUstruct);
    sLUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        sSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(err_bounds);
    SUPERLU_FREE(berr);

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
