/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for PDGSSVX example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PDDRIVE.
 *
 * This example illustrates how to use PDGSSVX with the full
 * (default) options to solve a linear system.
 * 
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pdgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pddrive -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
    double   *b, *xtrue;
    int    m, n;
    int      nprow, npcol,lookahead,colperm;
    int      iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();
    int ii, omp_mpi_level;
	MPI_Comm parent;
	float result[2];
	result[0]=0.0;
	result[1]=0.0;
	
    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */
	lookahead = 10; 
	colperm = 4;

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    //MPI_Init( &argc, &argv );
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level); 
	MPI_Comm_get_parent(&parent);   	
	
	

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
	      case 'p': colperm = atoi(*cpp);
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
    // printf("%s\n", postfix);
	
    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    dcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------*/

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
		options.DiagInv       = NO;
     */
    set_default_options_dist(&options);
	options.IterRefine = NOREFINE;
	options.DiagInv = YES;							   
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO; 
    options.ReplaceTinyPivot = YES;
#endif


	options.ColPerm           = colperm;
	options.num_lookaheads           = lookahead;


    if (!iam) {
	//print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    m = A.nrow;
    n = A.ncol;

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    /* Call the linear equation solver. */
    pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);


    /* Check the accuracy of the solution. */
    pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
		     nrhs, b, ldb, xtrue, ldx, grid.comm);

    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */

	
	
	
	/* sending the results (numerical factorization time and total memory) to the parent process */ 
	    float total;
		superlu_dist_mem_usage_t num_mem_usage;
	    dQuerySpace_dist(n, &LUstruct, &grid, &stat, &num_mem_usage);
	    MPI_Allreduce( &num_mem_usage.total, &total,
		       1, MPI_FLOAT, MPI_SUM, grid.comm );
			   
		result[0] = stat.utime[FACT];   
		result[1] = total * 1e-6;     
		if (!iam) {
			printf("returning data:\n"
		   "    Factor time :        %8.2f\n    Total MEM : %8.2f\n",
		   stat.utime[FACT], total * 1e-6);
			printf("**************************************************\n");
			fflush(stdout);
		}	
	
	
	
		//MPI_Bcast(result,2,MPI_FLOAT,0,parent);
		
    		//if (!iam) {
                 //   MPI_Send( result, 2, MPI_FLOAT, 0,
                  //           2, &parent);
    	//	}

	
	
    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&ScalePermstruct);
    dDestroy_LU(n, &grid, &LUstruct);
    dLUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        dSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(berr);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
if(parent!=MPI_COMM_NULL)
	MPI_Reduce(result, MPI_BOTTOM, 2, MPI_FLOAT,MPI_MAX, 0, parent);
    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
	   
	if(parent!=MPI_COMM_NULL)
	MPI_Comm_disconnect(&parent);
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
