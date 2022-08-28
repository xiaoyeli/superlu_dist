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
 * -- Distributed SuperLU routine (version 8.1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * AUgust 27, 2022  Add batch option
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
    int      nprow, npcol, lookahead, colperm, rowperm, ir, symbfact, batch;
    int      iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();
    int ii, omp_mpi_level;
    int ldumap, myrank, p; /* The following variables are used for batch solves */
    int*    usermap;
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */
    lookahead = -1;
    colperm = -1;
    rowperm = -1;
    ir = -1;
    symbfact = -1;
    batch = 0;

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
	      case 's': symbfact = atoi(*cpp);
		        break;
              case 'i': ir = atoi(*cpp);
                        break;
              case 'b': batch = atoi(*cpp);
                        break;
	    }
	} else { /* Last arg is considered a filename */
	    if ( !(fp = fopen(*cpp, "r")) ) {
                ABORT("File does not exist");
            }
	    break;
	}
    }

    if ( batch ) { /* in the batch mode: create multiple SuperLU grids,
		      each grid solving one linear system. */
	/* ------------------------------------------------------------
	   INITIALIZE MULTIPLE SUPERLU PROCESS GRIDS. 
	   ------------------------------------------------------------*/
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        usermap = SUPERLU_MALLOC(nprow*npcol * sizeof(int));
        ldumap = nprow;
        int color = myrank/(nprow*npcol); /* Assuming each grid uses the same number of nprow and npcol */
	MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &SubComm);
        p = 0;    
        for (int i = 0; i < nprow; ++i)
    	    for (int j = 0; j < npcol; ++j) usermap[i+j*ldumap] = p++;
        superlu_gridmap(SubComm, nprow, npcol, usermap, ldumap, &grid);
        SUPERLU_FREE(usermap);

#ifdef GPU_ACC
        /* Binding each MPI to a GPU device */
        char *ttemp;
        ttemp = getenv ("SUPERLU_BIND_MPI_GPU");

        if (ttemp) {
	    int devs, rank;
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // MPI_COMM_WORLD needs to be used here instead of SubComm
	    gpuGetDeviceCount(&devs);  // Returns the number of compute-capable devices
	    gpuSetDevice(rank % devs); // Set device to be used for GPU executions
        }

        // This is to initialize GPU, which can be costly. 
        double t1 = SuperLU_timer_();                       
        gpuFree(0);
        double t2 = SuperLU_timer_();    
        if(!myrank)printf("first gpufree time: %7.4f\n",t2-t1);
        gpublasHandle_t hb;           
        gpublasCreate(&hb);
        if(!myrank)printf("first blas create time: %7.4f\n",SuperLU_timer_()-t2);
        gpublasDestroy(hb);
#endif
        // printf("grid.iam %5d, myrank %5d\n",grid.iam,myrank);
        // fflush(stdout);

    } else {
        /* ------------------------------------------------------------
           INITIALIZE THE SUPERLU PROCESS GRID.
           ------------------------------------------------------------ */
        superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);
	
#ifdef GPU_ACC
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        double t1 = SuperLU_timer_();                       
        gpuFree(0);
        double t2 = SuperLU_timer_();    
        if(!myrank)printf("first gpufree time: %7.4f\n",t2-t1);
        gpublasHandle_t hb;           
        gpublasCreate(&hb);
        if(!myrank)printf("first blas create time: %7.4f\n",SuperLU_timer_()-t2);
        gpublasDestroy(hb);
#endif
    }
    
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
    if ( (iam >= nprow * npcol) || (iam == -1) ) goto out;
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
        options.IterRefine        = SLU_DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES;
	options.DiagInv           = NO;
     */
    set_default_options_dist(&options);
#if 0
    options.RowPerm = LargeDiag_HWPM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO; 
    options.ReplaceTinyPivot = YES;
#endif

    if (rowperm != -1) options.RowPerm = rowperm;
    if (colperm != -1) options.ColPerm = colperm;
    if (lookahead != -1) options.num_lookaheads = lookahead;
    if (ir != -1) options.IterRefine = ir;
    if (symbfact != -1) options.ParSymbFact = symbfact;

    if (!iam) {
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

    if ( info ) {  /* Something is wrong */
        if ( iam==0 ) {
	    printf("ERROR: INFO = %d returned from pdgssvx()\n", info);
	    fflush(stdout);
	}
    } else {
        /* Check the accuracy of the solution. */
        pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
		         nrhs, b, ldb, xtrue, ldx, grid.comm);
    }

    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&ScalePermstruct);
    dDestroy_LU(n, &grid, &LUstruct);
    dLUstructFree(&LUstruct);
    dSolveFinalize(&options, &SOLVEstruct);
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(berr);
    fclose(fp);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    if ( batch ) {
        result_min[0] = stat.utime[FACT];   
        result_min[1] = stat.utime[SOLVE];  
        result_max[0] = stat.utime[FACT];   
        result_max[1] = stat.utime[SOLVE];    
        MPI_Allreduce(MPI_IN_PLACE, result_min, 2, MPI_FLOAT,MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, result_max, 2, MPI_FLOAT,MPI_MAX, MPI_COMM_WORLD);
        if (!myrank) {
            printf("Batch solves returning data:\n");
            printf("    Factor time over all grids.  Min: %8.4f Max: %8.4f\n",result_min[0], result_max[0]);
                        printf("    Solve time over all grids.  Min: %8.4f Max: %8.4f\n",result_min[1], result_max[1]);
            printf("**************************************************\n");
            fflush(stdout);
        }
    }
    
    superlu_gridexit(&grid);
    if ( iam != -1 ) PStatFree(&stat);
    
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
