/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Driver program for PZGSSVX3D example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology.
 * May 10, 2019
 *
 */
#include "superlu_zdefs.h"  

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PZDRIVE3D.
 *
 * This example illustrates how to use PZGSSVX3D with the full
 * (default) options to solve a linear system.
 *
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pzgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * The program may be run by typing
 *    mpiexec -np <p> pzdrive3d -r <proc rows> -c <proc columns> \
 *                                   -d <proc Z-dimension> <input_file>
 * NOTE: total number of processes p = r * c * d
 *       d must be a power-of-two, e.g., 1, 2, 4, ...
 *
 * </pre>
 */
 
static void matCheck(int n, int m, doublecomplex* A, int LDA,
       doublecomplex* B, int LDB)
{
    for(int j=0; j<m;j++)
        for (int i = 0; i < n; ++i) {
	    assert( (A[i+ LDA*j].r == B[i+ LDB*j].r)
	    	    && (A[i+ LDA*j].i == B[i+ LDB*j].i) );
	}
    printf("B check passed\n");
    return;
}

static void checkNRFMT(NRformat_loc*A, NRformat_loc*B)
{
    /*
    int_t nnz_loc;
    int_t m_loc;
    int_t fst_row;
    void  *nzval;
    int_t *rowptr;
    int_t *colind;
    */

    assert(A->nnz_loc == B->nnz_loc);
    assert(A->m_loc == B->m_loc);
    assert(A->fst_row == B->fst_row);

#if 0
    double *Aval = (double *)A->nzval, *Bval = (double *)B->nzval;
    PrintDouble5("A", A->nnz_loc, Aval);
    PrintDouble5("B", B->nnz_loc, Bval);
    fflush(stdout);
#endif

    doublecomplex * Aval = (doublecomplex *) A->nzval;
    doublecomplex * Bval = (doublecomplex *) B->nzval;
    for (int_t i = 0; i < A->nnz_loc; i++)
    {
        assert( (Aval[i].r == Bval[i].r) && (Aval[i].i == Bval[i].i) );
        assert((A->colind)[i] == (B->colind)[i]);
	printf("colind[] correct\n");
    }

    for (int_t i = 0; i < A->m_loc + 1; i++)
    {
        assert((A->rowptr)[i] == (B->rowptr)[i]);
    }

    printf("Matrix check passed\n");

}

int
main (int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;  // Now, A is on all 3D processes  
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zSOLVEstruct_t SOLVEstruct;
    gridinfo3d_t grid;
    double *berr;
    doublecomplex *b, *xtrue;
    int_t m, n;
    int nprow, npcol, npdep;
    int iam, info, ldb, ldx, nrhs;
    char **cpp, c, *suffix;
    FILE *fp, *fopen ();
    extern int cpp_defs ();
    int ii, omp_mpi_level;

    nprow = 1;            /* Default process rows.      */
    npcol = 1;            /* Default process columns.   */
    npdep = 1;            /* replication factor must be power of two */
    nrhs = 1;             /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------ */
    // MPI_Init (&argc, &argv);
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!rank) printf("The MPI library doesn't provide MPI_THREAD_MULTIPLE \n");
    }

    /* Parse command line argv[]. */
    for (cpp = argv + 1; *cpp; ++cpp)
    {
        if (**cpp == '-')
        {
            c = *(*cpp + 1);
            ++cpp;
            switch (c)
            {
            case 'h':
                printf ("Options:\n");
                printf ("\t-r <int>: process rows    (default %d)\n", nprow);
                printf ("\t-c <int>: process columns (default %d)\n", npcol);
                printf ("\t-d <int>: process Z-dimension (default %d)\n", npdep);
                exit (0);
                break;
            case 'r':
                nprow = atoi (*cpp);
                break;
            case 'c':
                npcol = atoi (*cpp);
                break;
            case 'd':
                npdep = atoi (*cpp);
                break;
            }
        }
        else
        {   /* Last arg is considered a filename */
            if (!(fp = fopen (*cpp, "r")))
            {
                ABORT ("File does not exist");
            }
            break;
        }
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
    superlu_gridinit3d (MPI_COMM_WORLD, nprow, npcol, npdep, &grid);

    if(grid.iam==0) {
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
    if (iam == -1)     goto out;
    if (!iam) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
	printf("3D process grid: %d X %d X %d\n", nprow, npcol, npdep);
	//printf("2D Process grid: %d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE.
       ------------------------------------------------------------ */
    for (ii = 0; ii<strlen(*cpp); ii++) {
	if((*cpp)[ii]=='.'){
	    suffix = &((*cpp)[ii+1]);
	    // printf("%s\n", suffix);
	}
    }

#define NRFRMT
#ifndef NRFRMT
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
	zcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, suffix, &(grid.grid2d));
	
#else
    // *fp0 = *fp;
    zcreate_matrix_postfix3d(&A, nrhs, &b, &ldb,
                             &xtrue, &ldx, fp, suffix, &(grid));
    //printf("ldx %d, ldb %d\n", ldx, ldb);
    
#if 0  // following code is only for checking *Gather* routine
    NRformat_loc *Astore, *Astore0;
    doublecomplex* B2d;
    NRformat_loc Atmp = dGatherNRformat_loc(
                            (NRformat_loc *) A.Store,
                            b, ldb, nrhs, &B2d,
                            &grid);
    Astore = &Atmp;
    SuperMatrix Aref;
    doublecomplex *bref, *xtrueref;
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
    {
        zcreate_matrix_postfix(&Aref, nrhs, &bref, &ldb,
                               &xtrueref, &ldx, fp0, 
                               suffix, &(grid.grid2d));
        Astore0 = (NRformat_loc *) Aref.Store;

	/*
	if ( (grid.grid2d).iam == 0 ) {
	    printf(" iam %d\n", 0); 
	    checkNRFMT(Astore, Astore0);
	} else if ((grid.grid2d).iam == 1 ) {
	    printf(" iam %d\n", 1); 
	    checkNRFMT(Astore, Astore0);
	} 
	*/
    
	// bref, xtrueref are created on 2D
        matCheck(Astore->m_loc, nrhs, B2d, Astore->m_loc, bref, ldb);
    }
    // MPI_Finalize(); exit(0);
    #endif
#endif

    if (!(berr = doubleMalloc_dist (nrhs)))
        ABORT ("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------ */

    /* Set the default input options:
       options.Fact              = DOFACT;
       options.Equil             = YES;
       options.ParSymbFact       = NO;
       options.ColPerm           = METIS_AT_PLUS_A;
       options.RowPerm           = LargeDiag_MC64;
       options.ReplaceTinyPivot  = YES;
       options.IterRefine        = DOUBLE;
       options.Trans             = NOTRANS;
       options.SolveInitialized  = NO;
       options.RefineInitialized = NO;
       options.PrintStat         = YES;
       options->num_lookaheads    = 10;
       options->lookahead_etree   = NO;
       options->SymPattern        = NO;
       options.DiagInv           = NO;
     */
    set_default_options_dist (&options);
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = NO;
#endif

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

#ifdef NRFRMT  // matrix is on 3D process grid
    m = A.nrow;
    n = A.ncol;
#else
    if ( grid.zscp.Iam == 0 )  // Process layer 0
    {
	m = A.nrow;
        n = A.ncol;
    }
    // broadcast m, n to all the process layers;
    MPI_Bcast( &m, 1, mpi_int_t, 0,  grid.zscp.comm);
    MPI_Bcast( &n, 1, mpi_int_t, 0,  grid.zscp.comm);
#endif    

    /* Initialize ScalePermstruct and LUstruct. */
    zScalePermstructInit (m, n, &ScalePermstruct);
    zLUstructInit (n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit (&stat);

    /* Call the linear equation solver. */
    pzgssvx3d (&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    pzinf_norm_error (iam, ((NRformat_loc *) A.Store)->m_loc,
                          nrhs, b, ldb, xtrue, ldx, grid.comm);
    fflush(stdout);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------ */

    if ( grid.zscp.Iam == 0 ) { // process layer 0

	PStatPrint (&options, &stat, &(grid.grid2d)); /* Print 2D statistics.*/

        zDestroy_LU (n, &(grid.grid2d), &LUstruct);
        if (options.SolveInitialized) {
            zSolveFinalize (&options, &SOLVEstruct);
        }
    } else { // Process layers not equal 0
        zDeAllocLlu_3d(n, &LUstruct, &grid);
        zDeAllocGlu_3d(&LUstruct);
    }

    Destroy_CompRowLoc_Matrix_dist (&A);
    SUPERLU_FREE (b);
    SUPERLU_FREE (xtrue);
    SUPERLU_FREE (berr);
    zScalePermstructFree (&ScalePermstruct);
    zLUstructFree (&LUstruct);
    PStatFree (&stat);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
out:
    superlu_gridexit3d (&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------ */
    MPI_Finalize ();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit main()");
#endif

}


int
cpp_defs ()
{
    printf (".. CPP definitions:\n");
#if ( PRNTlevel>=1 )
    printf ("\tPRNTlevel = %d\n", PRNTlevel);
#endif
#if ( DEBUGlevel>=1 )
    printf ("\tDEBUGlevel = %d\n", DEBUGlevel);
#endif
#if ( PROFlevel>=1 )
    printf ("\tPROFlevel = %d\n", PROFlevel);
#endif
    printf ("....\n");
    return 0;
}
