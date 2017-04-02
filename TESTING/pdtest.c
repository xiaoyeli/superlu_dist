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
 * -- Distributed SuperLU routine (version 5.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 16, 2017
 * </pre>
 */
/*
 * File name:		pdtest.c
 * Purpose:             MAIN test program
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include "superlu_ddefs.h"

#define NTESTS    5      /* Number of test types */
#define NTYPES    11     /* Number of matrix types */
#define NTRAN     2    
#define THRESH    20.0
#define FMT1      "%10s:n=%d, test(%d)=%12.5g\n"
#define	FMT2      "%10s:fact=%4d, trans=%4d, equed=%c, n=%d, imat=%d, test(%d)=%12.5g\n"
#define FMT3      "%10s:info=%d, izero=%d, n=%d, nrhs=%d, imat=%d, nfail=%d\n"


static void
parse_command_line(int argc, char *argv[], int *nprow, int *npcol,
		   char *matrix_type, int *n, int *relax, int *maxsuper,
		   int *fill_ratio, int *min_gemm_gpu_offload,
		   int *nrhs, FILE **fp);

extern int
pdcompute_resid(int m, int n, int nrhs, SuperMatrix *A,
		double *x, int ldx, double *b, int ldb,
		gridinfo_t *grid, SOLVEstruct_t *SOLVEstruct, double *resid);

/*! \brief Copy matrix A into matrix B, in distributed compressed row format. */
void
dCopy_CompRowLoc_Matrix_dist(SuperMatrix *A, SuperMatrix *B)
{
    NRformat_loc *Astore;
    NRformat_loc *Bstore;
    int_t i, nnz_loc, m_loc;

    B->Stype = A->Stype;
    B->Dtype = A->Dtype;
    B->Mtype = A->Mtype;
    B->nrow = A->nrow;;
    B->ncol = A->ncol;
    Astore = (NRformat_loc *) A->Store;
    Bstore = (NRformat_loc *) B->Store;
    Bstore->nnz_loc = Astore->nnz_loc;
    nnz_loc = Astore->nnz_loc;
    Bstore->m_loc = Astore->m_loc;
    m_loc = Astore->m_loc;
    Bstore->fst_row = Astore->fst_row;
    memcpy(Bstore->nzval, Astore->nzval, nnz_loc * sizeof(double));
    memcpy(Bstore->colind, Astore->colind, nnz_loc * sizeof(int_t));
    memcpy(Bstore->rowptr, Astore->rowptr, (m_loc+1) * sizeof(int_t));
}

int main(int argc, char *argv[])
{
/*
 * <pre>
 * Purpose
 * =======
 *
 * PDTEST is the main test program for the DOUBLE linear 
 * equation driver routines PDGSSVX.
 * 
 * The program is invoked by a shell script file -- dtest.csh.
 * The output from the tests are written into a file -- dtest.out.
 */
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A, Asave;
    NRformat_loc *Astore;
    ScalePermstruct_t ScalePermstruct;
    LUstruct_t LUstruct;
    SOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *nzval_save;
    int_t    *colind_save, *rowptr_save;
    double   *berr;
    double   *b, *bsave, *xtrue, *solx;
    int    i, j, m, n;
    int    nprow, npcol;
    int    iam, info, ldb, ldx, nrhs;
    char     **cpp, c;
    FILE *fp, *fopen();
    char matrix_type[8], equed[1];
    int  relax, maxsuper=0, fill_ratio=0, min_gemm_gpu_offload=0;
    int    equil, ifact, nfact, iequil, iequed, prefact, notfactored;
    int    nt, nrun, nfail, nerrs, imat, fimat, nimat=1;
    fact_t fact;
    double rowcnd, colcnd, amax;
    double result[NTESTS];

    /* Fixed set of parameters */
    int     iseed[]  = {1988, 1989, 1990, 1991};
    char    equeds[]  = {'N', 'R', 'C', 'B'};
    fact_t  facts[] = {FACTORED, DOFACT, SamePattern, SamePattern_SameRowPerm};
    trans_t transs[]  = {NOTRANS, TRANS, CONJ};

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init( &argc, &argv );

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID. 
       ------------------------------------------------------------*/
    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);

    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( iam >= nprow * npcol )	goto out;
    if ( !iam ) {
        printf("\tProcess grid\t%d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* Parse command line argv[]. */
    parse_command_line(argc, argv, &nprow, &npcol, matrix_type, &n,
		       &relax, &maxsuper,
		       &fill_ratio, &min_gemm_gpu_offload, &nrhs, &fp);

    /* Set the default input options. */
    set_default_options_dist(&options);
	
    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    /* Loop through all the input options. */
    for (imat = fimat; imat < nimat; ++imat) { /* All matrix types */
	/* ------------------------------------------------------------
	   GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
	   ------------------------------------------------------------*/
	dcreate_matrix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);

	m = A.nrow;
	n = A.ncol;

	/* Initialize ScalePermstruct and LUstruct. */
	ScalePermstructInit(m, n, &ScalePermstruct);
	LUstructInit(n, &LUstruct);
	
	/* Save a copy of matrix A in Asave. */
	Astore = (NRformat_loc *) A.Store;
	nzval_save = (double *) doubleMalloc_dist(Astore->nnz_loc);
	colind_save = (int_t *) intMalloc_dist(Astore->nnz_loc);
	rowptr_save = (int_t *) intMalloc_dist(Astore->m_loc + 1);
	dCreate_CompRowLoc_Matrix_dist(&Asave, m, n,
				       Astore->nnz_loc, Astore->m_loc, Astore->fst_row,
				       nzval_save, colind_save, rowptr_save,
				       SLU_NR_loc, SLU_D, SLU_GE);
	dCopy_CompRowLoc_Matrix_dist(&A, &Asave);

	if ( !iam ) { printf("after create Asave\n"); fflush(stdout); }

	if ( !(bsave = doubleMalloc_dist(ldb * nrhs)) )
	    ABORT("Malloc fails for bsave[]");
	for (j = 0; j < nrhs; ++j)
	    for (i = 0; i < ldb; ++i) bsave[i+j*ldb] = b[i+j*ldb];
	
	if ( !(berr = doubleMalloc_dist(nrhs)) )
	    ABORT("Malloc fails for berr[].");
	
	for (iequed = 0; iequed < 4; ++iequed) {
	    *equed = equeds[iequed];
	    if (iequed == 0) nfact = 4;
	    else nfact = 1; /* Only test factored, pre-equilibrated matrix */
	    
	    for (ifact = 0; ifact < nfact; ++ifact) {
		fact = facts[ifact];
#if 0
		options.Fact = fact;
		for (equil = 0; equil < 2; ++equil) {
		    options.Equil = equil;
		    prefact   = ( options.Fact == FACTORED ||
				  options.Fact == SamePattern_SameRowPerm );
                                /* Need a first factor */
		    notfactored = (options.Fact != FACTORED);  /* Not factored */

		    /* Restore the matrix A. */
		    dCopy_CompRowLoc_Matrix_dist(&Asave, &A);

		    if ( options.Fact == FACTORED) {
                        if ( equil || iequed ) {
			    /* Compute row and column scale factors to
			       equilibrate matrix A.    */
			    pdgsequ(&A, R, C, &rowcnd, &colcnd, &amax, &info,
				    &grid);

			    /* Force equilibration. */
			    if ( info==0 && n > 0 ) {
				if ( strncmp(equed, "R", 1)==0 ) {
				    rowcnd = 0.;
				    colcnd = 1.;
				    ScalePermstruct->DiagScale = ROW;
				} else if ( strncmp(equed, "C", 1)==0 ) {
				    rowcnd = 1.;
				    colcnd = 0.;
				    ScalePermstruct->DiagScale = COL;
				} else if ( strncmp(equed, "B", 1)==0 ) {
				    rowcnd = 0.;
				    colcnd = 0.;
				    ScalePermstruct->DiagScale = BOTH;
				}
			    }
			
			    /* Equilibrate the matrix. */
			    pdlaqgs(&A, R, C, rowcnd, colcnd, amax, equed);
			}
		    }

		    if ( prefact ) { /* Need a first factor */
			
		        /* Save Fact option. */
		        fact = options.Fact;
			options.Fact = DOFACT;

			/* Initialize the statistics variables. */
			PStatInit(&stat);
	
			/* Only performs factorization. */
			int nrhs1 = 0;
			pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs1,
				&grid, &LUstruct, &SOLVEstruct,
				berr, &stat, &info);
			if ( info ) { 
                            printf("** First factor: info %d, equed %c\n",
				   info, *equed);
			}
		        /* Restore Fact option. */
			options.Fact = fact;
		    } /* if .. first time factor */

		    /* Restore the matrix A. */
		    dCopy_CompRowLoc_Matrix_dist(&Asave, &A);

		    /* Set the right-hand side. */
		    /* ... to be coded .. */
			
		    /*----------------
		     * Test pdgssvx
		     *----------------*/
    
		    /* Equilibrate the matrix if fact = FACTORED and
		       equed = 'R', 'C', or 'B'.   */
		    if ( options.Fact == FACTORED &&
			 (equil || iequed) && n > 0 ) {
			pdlaqgs(&A, R, C, rowcnd, colcnd, amax, equed);
		    }
			
		    /* Solve the system and compute the error bounds.      */
		    pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs,
			    &grid, &LUstruct, &SOLVEstruct,
			    berr, &stat, &info);
		    if ( info && info != izero ) {
			printf(FMT3, "pdgssvx",info,izero,n,nrhs,imat,nfail);
		    } else {
#if 0
			dgst02(trans, m, n, nrhs, &Asave, solx, ldx,
					  wwork, ldb, &result[1]);
#endif
			/* Compute residual of the computed solution.*/
			solx = b;
			pdcompute_resid(m, n, nrhs, &Asave, solx, ldx, bsave, ldb,
					&grid, &SOLVEstruct, &result[1]);
			
#if 0  /* how to get RCOND? */
			/* Check solution accuracy from generated exact solution. */
			dgst04(n, nrhs, solx, ldx, xact, ldx, rcond,
					  &result[2]);
			pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
					 nrhs, b, ldb, xtrue, ldx, &grid);
#endif

			/* Check the error bounds from iterative refinement. */
			dgst07(trans, n, nrhs, &ASAV, bsav, ldb,
					  solx, ldx, xact, ldx, ferr, berr,
					  &result[3]);

			/* Print information about the tests that did
				   not pass the threshold.    */
			for (i = k1; i < NTESTS; ++i) {
			    if ( result[i] >= THRESH ) {
				printf(FMT2, "pdgssvx",
				       options.Fact, trans, *equed,
				       n, imat, i, result[i]);
				++nfail;
			    }
			}
			nrun += NTESTS;
		    } /* end else .. info == 0 */
		   
		    /* Destroy data structures. */
		    
		} /* end for equil ... */
#endif

	    } /* end for ifact ... */

	} /* end for iequed ... */

	/* ------------------------------------------------------------
	   WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME.
	   ------------------------------------------------------------*/
	
	/* Initialize the statistics variables. */
	PStatInit(&stat);
	
	if ( !iam ) {printf("\tBefore pdgssvx:\n"); fflush(stdout);}

	/* Call the linear equation solver. */
	pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
		&LUstruct, &SOLVEstruct, berr, &stat, &info);
	
	
	/* Check the accuracy of the solution. */
	if ( !iam ) printf("\tSolve the first system:\n");
	pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
			 nrhs, b, ldb, xtrue, ldx, &grid);
#if 1
	/* Compute residual of the computed solution.*/
	solx = b;
	pdcompute_resid(m, n, nrhs, &Asave, solx, ldx, bsave, ldb,
			&grid, &SOLVEstruct, &result[1]);
	if ( !iam ) printf("Residual test result: %12.5g\n", result[1]);
#endif

	PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
	PStatFree(&stat);
	
	/* ------------------------------------------------------------
	   DEALLOCATE STORAGE.
	   ------------------------------------------------------------*/
	Destroy_CompRowLoc_Matrix_dist(&A);
	Destroy_CompRowLoc_Matrix_dist(&Asave);
	ScalePermstructFree(&ScalePermstruct);
	Destroy_LU(n, &grid, &LUstruct);
	LUstructFree(&LUstruct);
	if ( options.SolveInitialized ) {
	    dSolveFinalize(&options, &SOLVEstruct);
	}
	SUPERLU_FREE(b);
	SUPERLU_FREE(bsave);
	SUPERLU_FREE(xtrue);
	SUPERLU_FREE(berr);
	
    } /* end for imat ... */

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

/*  
 * Parse command line options to get various input parameters.
 */
static void
parse_command_line(int argc, char *argv[], int *nprow, int *npcol,
		   char *matrix_type, int *n, int *relax, int *maxsuper,
		   int *fill_ratio, int *min_gemm_gpu_offload,
		   int *nrhs, FILE **fp)
{
    int c;
    extern char *optarg;
    char  str[20];

    while ( (c = getopt(argc, argv, "hr:c:t:n:x:m:b:g:s:f:")) != EOF ) {
	switch (c) {
	  case 'h':
	    printf("Options:\n");
	    printf("\t-r <int> - process rows\n");
	    printf("\t-c <int> - process columns\n");
	    printf("\t-n <int> - matrix dimension\n");
	    printf("\t-x <int> - granularity of relaxed supernodes\n");
	    printf("\t-m <int> - maximum size of supernode\n");
	    printf("\t-b <int> - estimated fill ratio to allocate storage\n");
	    printf("\t-g <int> - minimum size of GEMM to offload to GPU\n");
	    printf("\t-s <int> - number of right-hand sides\n");
	    printf("\t-f <char[]> - file name storing a sparse matrix\n");
	    exit(1);
	    break;
	  case 'r': *nprow = atoi(optarg);
	            break;
	  case 'c': *npcol = atoi(optarg);
	            break;
	  case 'n': *n = atoi(optarg);
	            break;
	  case 'x': c = atoi(optarg); 
	            sprintf(str, "%d", c);
	            setenv("NREL", str, 1);
	            printf("Reset relax env. variable to %d\n", c);
	            break;
	  case 'm': c = atoi(optarg); 
	            sprintf(str, "%d", c);
		    setenv("NSUP", str, 1);
		    printf("Reset maxsuper env. variable to %d\n", c);
	            break;
	  case 'b': c = atoi(optarg); 
	            sprintf(str, "%d", c);
		    setenv("FILL", str, 1);
		    printf("Reset fill_ratio env. variable to %d\n", c);
	            break;
	  case 'g': c = atoi(optarg); 
	            sprintf(str, "%d", c);
		    setenv("N_GEMM", str, 1);
		    printf("Reset min_gemm_gpu_offload env. variable to %d\n", c);
	            break;
	  case 's': *nrhs = atoi(optarg); 
	            break;
          case 'f':
                    if ( !(*fp = fopen(optarg, "r")) ) {
                        ABORT("File does not exist");
                    }
                    printf(".. test sparse matrix in file: %s\n", optarg);
                    break;
  	}
    }
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
