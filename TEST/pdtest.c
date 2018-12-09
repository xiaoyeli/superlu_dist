/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for testing PDGSSVX.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 5.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 30, 2017
 * </pre>
 */
/*
 * File name:		pdtest.c
 * Purpose:             MAIN test program
 */
#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h>
#ifdef _MSC_VER
#include <wingetopt.h>
#else
#include <getopt.h>
#endif
#include <math.h>
#include "superlu_dist_config.h"
#include "superlu_ddefs.h"

#define NTESTS 1 /*5*/      /* Number of test types */
#define NTRAN  2    
#define THRESH 20.0
#define FMT1   "%10s:n=%d, test(%d)=%12.5g\n"
#define	FMT2   "%10s:fact=%4d, DiagScale=%d, n=%d, imat=%d, test(%d)=%12.5g, berr=%12.5g\n"
#define FMT3   "%10s:info=%d, izero=%d, n=%d, nrhs=%d, imat=%d, nfail=%d\n"


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
dCopy_CompRowLoc_NoAllocation(SuperMatrix *A, SuperMatrix *B)
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

/*! \brief Print a summary of the testing results. */
void
PrintSumm(char *type, int nfail, int nrun, int nerrs)
{
    if ( nfail > 0 )
	printf("%3s driver: %d out of %d tests failed to pass the threshold\n",
	       type, nfail, nrun);
    else
	printf("All tests for %3s driver passed the threshold (%6d tests run)\n", type, nrun);

    if ( nerrs > 0 )
	printf("%6d error messages recorded\n", nerrs);
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
    double   *berr, *R, *C;
    double   *b, *bsave, *xtrue, *solx;
    int    i, j, m, n, izero = 0;
    int    nprow, npcol;
    int    iam, info, ldb, ldx, nrhs;
    int_t  iinfo;
    char     **cpp, c;
    FILE *fp, *fopen();
    char matrix_type[8], equed[1];
    int  relax, maxsuper=sp_ienv_dist(3), fill_ratio=sp_ienv_dist(6),
         min_gemm_gpu_offload=0;
    int    equil, ifact, nfact, iequil, iequed, prefact, notfactored, diaginv;
    int    nt, nrun=0, nfail=0, nerrs=0, imat, fimat=0;
    int    nimat=1;  /* Currently only test a sparse matrix read from a file. */
    fact_t fact;
    double rowcnd, colcnd, amax;
    double result[NTESTS];

    /* Fixed set of parameters */
    int     iseed[]  = {1988, 1989, 1990, 1991};
    char    equeds[]  = {'N', 'R', 'C', 'B'};
    DiagScale_t equils[] = {NOEQUIL, ROW, COL, BOTH};
    fact_t  facts[] = {FACTORED, DOFACT, SamePattern, SamePattern_SameRowPerm};
    trans_t transs[]  = {NOTRANS, TRANS, CONJ};

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */
    for (i = 0; i < NTESTS; ++i) result[i] = 0.0;

    /* Parse command line argv[]. */
    parse_command_line(argc, argv, &nprow, &npcol, matrix_type, &n,
		       &relax, &maxsuper,
		       &fill_ratio, &min_gemm_gpu_offload, &nrhs, &fp);

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
    if ( 0 ) {
        printf("\tProcess grid\t%d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* Set the default input options. */
    set_default_options_dist(&options);
    options.PrintStat = NO;
	
    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");
	
    /* Loop through all the input options. */
    for (imat = fimat; imat < nimat; ++imat) { /* All matrix types */
	//if (!iam) printf("imat loop ... %d\n", imat);
	/* ------------------------------------------------------------
	   GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
	   ------------------------------------------------------------*/
	dcreate_matrix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);

	m = A.nrow;
	n = A.ncol;

	if ( !(bsave = doubleMalloc_dist(ldb * nrhs)) )
	    ABORT("Malloc fails for bsave[]");
	for (j = 0; j < nrhs; ++j)
	    for (i = 0; i < ldb; ++i) bsave[i+j*ldb] = b[i+j*ldb];

	/* Save a copy of matrix A in Asave. */
	Astore = (NRformat_loc *) A.Store;
	int_t nnz_loc = Astore->nnz_loc;
	int_t m_loc = Astore->m_loc;
	nzval_save = (double *) doubleMalloc_dist(nnz_loc);
	colind_save = (int_t *) intMalloc_dist(nnz_loc);
	rowptr_save = (int_t *) intMalloc_dist(m_loc + 1);
	dCreate_CompRowLoc_Matrix_dist(&Asave, m, n, nnz_loc, m_loc, Astore->fst_row,
				       nzval_save, colind_save, rowptr_save,
				       SLU_NR_loc, SLU_D, SLU_GE);
	dCopy_CompRowLoc_NoAllocation(&A, &Asave);

	for (iequed = 0; iequed < 4; ++iequed) {
	    int what_equil = equils[iequed];
	    if (iequed == 0) nfact = 4;
	    else { /* Only test factored, pre-equilibrated matrix */
		nfact = 1;
		options.RowPerm = NOROWPERM; /* Turn off MC64 */
	    }
	    //if (!iam) printf("iequed loop ... %d\n", iequed);

	    for (ifact = 0; ifact < nfact; ++ifact) {
		fact = facts[ifact];
		options.Fact = fact;
		//if (!iam) printf("ifact loop ... %d\n", ifact);
#ifdef SLU_HAVE_LAPACK 
	        for (diaginv = 0; diaginv < 2; ++diaginv) {
#endif
		    for (equil = 0; equil < 2; ++equil) {

		    	//if (!iam) printf("equil loop ... %d\n", equil);

		    	options.Equil = equil;

		    	/* Need a first factor */
		    	prefact   = ( options.Fact == FACTORED ||
				     options.Fact == SamePattern ||
				     options.Fact == SamePattern_SameRowPerm );

		        /* Restore the matrix A. */
		        dCopy_CompRowLoc_NoAllocation(&Asave, &A);

		        /* Initialize ScalePermstruct and LUstruct. */
		        ScalePermstructInit(m, n, &ScalePermstruct);
		        LUstructInit(n, &LUstruct);

		        if ( prefact ) {

			    R = (double *) SUPERLU_MALLOC(m*sizeof(double));
			    C = (double *) SUPERLU_MALLOC(n*sizeof(double));
			
			    /* Later call to PDGSSVX only needs to solve. */
                            if ( equil || iequed ) {
			        /* Compute row and column scale factors to
			           equilibrate matrix A.    */
			        pdgsequ(&A, R, C, &rowcnd, &colcnd, &amax,
                                    &iinfo,&grid);

			        /* Force equilibration. */
			    	if ( iinfo==0 && n > 0 ) {
				   if ( what_equil == ROW ) {
				      rowcnd = 0.;
				      colcnd = 1.;
				      ScalePermstruct.DiagScale = ROW;
				      ScalePermstruct.R = R;
				   } else if ( what_equil == COL ) {
				      rowcnd = 1.;
				      colcnd = 0.;
				      ScalePermstruct.DiagScale = COL;
				      ScalePermstruct.C = C;
				   } else if ( what_equil == BOTH ) {
				      rowcnd = 0.;
				      colcnd = 0.;
				      ScalePermstruct.DiagScale = BOTH;
				      ScalePermstruct.R = R;
				      ScalePermstruct.C = C;
				   }
			        }
			
			        /* Equilibrate the matrix. */
			    	pdlaqgs(&A, R, C, rowcnd, colcnd, amax, equed);
			    	// printf("after pdlaqgs: *equed %c\n", *equed);

			    	/* Not equilibrate anymore when calling 
				   PDGSSVX, so, no malloc/free {R,C}
				   inside PDGSSVX. */
			    	options.Equil = NO;
			    } /* end if (equil || iequed) */
		    	} /* end if prefact */

		        if ( prefact ) { /* Need a first factor */
			
			    /* Save Fact option. */
		            fact = options.Fact;
			    options.Fact = DOFACT;

			    /* Initialize the statistics variables. */
			    PStatInit(&stat);
	
			    int nrhs1 = 0; /* Only performs factorization */
			    pdgssvx(&options, &A, &ScalePermstruct, b,
                                ldb, nrhs1, &grid, &LUstruct, &SOLVEstruct,
				berr, &stat, &info);

			    if ( info ) {
			        printf("** First factor: nrun %d: fact %d, info %d, "
				   "equil %d, what_equil %d, DiagScale %d \n",
				   nrun, fact, info, equil, what_equil,
				   ScalePermstruct.DiagScale);
			    }

			    PStatFree(&stat);

		            /* Restore Fact option. */
			    options.Fact = fact;
			    if ( fact == SamePattern ) {
			        // {L,U} not re-used in subsequent call to PDGSSVX.
			        Destroy_LU(n, &grid, &LUstruct);
			    } else if (fact == SamePattern_SameRowPerm) {
			        // {L,U} structure is re-used in subsequent call to PDGSSVX.
				dZeroLblocks(iam, n, &grid, &LUstruct);
                            }

		        } /* end if .. first time factor */

		        /*----------------
		     	 * Test pdgssvx
		         *----------------*/

		        if ( options.Fact != FACTORED ) {
			    /* Restore the matrix A. */
			    dCopy_CompRowLoc_NoAllocation(&Asave, &A);
			    if (fact == SamePattern_SameRowPerm && iam == 0) {
                                /* Perturb the 1st diagonal of the matrix 
                                   to larger value, so to have a different A. */
                                ((double *) Astore->nzval)[0] += 1.0e-8;
                             }

		        } 

		        /* Set the right-hand side. */
		        dCopy_Dense_Matrix_dist(m_loc, nrhs, bsave, ldb, b, ldb);

		        PStatInit(&stat);

		    /*if ( !iam ) printf("\ttest pdgssvx: nrun %d, iequed %d, equil %d, fact %d\n", 
		      nrun, iequed, equil, options.Fact);*/
		        /* Testing PDGSSVX: solve and compute the error bounds. */
		        pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs,
			    &grid, &LUstruct, &SOLVEstruct,
			    berr, &stat, &info);

		        PStatFree(&stat);
#if 0
		        pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
				     nrhs, b, ldb, xtrue, ldx, &grid);
#endif
		        if ( info ) {
			    printf(FMT3, "pdgssvx",info,izero,n,nrhs,imat,nfail);
		        } else {
			    /* Restore the matrix A. */
			    dCopy_CompRowLoc_NoAllocation(&Asave, &A);

			    /* Compute residual of the computed solution.*/
			    solx = b;
			    pdcompute_resid(m, n, nrhs, &A, solx, ldx,
                                        bsave, ldb, &grid, &SOLVEstruct, &result[0]);
			
#if 0  /* how to get RCOND? */
			/* Check solution accuracy from generated exact solution. */
			    dgst04(n, nrhs, solx, ldx, xact, ldx, rcond,
					  &result[2]);
			    pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
					 nrhs, b, ldb, xtrue, ldx, &grid);
#endif

			    /* Print information about the tests that did
			       not pass the threshold.    */
			    int k1 = 0;
			    for (i = k1; i < NTESTS; ++i) {
			        if ( result[i] >= THRESH ) {
				    printf(FMT2, "pdgssvx", options.Fact, 
				       ScalePermstruct.DiagScale,
				       n, imat, i, result[i], berr[0]);
				    ++nfail;
			        }
			    }
			    nrun += NTESTS;
		        } /* end else .. info == 0 */
		   
		        /* -------------------------------------------------
		           Deallocate storage associated with {L,U}.
		           ------------------------------------------------- */
		        if ( prefact ) {
			    SUPERLU_FREE(R);
			    SUPERLU_FREE(C);
			    ScalePermstruct.DiagScale = NOEQUIL; /* Avoid free R/C again. */
		        }
		        ScalePermstructFree(&ScalePermstruct);
		        Destroy_LU(n, &grid, &LUstruct);
		        LUstructFree(&LUstruct);
		        if ( options.SolveInitialized ) {
			    dSolveFinalize(&options, &SOLVEstruct);
		        }

		    } /* end for equil ... */

#ifdef SLU_HAVE_LAPACK 
                } /* end for diaginv ... */
#endif
	    } /* end for ifact ... */
		
	} /* end for iequed ... */
	
	/* ------------------------------------------------------------
	   DEALLOCATE STORAGE.
	   ------------------------------------------------------------*/
	Destroy_CompRowLoc_Matrix_dist(&A);
	Destroy_CompRowLoc_Matrix_dist(&Asave);
	//	ScalePermstructFree(&ScalePermstruct);
	SUPERLU_FREE(b);
	SUPERLU_FREE(bsave);
	SUPERLU_FREE(xtrue);

    } /* end for imat ... */

    /* Print a summary of the testing results. */
    if ( iam==0 ) PrintSumm("DGS", nfail, nrun, nerrs);

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
    char *xenvstr, *menvstr, *benvstr, *genvstr;
    xenvstr = menvstr = benvstr = genvstr = 0;

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
// Use putenv as exists on Windows
#ifdef _MSC_VER
#define putenv _putenv
#endif
	  case 'x': // c = atoi(optarg); 
	            // sprintf(str, "%d", c);
	            // setenv("NREL", str, 1);
		    xenvstr = (char*) malloc((6+strlen(optarg))*sizeof(char));
		    strcpy(xenvstr, "NREL=");
		    strcat(xenvstr, optarg);
		    putenv(xenvstr);
	            //printf("Reset relax env. variable to %d\n", c);
	            break;
	  case 'm': // c = atoi(optarg); 
	            // sprintf(str, "%d", c);
		    // setenv("NSUP", str, 1);
		    menvstr = (char*) malloc((6+strlen(optarg))*sizeof(char));
		    strcpy(menvstr, "NSUP=");
		    strcat(menvstr, optarg);
		    putenv(menvstr);
		    //printf("Reset maxsuper env. variable to %d\n", c);
	            break;
	  case 'b': // c = atoi(optarg); 
	            // sprintf(str, "%d", c);
		    // setenv("FILL", str, 1);
		    benvstr = (char*) malloc((6+strlen(optarg))*sizeof(char));
		    strcpy(benvstr, "FILL=");
		    strcat(benvstr, optarg);
		    putenv(benvstr);
		    //printf("Reset fill_ratio env. variable to %d\n", c);
	            break;
	  case 'g': // c = atoi(optarg); 
	            // sprintf(str, "%d", c);
		    // setenv("N_GEMM", str, 1);
		    genvstr = (char*) malloc((8+strlen(optarg))*sizeof(char));
		    strcpy(genvstr, "N_GEMM=");
		    strcat(genvstr, optarg);
		    putenv(genvstr);
		    //printf("Reset min_gemm_gpu_offload env. variable to %d\n", c);
	            break;
	  case 's': *nrhs = atoi(optarg); 
	            break;
          case 'f':
                    if ( !(*fp = fopen(optarg, "r")) ) {
                        ABORT("File does not exist");
                    }
                    //printf(".. test sparse matrix in file: %s\n", optarg);
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
