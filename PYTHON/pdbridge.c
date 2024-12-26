/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief The bridge routines for the Python interface
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.1.0) --
 * Lawrence Berkeley National Lab
 * November 10, 2024
 * </pre>
 */

#include "pdbridge.h"
#include <math.h>
#include <stdbool.h>

void pdbridge_init(int_t m, int_t n, int_t nnz, int_t *rowind, int_t *colptr , double *nzval, void ** pyobj, int argc, char *argv[])
{
    slu_handle* slu_obj = (slu_handle*)malloc(sizeof(slu_handle));
    double   *berr;
    double   *b, *xtrue;
    int    m1, n1;
    int      nprow, npcol, lookahead, colperm, rowperm, ir, symbfact, batch, sympattern;
    int      iam, info, ldb, ldx;
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
    // nrhs = 1;   /* Number of right-hand side. */
    lookahead = -1;
    colperm = -1;
    rowperm = -1;
    ir = -1;
    symbfact = -1;
    sympattern=0;
    batch = 0;

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------*/
    //MPI_Init( &argc, &argv );
    // MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level);


#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( VTUNE>=1 )
	__itt_pause();
#endif

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
    set_default_options_dist(&(slu_obj->options));
#if 0
    options.RowPerm = LargeDiag_HWPM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = YES;
#endif

    /* Parse command line argv[], may modify default options */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
            case 'h':
                printf("Options:\n");
                printf("\t-m <int>: symmetric pattern  (default %4d)\n", sympattern);
                printf("\t-r <int>: process rows       (default %4d)\n", nprow);
                printf("\t-c <int>: process columns    (default %4d)\n", npcol);
                printf("\t-p <int>: row permutation    (default %4d)\n", (slu_obj->options).RowPerm);
                printf("\t-q <int>: col permutation    (default %4d)\n", (slu_obj->options).ColPerm);
                printf("\t-s <int>: parallel symbolic? (default %4d)\n", (slu_obj->options).ParSymbFact);
                printf("\t-l <int>: lookahead level    (default %4d)\n", (slu_obj->options).num_lookaheads);
                printf("\t-i <int>: iter. refinement   (default %4d)\n", (slu_obj->options).IterRefine);
                printf("\t-b <int>: use batch mode?    (default %4d)\n", batch);
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
            case 'm': sympattern = atoi(*cpp);
                    break;                    
	    }
	} else { /* Last arg is considered a filename */
	    if ( !(fp = fopen(*cpp, "r")) ) {
                ABORT("File does not exist");
            }
	    break;
	}
    }

    /* Command line input to modify default options */
    if (rowperm != -1) (slu_obj->options).RowPerm = rowperm;
    if (colperm != -1) (slu_obj->options).ColPerm = colperm;
    if (lookahead != -1) (slu_obj->options).num_lookaheads = lookahead;
    if (ir != -1) (slu_obj->options).IterRefine = ir;
    if (symbfact != -1) (slu_obj->options).ParSymbFact = symbfact;
    if (sympattern==1) (slu_obj->options).SymPattern = YES;

    int superlu_acc_offload = sp_ienv_dist(10, &(slu_obj->options)); //get_acc_offload();
    
    /* In the batch mode: create multiple SuperLU grids,
        each grid solving one linear system. */
    if ( batch ) {

    } else { /* not batch mode */
        /* ------------------------------------------------------------
           INITIALIZE THE SUPERLU PROCESS GRID.
           ------------------------------------------------------------ */
        superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &(slu_obj->grid));
#ifdef GPU_ACC
        if (superlu_acc_offload) {
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            double t1 = SuperLU_timer_();
            gpuFree(0);
            double t2 = SuperLU_timer_();
            if(!myrank)printf("first gpufree time: %7.4f\n",t2-t1);
            gpublasHandle_t hb;
            gpublasCreate(&hb);
            if(!myrank)printf("first blas create time: %7.4f\n",SuperLU_timer_()-t2);
            gpublasDestroy(hb);
	}
#endif
    }

    if((slu_obj->grid).iam==0){
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
    iam = (slu_obj->grid).iam;
    if ( (iam >= nprow * npcol) || (iam == -1) ) goto out;
    if ( !iam ) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	// printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid:\t\t%d X %d\n", (int)(slu_obj->grid).nprow, (int)(slu_obj->grid).npcol);
	fflush(stdout);
    }

    /* print solver options */
    if (!iam) {
	print_options_dist(&(slu_obj->options));
	fflush(stdout);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    dcreate_matrix_from_csc(&(slu_obj->A),m, n, nnz, rowind, colptr, nzval, &(slu_obj->grid));
    // if ( !(berr = doubleMalloc_dist(nrhs)) )
	// ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------*/

    m1 = (slu_obj->A).nrow;
    n1 = (slu_obj->A).ncol;

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit(m1, n1, &(slu_obj->ScalePermstruct));
    dLUstructInit(n1, &(slu_obj->LUstruct));

    /* Initialize the statistics variables. */
    PStatInit(&(slu_obj->stat));

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pdbridge_init()");
#endif

    *pyobj = (void*)slu_obj;

}





void pdbridge_factor(void ** pyobj)
{
    slu_handle* slu_obj = (slu_handle*)(*pyobj);
    int_t    m_loc, fst_row, nnz_loc,row;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    double   *berr;
    double   *b;
    int    m1, n1;
    int    nrhs=1;
    int      nprow, npcol, lookahead, colperm, rowperm, ir, symbfact, batch;
    int      iam, info, ldb, ldx;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();
    int ii, omp_mpi_level,m;
    int ldumap, myrank, p; /* The following variables are used for batch solves */
    int*    usermap;
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;


    /* Bail out if I do not belong in the grid. */
    iam = (slu_obj->grid).iam;
    nprow = (slu_obj->grid).nprow;
    npcol = (slu_obj->grid).npcol;
    if ( (iam >= nprow * npcol) || (iam == -1) ) goto out;
    m = (slu_obj->A).nrow;


    /* Compute the number of rows to be distributed to local process */
    m_loc = m / ((slu_obj->grid).nprow * (slu_obj->grid).npcol); 
    m_loc_fst = m_loc;
    fst_row = iam * m_loc_fst;
    /* When m / procs is not an integer */
    if ((m_loc * (slu_obj->grid).nprow * (slu_obj->grid).npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == ((slu_obj->grid).nprow * (slu_obj->grid).npcol - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * ((slu_obj->grid).nprow * (slu_obj->grid).npcol - 1);
    }
        
    /* Get the local B */
    if ( !(b = doubleMalloc_dist(m_loc*nrhs)) )
        ABORT("Malloc fails for rhs[]");
    for (int j =0; j < nrhs; ++j) {
	for (int i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    b[j*m_loc+i] = 0; //dummy RHS
	}
    }
    ldb = m_loc;

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* Call the linear equation solver. */
    pdgssvx(&(slu_obj->options), &(slu_obj->A), &(slu_obj->ScalePermstruct), b, ldb, nrhs, &(slu_obj->grid),
	    &(slu_obj->LUstruct), &(slu_obj->SOLVEstruct), berr, &(slu_obj->stat), &info);


    PStatPrint(&(slu_obj->options), &(slu_obj->stat), &(slu_obj->grid));        /* Print the statistics. */
    slu_obj->options.Fact = FACTORED;

    // /* ------------------------------------------------------------
    //    DEALLOCATE STORAGE.
    //    ------------------------------------------------------------*/

    SUPERLU_FREE(b);
    SUPERLU_FREE(berr);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:


#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pdbridge_factor()");
#endif

    *pyobj = (void*)slu_obj;

}


void pdbridge_solve(void ** pyobj, int nrhs, double   *b_global)
{
    slu_handle* slu_obj = (slu_handle*)(*pyobj);
    int_t    m_loc, fst_row, nnz_loc,row;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    double   *berr;
    double   *b;
    int    m1, n1;
    int      nprow, npcol, lookahead, colperm, rowperm, ir, symbfact, batch;
    int      iam, info, ldb, ldx;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();
    int ii, omp_mpi_level,m;
    int ldumap, myrank, p; /* The following variables are used for batch solves */
    int*    usermap;
    float result_min[2];
    result_min[0]=1e10;
    result_min[1]=1e10;
    float result_max[2];
    result_max[0]=0.0;
    result_max[1]=0.0;
    MPI_Comm SubComm;


    /* Bail out if I do not belong in the grid. */
    iam = (slu_obj->grid).iam;
    nprow = (slu_obj->grid).nprow;
    npcol = (slu_obj->grid).npcol;
    if ( (iam >= nprow * npcol) || (iam == -1) ) goto out;
    m = (slu_obj->A).nrow;

    if (iam == 0) {
        MPI_Bcast( b_global, m*nrhs, MPI_DOUBLE, 0, (slu_obj->grid).comm );
    } else {
        MPI_Bcast( b_global, m*nrhs, MPI_DOUBLE, 0, (slu_obj->grid).comm );
    }


    /* Compute the number of rows to be distributed to local process */
    m_loc = m / ((slu_obj->grid).nprow * (slu_obj->grid).npcol); 
    m_loc_fst = m_loc;
    fst_row = iam * m_loc_fst;
    /* When m / procs is not an integer */
    if ((m_loc * (slu_obj->grid).nprow * (slu_obj->grid).npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == ((slu_obj->grid).nprow * (slu_obj->grid).npcol - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * ((slu_obj->grid).nprow * (slu_obj->grid).npcol - 1);
    }
        
    /* Get the local B */
    if ( !(b = doubleMalloc_dist(m_loc*nrhs)) )
        ABORT("Malloc fails for rhs[]");
    for (int j =0; j < nrhs; ++j) {
	for (int i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    b[j*m_loc+i] = b_global[j*m+row];
	}
    }
    ldb = m_loc;

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* Call the linear equation solver. */
    pdgssvx(&(slu_obj->options), &(slu_obj->A), &(slu_obj->ScalePermstruct), b, ldb, nrhs, &(slu_obj->grid),
	    &(slu_obj->LUstruct), &(slu_obj->SOLVEstruct), berr, &(slu_obj->stat), &info);

    for (int j =0; j < nrhs; ++j) {
	for (int i = 0; i < m_loc; ++i) {
	    row = fst_row + i;
	    b_global[j*m+row] = b[j*m_loc+i] ;
	}
    }

    PStatPrint(&(slu_obj->options), &(slu_obj->stat), &(slu_obj->grid));        /* Print the statistics. */
    // slu_obj->options.Fact = FACTORED;

    // /* ------------------------------------------------------------
    //    DEALLOCATE STORAGE.
    //    ------------------------------------------------------------*/

    SUPERLU_FREE(b);
    SUPERLU_FREE(berr);


    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:


#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pdbridge_solve()");
#endif

    *pyobj = (void*)slu_obj;

}





void pdbridge_logdet(void ** pyobj, int * sign, double * logdet)
{
    slu_handle* slu_obj = (slu_handle*)(*pyobj);
    Glu_persist_t *Glu_persist = slu_obj->LUstruct.Glu_persist;
    dLocalLU_t *Llu = slu_obj->LUstruct.Llu;
    int_t  **Lrowind_bc_ptr=Llu->Lrowind_bc_ptr;
    double **Lnzval_bc_ptr=Llu->Lnzval_bc_ptr;    
    int_t  *xsup=Glu_persist->xsup;
    int_t  *lsub;
    double *lusup;
    int m = (slu_obj->A).nrow;
    gridinfo_t *grid = &(slu_obj->grid);
    dScalePermstruct_t *ScalePermstruct = &(slu_obj->ScalePermstruct);
    int iam = grid->iam;
    int nsupers = Glu_persist->supno[m-1] + 1;
    int nswap=count_swaps(ScalePermstruct->perm_r,m);
    int sign_rowperm = (nswap % 2 == 0) ? 1 : -1;
    double logdet_Dr=0;
    int sign_Dr=1;
    double logdet_Dc=0;
    int sign_Dc=1;    
    int knsupc, krow, kcol, diag, lk, nsupr; 
    if(!iam){
        if(ScalePermstruct->DiagScale == ROW || ScalePermstruct->DiagScale == BOTH){
            for(int_t i=0; i<m; i++){
                logdet_Dr-=log(fabs(ScalePermstruct->R[i]));
                sign_Dr*=copysign(1.0, ScalePermstruct->R[i]);
            }
        }
        if(ScalePermstruct->DiagScale == COL || ScalePermstruct->DiagScale == BOTH){
            for(int_t i=0; i<m; i++){
                logdet_Dc-=log(fabs(ScalePermstruct->C[i]));
                sign_Dc*=copysign(1.0, ScalePermstruct->C[i]);
            }
        }
    }
    

    double logdet_U=0;
    int sign_U=1;
    for (int k = 0; k < nsupers; ++k) {
        knsupc = SuperSize( k );
        krow = PROW( k, grid );
        kcol = PCOL( k, grid );
        diag = PNUM( krow, kcol, grid);
        if ( iam == diag ) { /* Diagonal process. */
            lk = LBj( k, grid ); /* Local block number, column-wise */
            lsub = Lrowind_bc_ptr[lk];
            lusup = Lnzval_bc_ptr[lk];
            nsupr = lsub[1];

            for (int i = 0; i < knsupc; i++) {
                double diagonal = lusup[i * nsupr + i];  // Accessing diagonal element U(i,i)
                logdet_U+=log(fabs(diagonal));
                sign_U*=copysign(1.0, diagonal);                    
            }
        }
    }
#if ( PRNTlevel>=1 )
    printf("sign: %5d %5d %5d %5d\n",sign_Dr,sign_Dc,sign_rowperm,sign_U);
    printf("logdet: %10f %10f %10f \n",logdet_Dr,logdet_Dc,logdet_U);
    fflush(stdout);
#endif

    MPI_Allreduce(MPI_IN_PLACE,&logdet_U,1,MPI_DOUBLE,MPI_SUM,grid->comm);
    MPI_Allreduce(MPI_IN_PLACE,&sign_U,1,mpi_int_t,MPI_PROD,grid->comm);

    if(!iam){
        *sign = sign_Dr*sign_Dc*sign_rowperm*sign_U;
        *logdet = logdet_Dr + logdet_Dc + logdet_U;
    }

    MPI_Bcast( &sign, 1, mpi_int_t, 0,  grid->comm);
    MPI_Bcast( &logdet, 1, MPI_DOUBLE, 0,  grid->comm);


    *pyobj = (void*)slu_obj;

}


void pdbridge_free(void ** pyobj)
{
    int iam, m;
    slu_handle* slu_obj = (slu_handle*)(*pyobj);
    
    // /* ------------------------------------------------------------
    //    DEALLOCATE STORAGE.
    //    ------------------------------------------------------------*/

    m = (slu_obj->A).nrow;
    iam = (slu_obj->grid).iam;

    Destroy_CompRowLoc_Matrix_dist(&(slu_obj->A));
    dScalePermstructFree(&(slu_obj->ScalePermstruct));
    dDestroy_LU(m, &(slu_obj->grid), &(slu_obj->LUstruct));
    dLUstructFree(&(slu_obj->LUstruct));
    dSolveFinalize(&(slu_obj->options), &(slu_obj->SOLVEstruct));

    superlu_gridexit(&(slu_obj->grid));

    if ( iam != -1 ) PStatFree(&(slu_obj->stat));

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

    *pyobj = (void*)slu_obj;

}




int dcreate_matrix_from_csc(SuperMatrix *A,
                   int_t m, int_t n, int_t nnz, int_t *rowind0, int_t *colptr0, double   *nzval0, gridinfo_t *grid)
{
    SuperMatrix GA;              /* global A */
    double   *b_global, *xtrue_global;  /* replicated on all processes */
    int_t    *rowind, *colptr;	 /* global */
    double   *nzval;             /* global */
    double   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;	 /* local */
    // int_t    m, n, nnz;
    int_t    m_loc, fst_row, nnz_loc;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
			   when mod(m, p) is not zero. */ 
    int_t    row, col, i, j, relpos;
    int      iam;
    char     trans[1];
    int_t      *marker;
    int_t chunk= 2000000000;
    int count;
    int_t Nchunk;
    int_t remainder;

    iam = grid->iam;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dcreate_matrix()");
#endif

    if ( !iam ) {

	dallocateA_dist(n, nnz, &nzval, &rowind, &colptr);
    for (int_t i=0; i<nnz; i++){
        nzval[i]=nzval0[i];
        rowind[i]=rowind0[i];
    }
    for (int_t i=0; i<n+1; i++)
        colptr[i]=colptr0[i];

	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,     1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz,   1,   mpi_int_t,  0, grid->comm );

    
    Nchunk = CEILING(nnz,chunk);
    remainder =  nnz%chunk;
	MPI_Bcast( &Nchunk,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &remainder,   1,   mpi_int_t,  0, grid->comm );

    for (i = 0; i < Nchunk; ++i) {
       int_t idx=i*chunk;
       if(i==Nchunk-1){
            count=remainder;
       }else{
            count=chunk;
       }  
        MPI_Bcast( &nzval[idx],  count, MPI_DOUBLE, 0, grid->comm );
        MPI_Bcast( &rowind[idx], count, mpi_int_t,  0, grid->comm );       
    }

	MPI_Bcast( colptr, n+1, mpi_int_t,  0, grid->comm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &Nchunk,   1,   mpi_int_t,  0, grid->comm );
	MPI_Bcast( &remainder,   1,   mpi_int_t,  0, grid->comm );

	/* Allocate storage for compressed column representation. */
	dallocateA_dist(n, nnz, &nzval, &rowind, &colptr);

    for (i = 0; i < Nchunk; ++i) {
       int_t idx=i*chunk;
       if(i==Nchunk-1){
            count=remainder;
       }else{
            count=chunk;
       }  
        MPI_Bcast( &nzval[idx],  count, MPI_DOUBLE, 0, grid->comm );
        MPI_Bcast( &rowind[idx], count, mpi_int_t,  0, grid->comm );       
    }
	MPI_Bcast( colptr,  n+1, mpi_int_t,  0, grid->comm );
    }

#if 0
    nzval[0]=0.1;
#endif

    /* Compute the number of rows to be distributed to local process */
    m_loc = m / (grid->nprow * grid->npcol); 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * grid->nprow * grid->npcol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid->nprow * grid->npcol - 1)) /* last proc. gets all*/
	  m_loc = m - m_loc * (grid->nprow * grid->npcol - 1);
    }

    /* Create compressed column matrix for GA. */
    dCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval, rowind, colptr,
				SLU_NC, SLU_D, SLU_GE);
     
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
    // NCformat *Astore = GA.Store;
    // SUPERLU_FREE(Astore);

    /******************************************************/
    /* Change GA to a local A with NR_loc format */
    /******************************************************/

    /* Set up the local A in NR_loc format */
    dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				   nzval_loc, colind, rowptr,
				   SLU_NR_loc, SLU_D, SLU_GE);

    SUPERLU_FREE(marker);

#if ( DEBUGlevel>=1 )
    printf("sizeof(NRforamt_loc) %lu\n", sizeof(NRformat_loc));
    CHECK_MALLOC(iam, "Exit dcreate_matrix()");
#endif
    return 0;
}


int count_swaps(int perm[], int n) {
    bool visited[n];
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    int swaps = 0;
    for (int i = 0; i < n; i++) {
        if (visited[i] || perm[i] == i) {
            continue;
        }
        int cycle_len = 0;
        int x = i;
        while (!visited[x]) {
            visited[x] = true;
            x = perm[x];
            cycle_len++;
        }
        if (cycle_len > 1) {
            swaps += cycle_len - 1;
        }
    }
    return swaps;
}
