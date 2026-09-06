/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Example program for PDGSSVX3D_CSC_VBATCH
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.3) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 *
 */
#include <math.h>
#include "superlu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * This example illustrates how to use PDGSSVX3D_CSC_VBATCH to solve a batch
 * of systems repeatedly -- once per time step -- when the sparsity pattern of
 * every matrix in the batch stays the same from step to step.  It is the batch
 * counterpart of pddrive3d2.
 *
 * The first time step is solved with options.Fact = DOFACT, which does the
 * full preprocessing.  Every later step uses SamePattern_SameRowPerm, which
 * reuses, through the state carried in F[0]:
 *        ReqPtr, CeqPtr  : equilibration of each matrix in the batch
 *        RpivPtr         : row permutation of each matrix
 *        CpivPtr         : column permutation of each matrix
 *        F[0]            : etree, perm_c and symbolic factorization of the
 *                          stacked block-diagonal system, its distributed L/U
 *                          structure, the SpTRSV metadata, and the internal
 *                          process grid
 * so the per-step cost drops to the numerical factorization and the solve.
 *
 * The time-step decks are expected to be laid out one directory per step:
 *
 *      <prefix><t>/<base><t>_<d><suffix>          matrix d of step t
 *      <prefix><t>/<base><t>_rhs_<d><suffix>      its right-hand side
 *      <prefix><t>/<base><t>_x_<d><suffix>        its reference solution
 *
 * where <base> is the part of <prefix> after the last '/'.  For example, with
 * -f .dat and prefix .../IEEE39_IBL_100steps/IEEE39_IBL_ , step 7 of the batch
 * member 1 is read from
 *      .../IEEE39_IBL_100steps/IEEE39_IBL_7/IEEE39_IBL_7_1.dat
 *      .../IEEE39_IBL_100steps/IEEE39_IBL_7/IEEE39_IBL_7_rhs_1.dat
 *      .../IEEE39_IBL_100steps/IEEE39_IBL_7/IEEE39_IBL_7_x_1.dat
 *
 * The program may be run by typing
 *    mpiexec -np <p> pddrive3d_vbatch2 -b <batchCount> -t <nsteps> \
 *                                      -f <suffix> <prefix>
 *
 * </pre>
 */

/*! \brief Build the path of one file of one time step.
 *
 * kind is "" for the matrix, "rhs_" for the right-hand side, "x_" for the
 * reference solution.  Returns a malloc'd string the caller must free.
 */
static char *step_file_name(const char *prefix, const char *base, int t,
			    const char *kind, int d, const char *suffix)
{
    /* <prefix><t>/<base><t>_<kind><d><suffix> ; 12 chars is plenty for an int,
       plus the '/', the '_' and the terminating NUL. */
    size_t len = strlen(prefix) + strlen(base) + strlen(kind) + strlen(suffix)
	         + 3 * 12 + 4;
    char *name = SUPERLU_MALLOC(len);
    if ( !name ) ABORT("Malloc fails for the time-step file name");
    snprintf(name, len, "%s%d/%s%d_%s%d%s", prefix, t, base, t, kind, d, suffix);
    return name;
}

/*! \brief Open the batchCount files of one kind for time step t.
 *
 * Returns 1 on success.  With required = 0 a missing file is not fatal: any
 * file already opened is closed again and 0 is returned, which is how the
 * optional reference solutions are handled (not every step ships them).
 */
static int open_step_files(FILE **fp, const char *prefix, const char *base,
			   int t, const char *kind, int batchCount,
			   const char *suffix, int required)
{
    for (int d = 0; d < batchCount; ++d) {
	char *name = step_file_name(prefix, base, t, kind, d, suffix);
	if ( !(fp[d] = fopen(name, "r")) ) {
	    if ( required ) {
		fprintf(stderr, "Cannot open %s\n", name);
		ABORT("File does not exist");
	    }
	    SUPERLU_FREE(name);
	    for (int e = 0; e < d; ++e) fclose(fp[e]);
	    return 0;
	}
	SUPERLU_FREE(name);
    }
    return 1;
}

static void close_step_files(FILE **fp, int batchCount)
{
    for (int d = 0; d < batchCount; ++d) fclose(fp[d]);
}

int
main (int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    gridinfo3d_t grid;
    int nprow, npcol, npdep;
    int equil, colperm, rowperm, ir, lookahead, gmres = 0;
    int iam, info, nrhs;
    char **cpp, c, *suffix;
    extern int cpp_defs ();
    int omp_mpi_level, batchCount = 0, nsteps = 1, checkx = 1;
    int myrank;

    nprow = 1;            /* Default process rows.      */
    npcol = 1;            /* Default process columns.   */
    npdep = 1;            /* replication factor must be power of two */
    nrhs = 1;             /* Number of right-hand side. */
    equil = -1;
    colperm = -1;
    rowperm = -1;
    ir = -1;
    lookahead = -1;

    char postfix[10] = ".dat";
    char *prefix = NULL;

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------ */
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!rank) {
	    printf("The MPI library doesn't provide MPI_THREAD_MULTIPLE \n");
	    printf("\tprovided omp_mpi_level: %d\n", provided);
        }
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
                printf ("\t-b <int>: batch size, systems per time step\n");
                printf ("\t-t <int>: number of time steps (default %d)\n", nsteps);
                printf ("\t-x <int>: 1 = check against the _x_ reference "
			"solution files (default %d)\n", checkx);
                printf ("\t-f <str>: file name suffix (default %s)\n", postfix);
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
            case 'b': batchCount = atoi(*cpp);
                      break;
            case 't': nsteps = atoi(*cpp);
                      break;
            case 'x': checkx = atoi(*cpp);
                      break;
            case 'e': equil = atoi(*cpp);
                      break;
            case 'p': rowperm = atoi(*cpp);
                      break;
            case 'q': colperm = atoi(*cpp);
                      break;
            case 'g': gmres = atoi(*cpp);
                      break;
            case 'i': ir = atoi(*cpp);
                      break;
            case 's': nrhs = atoi(*cpp);
                      break;
            case 'l': lookahead = atoi(*cpp);
                      break;
            case 'f': strncpy(postfix, *cpp, sizeof(postfix) - 1);
                      postfix[sizeof(postfix) - 1] = '\0';
                      break;
            }
        }
        else
        {   /* Last arg is the time-step deck prefix. */
            prefix = *cpp;
            break;
        }
    }

    if ( prefix == NULL ) ABORT("Missing the time-step deck prefix; see -h");
    if ( batchCount <= 0 ) ABORT("Missing -b <batchCount>; see -h");

    /* <base> is the part of the prefix after the last '/'. */
    char *base = strrchr(prefix, '/');
    base = ( base ? base + 1 : prefix );

    /* suffix is the postfix without its leading '.' */
    suffix = ( postfix[0] == '.' ? &postfix[1] : postfix );

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
    superlu_gridinit3d (MPI_COMM_WORLD, nprow, npcol, npdep, &grid);
#ifdef GPU_ACC
    int superlu_acc_offload = get_acc_offload(&options);
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
    if(grid.iam==0) {
	MPI_Query_thread(&omp_mpi_level);
	switch (omp_mpi_level) {
	case MPI_THREAD_SINGLE:
	    printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
	    break;
	case MPI_THREAD_FUNNELED:
	    printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
	    break;
	case MPI_THREAD_SERIALIZED:
	    printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
	    break;
	case MPI_THREAD_MULTIPLE:
	    printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
	    break;
	}
        fflush(stdout);
    }

    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if (iam == -1)     goto out;
    if (!iam) {
	int v_major, v_minor, v_bugfix;
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);
	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);
	printf("Time-step deck prefix:\t%s\n", prefix);
	printf("3D process grid: %d X %d X %d\n", nprow, npcol, npdep);
	printf("batchCount %d, time steps %d\n", batchCount, nsteps);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter main()");
#endif

    set_default_options_dist (&options);
    options.Algo3d = YES;
    options.IterRefine = NOREFINE;

    options.batchCount = batchCount;
    if (equil != -1) options.Equil = equil;
    if (rowperm != -1) options.RowPerm = rowperm;
    if (colperm != -1) options.ColPerm = colperm;
    if (ir != -1) options.IterRefine = ir;
    options.UseGMRES = gmres;
    if (lookahead != -1) options.num_lookaheads = lookahead;

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    /* ------------------------------------------------------------
       ALLOCATE THE BATCH METADATA THAT LIVES ACROSS ALL TIME STEPS.
       ------------------------------------------------------------ */
    handle_t *SparseMatrix_handles = SUPERLU_MALLOC( batchCount * sizeof(handle_t) );
    handle_t Fstate[1] = {0}; /* carries the state reused across steps */
    handle_t *F = Fstate;
    double **RHSptr = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );
    double **xtrues = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );
    int *ldRHS = int32Malloc_dist(batchCount);
    int *ldX = int32Malloc_dist(batchCount);
    int *md = int32Malloc_dist(batchCount);
    int *nd = int32Malloc_dist(batchCount);
    int *nnzd = int32Malloc_dist(batchCount);

    /* These carry the preprocessing from step 0 into every later step. */
    double **ReqPtr = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );
    double **CeqPtr = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );
    int **RpivPtr = (int **) SUPERLU_MALLOC( batchCount * sizeof(int *) );
    int **CpivPtr = (int **) SUPERLU_MALLOC( batchCount * sizeof(int *) );
    DiagScale_t *DiagScale = (DiagScale_t *) SUPERLU_MALLOC( batchCount * sizeof(DiagScale_t) );
    double **Xptr = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );
    double **Berrs = (double **) SUPERLU_MALLOC( batchCount * sizeof(double *) );

    FILE **fpA = (FILE **) SUPERLU_MALLOC( batchCount * sizeof(FILE *) );
    FILE **fpB = (FILE **) SUPERLU_MALLOC( batchCount * sizeof(FILE *) );
    FILE **fpX = (FILE **) SUPERLU_MALLOC( batchCount * sizeof(FILE *) );


    double t_setup = 0.0, t_solve = 0.0;

    /* ------------------------------------------------------------
       MARCH THROUGH THE TIME STEPS.
       ------------------------------------------------------------ */
    for (int t = 0; t < nsteps; ++t) {

	double t0 = SuperLU_timer_();

	open_step_files(fpA, prefix, base, t, "", batchCount, postfix, 1);
	open_step_files(fpB, prefix, base, t, "rhs_", batchCount, postfix, 1);
	/* The reference solutions are optional; some steps of the EMT decks
	   ship no _x_ files, and a missing one only disables the forward-error
	   report for that step. */
	int havex = checkx &&
	    open_step_files(fpX, prefix, base, t, "x_", batchCount, postfix, 0);

	dcreate_batch_systems_rhsfile(SparseMatrix_handles, batchCount, nrhs,
				      RHSptr, ldRHS, xtrues, ldX,
				      fpA, fpB, (havex ? fpX : NULL),
				      suffix, &grid);

	close_step_files(fpA, batchCount);
	close_step_files(fpB, batchCount);
	if ( havex ) close_step_files(fpX, batchCount);

	for (int d = 0; d < batchCount; ++d) {
	    SuperMatrix *Ad = (SuperMatrix *) SparseMatrix_handles[d];
	    NCformat *Adstore = Ad->Store;
	    md[d] = Ad->nrow;
	    nd[d] = Ad->ncol;
	    nnzd[d] = Adstore->nnz;
	}

	if ( t == 0 ) {
	    /* Per-system output that persists across the whole run: the
	       permutations computed at step 0 are the ones every later step
	       reuses, so they must not be reallocated per step. */
	    for (int d = 0; d < batchCount; ++d) {
		DiagScale[d] = NOEQUIL;
		RpivPtr[d] = int32Malloc_dist(md[d]);
		CpivPtr[d] = int32Malloc_dist(nd[d]);
		Xptr[d] = doubleMalloc_dist( nd[d] * nrhs );
		Berrs[d] = doubleMalloc_dist( nrhs );
	    }
	    options.Fact = DOFACT;
	} else {
	    options.Fact = SamePattern_SameRowPerm;
	}

	t_setup += SuperLU_timer_() - t0;

	PStatInit (&stat);
	t0 = SuperLU_timer_();

	pdgssvx3d_csc_vbatch(&options, batchCount,
			      md, nd, nnzd, nrhs, SparseMatrix_handles,
			      RHSptr, ldRHS, ReqPtr, CeqPtr, RpivPtr, CpivPtr,
			      DiagScale, F, Xptr, ldX, Berrs, &grid, &stat, &info);

	t_solve += SuperLU_timer_() - t0;
	PStatFree (&stat);

	if ( info ) {
	    if ( !iam ) printf("ERROR: INFO = %d returned at step %d\n", info, t);
	    dbatch_systems_free(SparseMatrix_handles, batchCount, RHSptr, xtrues);
	    break;
	}

	if ( !iam ) {
	    printf("**** Step %d: backward / forward errors ****\n", t);
	    for (int d = 0; d < batchCount; ++d) {
		/* Forward error vs. the reference solution read from disk:
		     componentwise  max_i |x - xtrue|_i / |xtrue|_i
		     normwise       ||x - xtrue||_inf / ||xtrue||_inf
		   (first RHS column).  Only meaningful when this step shipped
		   reference solutions. */
		double ferr_cw = 0.0, dxmax = 0.0, xtmax = 0.0;
		if ( havex ) {
		    double *xc = Xptr[d], *xt = xtrues[d];
		    for (int i = 0; i < nd[d]; ++i) {
			double diff = fabs(xc[i] - xt[i]);
			double axt  = fabs(xt[i]);
			if ( axt > 0.0 && diff/axt > ferr_cw ) ferr_cw = diff/axt;
			if ( diff > dxmax ) dxmax = diff;
			if ( axt  > xtmax ) xtmax = axt;
		    }
		    printf("\tSystem %d: Berr = %e   Ferr_cwise = %e   Ferr_norm = %e\n",
			   d, Berrs[d][0], ferr_cw,
			   (xtmax > 0.0 ? dxmax/xtmax : 0.0));
		} else {
		    printf("\tSystem %d: Berr = %e\n", d, Berrs[d][0]);
		}
	    }
	    fflush(stdout);
	}

	/* The matrices were overwritten by Pc*Pr*R*A*C; the next step reads a
	   fresh copy in the original structure. */
	dbatch_systems_free(SparseMatrix_handles, batchCount, RHSptr, xtrues);

    } /* end for t ... nsteps */

    if ( !iam ) {
	printf("**************************************************\n");
	printf("Time steps                 : %d\n", nsteps);
	printf("Batch Setup time = %12.6f  (read + assemble, all steps)\n", t_setup);
	printf("Batch Solve time = %12.6f  (all steps)\n", t_solve);
	printf("Batch Solve time per step = %12.6f\n",
	       (nsteps > 0 ? t_solve / nsteps : 0.0));
	printf("**************************************************\n");
	fflush(stdout);
    }

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------ */
    dvbatch_free(F);

    for (int d = 0; d < batchCount; ++d) {
	if ( DiagScale[d] == ROW || DiagScale[d] == BOTH ) SUPERLU_FREE(ReqPtr[d]);
	if ( DiagScale[d] == COL || DiagScale[d] == BOTH ) SUPERLU_FREE(CeqPtr[d]);
	SUPERLU_FREE(RpivPtr[d]);
	SUPERLU_FREE(CpivPtr[d]);
	SUPERLU_FREE(Xptr[d]);
	SUPERLU_FREE(Berrs[d]);
    }
    SUPERLU_FREE(SparseMatrix_handles);
    SUPERLU_FREE(RHSptr);
    SUPERLU_FREE(xtrues);
    SUPERLU_FREE(ldRHS);
    SUPERLU_FREE(ldX);
    SUPERLU_FREE(md);
    SUPERLU_FREE(nd);
    SUPERLU_FREE(nnzd);
    SUPERLU_FREE(ReqPtr);
    SUPERLU_FREE(CeqPtr);
    SUPERLU_FREE(RpivPtr);
    SUPERLU_FREE(CpivPtr);
    SUPERLU_FREE(DiagScale);
    SUPERLU_FREE(Xptr);
    SUPERLU_FREE(Berrs);
    SUPERLU_FREE(fpA);
    SUPERLU_FREE(fpB);
    SUPERLU_FREE(fpX);

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

    return 0;
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
