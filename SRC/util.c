/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Utilities functions
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * February 1, 2003
 *
 * Modified: March 31, 2013
 *           January 29, 2018
 * </pre>
 */

#include <math.h>
#include <unistd.h>
#include "superlu_ddefs.h"

/*! \brief Deallocate the structure pointing to the actual storage of the matrix. */
void Destroy_SuperMatrix_Store_dist(SuperMatrix *A)
{
    SUPERLU_FREE(A->Store);
}

void Destroy_CompCol_Matrix_dist(SuperMatrix *A)
{
    NCformat *Astore = A->Store;
    SUPERLU_FREE(Astore->rowind);
    SUPERLU_FREE(Astore->colptr);
    if (Astore->nzval)
        SUPERLU_FREE(Astore->nzval);
    SUPERLU_FREE(Astore);
}

void Destroy_CompRowLoc_Matrix_dist(SuperMatrix *A)
{
    NRformat_loc *Astore = A->Store;
    SUPERLU_FREE(Astore->rowptr);
    SUPERLU_FREE(Astore->colind);
    SUPERLU_FREE(Astore->nzval);
    SUPERLU_FREE(Astore);
}

void Destroy_CompRow_Matrix_dist(SuperMatrix *A)
{
    SUPERLU_FREE(((NRformat *)A->Store)->rowptr);
    SUPERLU_FREE(((NRformat *)A->Store)->colind);
    SUPERLU_FREE(((NRformat *)A->Store)->nzval);
    SUPERLU_FREE(A->Store);
}

void Destroy_SuperNode_Matrix_dist(SuperMatrix *A)
{
    SUPERLU_FREE(((SCformat *)A->Store)->rowind);
    SUPERLU_FREE(((SCformat *)A->Store)->rowind_colptr);
    SUPERLU_FREE(((SCformat *)A->Store)->nzval);
    SUPERLU_FREE(((SCformat *)A->Store)->nzval_colptr);
    SUPERLU_FREE(((SCformat *)A->Store)->col_to_sup);
    SUPERLU_FREE(((SCformat *)A->Store)->sup_to_col);
    SUPERLU_FREE(A->Store);
}

/*! \brief A is of type Stype==NCP */
void Destroy_CompCol_Permuted_dist(SuperMatrix *A)
{
    SUPERLU_FREE(((NCPformat *)A->Store)->colbeg);
    SUPERLU_FREE(((NCPformat *)A->Store)->colend);
    SUPERLU_FREE(A->Store);
}

/*! \brief A is of type Stype==DN */
void Destroy_Dense_Matrix_dist(SuperMatrix *A)
{
    DNformat *Astore = A->Store;
    SUPERLU_FREE(Astore->nzval);
    SUPERLU_FREE(A->Store);
}

/*! \brief
 *
 * <pre>
 * Count the total number of nonzeros in factors L and U,  and in the 
 * symmetrically reduced L. 
 * </pre>
 */
void countnz_dist(const int_t n, int_t *xprune,
                  int_t *nnzL, int_t *nnzU,
                  Glu_persist_t *Glu_persist, Glu_freeable_t *Glu_freeable)
{
    int_t fnz, fsupc, i, j, nsuper;
    int_t jlen, irep;
    long long int nnzL0;
    int_t *supno, *xsup, *xlsub, *xusub, *usub;

    supno = Glu_persist->supno;
    xsup = Glu_persist->xsup;
    xlsub = Glu_freeable->xlsub;
    xusub = Glu_freeable->xusub;
    usub = Glu_freeable->usub;
    *nnzL = 0;
    *nnzU = 0;
    nnzL0 = 0;
    nsuper = supno[n];

    if (n <= 0)
        return;

    /* 
     * For each supernode in L.
     */
    for (i = 0; i <= nsuper; i++)
    {
        fsupc = xsup[i];
        jlen = xlsub[fsupc + 1] - xlsub[fsupc];

        for (j = fsupc; j < xsup[i + 1]; j++)
        {
            *nnzL += jlen;
            *nnzU += j - fsupc + 1;
            jlen--;
        }
        irep = xsup[i + 1] - 1;
        nnzL0 += xprune[irep] - xlsub[irep];
    }

    /* printf("\tNo of nonzeros in symm-reduced L = %ld\n", nnzL0);*/

    /* For each column in U. */
    for (j = 0; j < n; ++j)
    {
        for (i = xusub[j]; i < xusub[j + 1]; ++i)
        {
            fnz = usub[i];
            fsupc = xsup[supno[fnz] + 1];
            *nnzU += fsupc - fnz;
        }
    }
#if ( PRNTlevel>=2 )
    printf("\tNo of nonzeros in symm-reduced L = " IFMT ", nnzL " IFMT ", nnzU " IFMT "\n",
	   nnzL0, *nnzL, *nnzU);
#endif
    
}

/*! \brief
 *
 * <pre>
 * Fix up the data storage lsub for L-subscripts. It removes the subscript
 * sets for structural pruning,	and applies permuation to the remaining
 * subscripts.
 * </pre>
 */
int64_t
fixupL_dist(const int_t n, const int_t *perm_r,
            Glu_persist_t *Glu_persist, Glu_freeable_t *Glu_freeable)
{
    register int_t nsuper, fsupc, nextl, i, j, k, jstrt;
    register long long int lsub_size;
    int_t *xsup, *lsub, *xlsub;

    if (n <= 1)
        return 0;

    xsup = Glu_persist->xsup;
    lsub = Glu_freeable->lsub;
    xlsub = Glu_freeable->xlsub;
    nextl = 0;
    nsuper = (Glu_persist->supno)[n];
    lsub_size = xlsub[n];

    /* 
     * For each supernode ...
     */
    for (i = 0; i <= nsuper; i++)
    {
        fsupc = xsup[i];
        jstrt = xlsub[fsupc];
        xlsub[fsupc] = nextl;
        for (j = jstrt; j < xlsub[fsupc + 1]; j++)
        {
            lsub[nextl] = perm_r[lsub[j]]; /* Now indexed into P*A */
            nextl++;
        }
        for (k = fsupc + 1; k < xsup[i + 1]; k++)
            xlsub[k] = nextl; /* Other columns in supernode i */
    }

    xlsub[n] = nextl;
    return lsub_size;
}

/*! \brief Set the default values for the options argument.
 */
void set_default_options_dist(superlu_dist_options_t *options)
{
    options->Fact = DOFACT;
    options->Equil = YES;
    options->ParSymbFact = NO;
#ifdef HAVE_PARMETIS
    options->ColPerm = METIS_AT_PLUS_A;
#else
    options->ColPerm = MMD_AT_PLUS_A;
#endif
    options->RowPerm = LargeDiag_MC64;
    options->ReplaceTinyPivot = NO;
    options->IterRefine = SLU_DOUBLE;
    options->Trans = NOTRANS;
    options->SolveInitialized = NO;
    options->RefineInitialized = NO;
    options->PrintStat = YES;
    options->lookahead_etree = NO;
    options->num_lookaheads = 10;
    options->superlu_maxsup = 256;
    options->superlu_relax = 60;
    strcpy(options->superlu_rankorder, "Z"); 
    strcpy(options->superlu_lbs, "GD");
    options->superlu_acc_offload = 1;
    options->superlu_n_gemm = 5000;
    options->superlu_max_buffer_size = 256000000;
    options->superlu_num_gpu_streams = 8;
    options->SymPattern = NO;
    options->Algo3d = NO;
#ifdef SLU_HAVE_LAPACK
    options->DiagInv = YES;
#else
    options->DiagInv = NO;
#endif
    options->Use_TensorCore    = NO;
}

/*! \brief Print the options setting.
 */
void print_options_dist(superlu_dist_options_t *options)
{
    if (options->PrintStat == NO)
        return;

    printf("**************************************************\n");
    printf(".. options:\n");
    printf("**    Fact                      : %4d\n", options->Fact);
    printf("**    Equil                     : %4d\n", options->Equil);
    printf("**    DiagInv                   : %4d\n", options->DiagInv);
    printf("**    ParSymbFact               : %4d\n", options->ParSymbFact);
    printf("**    ColPerm                   : %4d\n", options->ColPerm);
    printf("**    RowPerm                   : %4d\n", options->RowPerm);
    printf("**    ReplaceTinyPivot          : %4d\n", options->ReplaceTinyPivot);
    printf("**    IterRefine                : %4d\n", options->IterRefine);
    printf("**    Trans                     : %4d\n", options->Trans);
    printf("**    SymPattern                : %4d\n", options->SymPattern);
    printf("**    lookahead_etree           : %4d\n", options->lookahead_etree);
    printf("**    Use_TensorCore            : %4d\n", options->Use_TensorCore);
    printf("**    Use 3D algorithm          : %4d\n", options->Algo3d);
    printf("**    num_lookaheads            : %4d\n", options->num_lookaheads);
    printf("** parameters that can be altered by environment variables:\n");
    printf("**    superlu_relax             : %4d\n", sp_ienv_dist(2, options));
    printf("**    superlu_maxsup            : %4d\n", sp_ienv_dist(3, options));
    printf("**    min GEMM m*k*n to use GPU : %d\n", sp_ienv_dist(7, options));
    printf("**    GPU buffer size           : %10d\n", sp_ienv_dist(8, options));
    printf("**    GPU streams               : %4d\n", sp_ienv_dist(9, options));
    printf("**    estimated fill ratio      : %4d\n", sp_ienv_dist(6, options));
    printf("**************************************************\n");
}

/*! \brief Print the blocking parameters.
 */
void print_sp_ienv_dist(superlu_dist_options_t *options)
{
    if (options->PrintStat == NO)
        return;

    printf("**************************************************\n");
    printf(".. blocking parameters from sp_ienv():\n");
    printf("**    relaxation                 : %d\n", sp_ienv_dist(2, options));
    printf("**    max supernode              : %d\n", sp_ienv_dist(3, options));
    printf("**    estimated fill ratio       : %d\n", sp_ienv_dist(6, options));
    printf("**    min GEMM m*k*n to use GPU  : %d\n", sp_ienv_dist(7, options));
    printf("**************************************************\n");
}

void pxgstrs_finalize(pxgstrs_comm_t *gstrs_comm)
{
    SUPERLU_FREE(gstrs_comm->B_to_X_SendCnt);
    SUPERLU_FREE(gstrs_comm->X_to_B_SendCnt);
    SUPERLU_FREE(gstrs_comm->ptr_to_ibuf);
    SUPERLU_FREE(gstrs_comm);
}

/*! \brief Diagnostic print of segment info after panel_dfs().
 */
void print_panel_seg_dist(int_t n, int_t w, int_t jcol, int_t nseg,
                          int_t *segrep, int_t *repfnz)
{
    int j, k;

    for (j = jcol; j < jcol + w; j++)
    {
        printf("\tcol %d:\n", j);
        for (k = 0; k < nseg; k++)
            printf("\t\tseg %d, segrep %d, repfnz %d\n", k,
                   (int)segrep[k], (int)repfnz[(j - jcol) * n + segrep[k]]);
    }
}

void PStatInit(SuperLUStat_t *stat)
{
    register int_t i;

    if (!(stat->utime = SUPERLU_MALLOC(NPHASES * sizeof(double))))
        ABORT("Malloc fails for stat->utime[]");
    if (!(stat->ops = (flops_t *)SUPERLU_MALLOC(NPHASES * sizeof(flops_t))))
        ABORT("SUPERLU_MALLOC fails for stat->ops[]");
    for (i = 0; i < NPHASES; ++i)
    {
        stat->utime[i] = 0.;
        stat->ops[i] = 0.;
    }
    stat->TinyPivots = stat->RefineSteps = 0;
    stat->current_buffer = stat->peak_buffer = 0.0;
    stat->gpu_buffer = 0.0;
}

void PStatPrint(superlu_dist_options_t *options, SuperLUStat_t *stat, gridinfo_t *grid)
{
    double *utime = stat->utime;
    flops_t *ops = stat->ops;
    int_t iam = grid->iam;
    flops_t flopcnt, factflop, solveflop;

    if (options->PrintStat == NO)
        return;

    if (!iam && options->Fact != FACTORED)
    {
        printf("**************************************************\n");
        printf("**** Time (seconds) ****\n");
        if ( options->Equil != NO )
	    printf("\tEQUIL time         %8.3f\n", utime[EQUIL]);
	if ( options->RowPerm != NOROWPERM )
	    printf("\tROWPERM time       %8.3f\n", utime[ROWPERM]);
	if ( options->ColPerm != NATURAL )
	    printf("\tCOLPERM time       %8.3f\n", utime[COLPERM]);
        printf("\tSYMBFACT time      %8.3f\n", utime[SYMBFAC]);
	printf("\tDISTRIBUTE time    %8.3f\n", utime[DIST]);
    }

    MPI_Reduce(&ops[FACT], &flopcnt, 1, MPI_FLOAT, MPI_SUM,
               0, grid->comm);
    factflop = flopcnt;
    if ( !iam && options->Fact != FACTORED ) {
	printf("\tFACTOR time        %8.3f\n", utime[FACT]);
	if ( utime[FACT] != 0.0 )
	    printf("\tFactor flops\t%e\tMflops \t%8.2f\n",
		   flopcnt,
		   flopcnt*1e-6/utime[FACT]);
    }

    MPI_Reduce(&ops[SOLVE], &flopcnt, 1, MPI_FLOAT, MPI_SUM,
               0, grid->comm);
    solveflop = flopcnt;
    if (!iam)
    {
        printf("\tSOLVE time         %8.3f\n", utime[SOLVE]);
        if (utime[SOLVE] != 0.0)
            printf("\tSolve flops\t%e\tMflops \t%8.2f\n",
                   flopcnt,
                   flopcnt * 1e-6 / utime[SOLVE]);
        if (options->IterRefine != NOREFINE)
        {
            printf("\tREFINEMENT time    %8.3f\tSteps%8d\n\n",
                   utime[REFINE], stat->RefineSteps);
        }
        printf("**************************************************\n");
    }

    double *utime1, *utime2, *utime3, *utime4;
    flops_t *ops1;
#if (PROFlevel >= 1)
    fflush(stdout);
    MPI_Barrier(grid->comm);

    {
        int_t i, P = grid->nprow * grid->npcol;
        flops_t b, maxflop;

        if (!iam)
            utime1 = doubleMalloc_dist(P);
        if (!iam)
            utime2 = doubleMalloc_dist(P);
        if (!iam)
            utime3 = doubleMalloc_dist(P);
        if (!iam)
            utime4 = doubleMalloc_dist(P);
        if (!iam)
            ops1 = (flops_t *)SUPERLU_MALLOC(P * sizeof(flops_t));

        // fflush(stdout);
        // if ( !iam ) printf("\n.. Tree max sizes:\tbtree\trtree\n");
        // fflush(stdout);
        // sleep(2.0);
        // MPI_Barrier( grid->comm );
        // for (i = 0; i < P; ++i) {
        // if ( iam == i) {
        // printf("\t\t%d %5d %5d\n", iam, stat->MaxActiveBTrees,stat->MaxActiveRTrees);
        // fflush(stdout);
        // }
        // MPI_Barrier( grid->comm );
        // }

        // sleep(2.0);

        MPI_Barrier(grid->comm);

        if (!iam)
            printf("\n.. FACT time breakdown:\tcomm\ttotal\n");

        MPI_Gather(&utime[COMM], 1, MPI_DOUBLE, utime1, 1, MPI_DOUBLE, 0, grid->comm);
        MPI_Gather(&utime[FACT], 1, MPI_DOUBLE, utime2, 1, MPI_DOUBLE, 0, grid->comm);
        if (!iam)
            for (i = 0; i < P; ++i)
            {
                printf("\t\t(%d)%8.2f%8.2f\n", i, utime1[i], utime2[i]);
            }
        fflush(stdout);
        MPI_Barrier(grid->comm);

        if (!iam)
            printf("\n.. FACT ops distribution:\n");
        MPI_Gather(&ops[FACT], 1, MPI_FLOAT, ops1, 1, MPI_FLOAT, 0, grid->comm);

        if (!iam)
            for (i = 0; i < P; ++i)
            {
                printf("\t\t(%d)\t%e\n", i, ops1[i]);
            }
        fflush(stdout);
        MPI_Barrier(grid->comm);

        MPI_Reduce(&ops[FACT], &maxflop, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm);

        if (!iam)
        {
            b = factflop / P / maxflop;
            printf("\tFACT load balance: %.2f\n", b);
        }
        fflush(stdout);
        MPI_Barrier(grid->comm);

        if (!iam)
            printf("\n.. SOLVE time breakdown:\tcommL \tgemmL\ttrsmL\ttotal\n");

        MPI_Gather(&utime[SOL_COMM], 1, MPI_DOUBLE, utime1, 1, MPI_DOUBLE, 0, grid->comm);
        MPI_Gather(&utime[SOL_GEMM], 1, MPI_DOUBLE, utime2, 1, MPI_DOUBLE, 0, grid->comm);
        MPI_Gather(&utime[SOL_TRSM], 1, MPI_DOUBLE, utime3, 1, MPI_DOUBLE, 0, grid->comm);
        MPI_Gather(&utime[SOL_TOT], 1, MPI_DOUBLE, utime4, 1, MPI_DOUBLE, 0, grid->comm);
        if (!iam)
            for (i = 0; i < P; ++i)
            {
                printf("\t\t\t%d%10.5f%10.5f%10.5f%10.5f\n", i, utime1[i], utime2[i], utime3[i], utime4[i]);
            }
        fflush(stdout);
        MPI_Barrier(grid->comm);

        if (!iam)
            printf("\n.. SOLVE ops distribution:\n");
        MPI_Gather(&ops[SOLVE], 1, MPI_FLOAT, ops1, 1, MPI_FLOAT, 0, grid->comm);
        if (!iam)
            for (i = 0; i < P; ++i)
            {
                printf("\t\t%d\t%e\n", i, ops1[i]);
            }
        MPI_Reduce(&ops[SOLVE], &maxflop, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
        if (!iam)
        {
            b = solveflop / P / maxflop;
            printf("\tSOLVE load balance: %.2f\n", b);
            fflush(stdout);
        }
    }

    if (!iam)
    {
        SUPERLU_FREE(utime1);
        SUPERLU_FREE(utime2);
        SUPERLU_FREE(utime3);
        SUPERLU_FREE(utime4);
        SUPERLU_FREE(ops1);
    }

#endif

    /*  if ( !iam ) fflush(stdout);  CRASH THE SYSTEM pierre.  */
}

void PStatFree(SuperLUStat_t *stat)
{
    SUPERLU_FREE(stat->utime);
    SUPERLU_FREE(stat->ops);
}

/*! \brief Fills an integer array with a given value.
 */
void ifill_dist(int_t *a, int_t alen, int_t ival)
{
    register int_t i;
    for (i = 0; i < alen; i++)
        a[i] = ival;
}

void get_diag_procs(int_t n, Glu_persist_t *Glu_persist, gridinfo_t *grid,
                    int_t *num_diag_procs, int_t **diag_procs, int_t **diag_len)
{
    int_t i, j, k, knsupc, nprow, npcol, nsupers, pkk;
    int_t *xsup;

    i = j = *num_diag_procs = pkk = 0;
    nprow = grid->nprow;
    npcol = grid->npcol;
    nsupers = Glu_persist->supno[n - 1] + 1;
    xsup = Glu_persist->xsup;

    do
    {
        ++(*num_diag_procs);
        ++i;
	i = (i) % nprow;
        ++j;
	j = (j) % npcol;
        pkk = PNUM(i, j, grid);
    } while (pkk != 0); /* Until wrap back to process 0 */
    if (!(*diag_procs = intMalloc_dist(*num_diag_procs)))
        ABORT("Malloc fails for diag_procs[]");
    if (!(*diag_len = intCalloc_dist(*num_diag_procs)))
        ABORT("Calloc fails for diag_len[]");
    for (i = j = k = 0; k < *num_diag_procs; ++k)
    {
        pkk = PNUM(i, j, grid);
        (*diag_procs)[k] = pkk;
	++i;
        i = (i) % nprow;
	++j;
        j = (j) % npcol;
    }
    for (k = 0; k < nsupers; ++k)
    {
        knsupc = SuperSize(k);
        i = k % *num_diag_procs;
        (*diag_len)[i] += knsupc;
    }
}

/*! \brief Get the statistics of the supernodes 
 */
#define NBUCKS 10
static int max_sup_size;

void super_stats_dist(int_t nsuper, int_t *xsup)
{
    register int nsup1 = 0;
    int_t i, isize, whichb, bl, bh;
    int_t bucket[NBUCKS];

    max_sup_size = 0;

    for (i = 0; i <= nsuper; i++)
    {
        isize = xsup[i + 1] - xsup[i];
        if (isize == 1)
            nsup1++;
        if (max_sup_size < isize)
            max_sup_size = isize;
    }

    printf("    Supernode statistics:\n\tno of super = %d\n", (int)nsuper + 1);
    printf("\tmax supernode size = %d\n", max_sup_size);
    printf("\tno of size 1 supernodes = %d\n", nsup1);

    /* Histogram of the supernode sizes */
    ifill_dist(bucket, NBUCKS, 0);

    for (i = 0; i <= nsuper; i++)
    {
        isize = xsup[i + 1] - xsup[i];
        whichb = (float)isize / max_sup_size * NBUCKS;
        if (whichb >= NBUCKS)
            whichb = NBUCKS - 1;
        bucket[whichb]++;
    }

    printf("\tHistogram of supernode sizes:\n");
    for (i = 0; i < NBUCKS; i++)
    {
        bl = (float)i * max_sup_size / NBUCKS;
        bh = (float)(i + 1) * max_sup_size / NBUCKS;
        printf("\tsnode: %d-%d\t\t%d\n", (int)bl + 1, (int)bh, (int)bucket[i]);
    }
}

/*! \brief Check whether repfnz[] == EMPTY after reset.
 */
void check_repfnz_dist(int_t n, int_t w, int_t jcol, int_t *repfnz)
{
    int jj, k;

    for (jj = jcol; jj < jcol + w; jj++)
        for (k = 0; k < n; k++)
            if (repfnz[(jj - jcol) * n + k] != EMPTY)
            {
                fprintf(stderr, "col %d, repfnz_col[%d] = %d\n",
                        jj, k, (int)repfnz[(jj - jcol) * n + k]);
                ABORT("check_repfnz_dist");
            }
}

void PrintInt10(char *name, int_t len, int_t *x)
{
    register int_t i;

    printf("%10s:", name);
    for (i = 0; i < len; ++i)
    {
        if (i % 10 == 0)
            printf("\n\t[" IFMT "-" IFMT "]", i, i + 9);
        printf(IFMT, x[i]);
    }
    printf("\n");
}

void PrintInt32(char *name, int len, int *x)
{
    register int i;

    printf("%10s:", name);
    for (i = 0; i < len; ++i)
    {
        if (i % 10 == 0)
            printf("\n\t[%2d-%2d]", i, i + 9);
        printf("%6d", x[i]);
    }
    printf("\n");
}

int file_PrintInt10(FILE *fp, char *name, int_t len, int_t *x)
{
    register int_t i;

    fprintf(fp, "%10s:", name);
    for (i = 0; i < len; ++i)
    {
        if (i % 10 == 0)
            fprintf(fp, "\n\t[" IFMT "-" IFMT "]", i, i + 9);
        fprintf(fp, IFMT, x[i]);
    }
    fprintf(fp, "\n");
    return 0;
}

int file_PrintInt32(FILE *fp, char *name, int len, int *x)
{
    register int i;

    fprintf(fp, "%10s:", name);
    for (i = 0; i < len; ++i)
    {
        if (i % 10 == 0)
            fprintf(fp, "\n\t[%2d-%2d]", i, i + 9);
        fprintf(fp, "%6d", x[i]);
    }
    fprintf(fp, "\n");
    return 0;
}

int_t CheckZeroDiagonal(int_t n, int_t *rowind, int_t *colbeg, int_t *colcnt)
{
    register int_t i, j, zd, numzd = 0;

    for (j = 0; j < n; ++j)
    {
        zd = 0;
        for (i = colbeg[j]; i < colbeg[j] + colcnt[j]; ++i)
        {
            /*if ( iperm[rowind[i]] == j ) zd = 1;*/
            if (rowind[i] == j)
            {
                zd = 1;
                break;
            }
        }
        if (zd == 0)
        {
#if (PRNTlevel >= 2)
            printf(".. Diagonal of column %d is zero.\n", j);
#endif
            ++numzd;
        }
    }

    return numzd;
}

/* --------------------------------------------------------------------------- */
void isort(int_t N, int_t *ARRAY1, int_t *ARRAY2)
{
    /*
 * Purpose
 * =======
 * Use quick sort algorithm to sort ARRAY1 and ARRAY2 in the increasing
 * order of ARRAY1.
 *
 * Arguments
 * =========
 * N       (input) INTEGER
 *          On entry, specifies the size of the arrays.
 *
 * ARRAY1  (input/output) integer array of length N
 *          On entry, contains the array to be sorted.
 *          On exit, contains the sorted array.
 *
 * ARRAY2  (input/output) integer array of length N
 *          On entry, contains the array to be sorted.
 *          On exit, contains the sorted array.
 */
    int_t IGAP, I, J;
    int_t TEMP;
    IGAP = N / 2;
    while (IGAP > 0)
    {
        for (I = IGAP; I < N; I++)
        {
            J = I - IGAP;
            while (J >= 0)
            {
                if (ARRAY1[J] > ARRAY1[J + IGAP])
                {
                    TEMP = ARRAY1[J];
                    ARRAY1[J] = ARRAY1[J + IGAP];
                    ARRAY1[J + IGAP] = TEMP;
                    TEMP = ARRAY2[J];
                    ARRAY2[J] = ARRAY2[J + IGAP];
                    ARRAY2[J + IGAP] = TEMP;
                    J = J - IGAP;
                }
                else
                {
                    break;
                }
            }
        }
        IGAP = IGAP / 2;
    }
}

void isort1(int_t N, int_t *ARRAY)
{
    /*
 * Purpose
 * =======
 * Use quick sort algorithm to sort ARRAY in increasing order.
 *
 * Arguments
 * =========
 * N       (input) INTEGER
 *          On entry, specifies the size of the arrays.
 *
 * ARRAY   (input/output) DOUBLE PRECISION ARRAY of LENGTH N
 *          On entry, contains the array to be sorted.
 *          On exit, contains the sorted array.
 *
 */
    int_t IGAP, I, J;
    int_t TEMP;
    IGAP = N / 2;
    while (IGAP > 0)
    {
        for (I = IGAP; I < N; I++)
        {
            J = I - IGAP;
            while (J >= 0)
            {
                if (ARRAY[J] > ARRAY[J + IGAP])
                {
                    TEMP = ARRAY[J];
                    ARRAY[J] = ARRAY[J + IGAP];
                    ARRAY[J + IGAP] = TEMP;
                    J = J - IGAP;
                }
                else
                {
                    break;
                }
            }
        }
        IGAP = IGAP / 2;
    }
}

/* Only log the memory for the buffer space, excluding the LU factors */
void log_memory(int64_t cur_bytes, SuperLUStat_t *stat)
{
    stat->current_buffer += (float)cur_bytes;
    if (cur_bytes > 0)
    {
        stat->peak_buffer =
            SUPERLU_MAX(stat->peak_buffer, stat->current_buffer);
    }
}

void print_memorylog(SuperLUStat_t *stat, char *msg)
{
    printf("__ %s (MB):\n\tcurrent_buffer : %8.2f\tpeak_buffer : %8.2f\n",
           msg, stat->current_buffer, stat->peak_buffer);
}

int compare_pair(const void *a, const void *b)
{
    return (((struct superlu_pair *)a)->val - ((struct superlu_pair *)b)->val);
}

int get_thread_per_process()
{
    char *ttemp;
    ttemp = getenv("THREAD_PER_PROCESS");

    if (ttemp)
        return atoi(ttemp);
    else
        return 1;
}

#if 0 // not used anymore
int_t get_max_buffer_size()
{
    char *ttemp;
    ttemp = getenv("SUPERLU_MAX_BUFFER_SIZE");
    if (ttemp)
        return atoi(ttemp);
    else
        return 200000000; // 5000000
}
#endif

int_t
get_gpublas_nb ()
{
    char *ttemp;
    ttemp = getenv ("GPUBLAS_NB");
    if (ttemp)
        return atoi(ttemp);
    else
        return 512;     // 64 
}

int_t
get_num_gpu_streams ()
{
    char *ttemp;
    ttemp = getenv ("SUPERLU_NUM_GPU_STREAMS");
    if (ttemp)
        return atoi(ttemp);
    else if (getenv ("NUM_GPU_STREAMS")) 
        return atoi(getenv ("NUM_GPU_STREAMS"));   
    else
        return 8;
}

int_t get_min(int_t *sums, int_t nprocs)
{
    int_t min_ind, min_val;
    min_ind = 0;
    min_val = 2147483647;
    for (int i = 0; i < nprocs; i++)
    {
        if (sums[i] < min_val)
        {
            min_val = sums[i];
            min_ind = i;
        }
    }

    return min_ind;
}

int_t static_partition(struct superlu_pair *work_load, int_t nwl, int_t *partition,
                       int_t ldp, int_t *sums, int_t *counts, int nprocs)
{
    //initialization loop
    for (int i = 0; i < nprocs; ++i)
    {
        counts[i] = 0;
        sums[i] = 0;
    }
    qsort(work_load, nwl, sizeof(struct superlu_pair), compare_pair);
    // for(int i=0;i<nwl;i++)
    for (int i = nwl - 1; i >= 0; i--)
    {
        int_t ind = get_min(sums, nprocs);
        // printf("ind %d\n",ind );
        partition[ldp * ind + counts[ind]] = work_load[i].ind;
        counts[ind]++;
        sums[ind] += work_load[i].val;
    }

    return 0;
}

/*
 * Search for the metadata of the j-th block in a U panel.
 */
void arrive_at_ublock(int_t j,      /* j-th block in a U panel */
                      int_t *iukp,  /* output : point to index[] of j-th block */
                      int_t *rukp,  /* output : point to nzval[] of j-th block */
                      int_t *jb,    /* Global block number of block U(k,j). */
                      int_t *ljb,   /* Local block number of U(k,j). */
                      int_t *nsupc, /* supernode size of destination block */
                      int_t iukp0,  /* input : search starting point */
                      int_t rukp0,
                      int_t *usub,   /* U subscripts */
                      int_t *perm_u, /* permutation vector from static schedule */
                      int_t *xsup,   /* for SuperSize and LBj */
                      gridinfo_t *grid)
{
    int_t jj;
    *iukp = iukp0; /* point to the first block in index[] */
    *rukp = rukp0; /* point to the start of nzval[] */

    /* Sherry -- why always starts from 0 ?? Can continue at 
       the column left from last search.  */
    /* Caveat: There is a permutation perm_u involved for j. That's why
       the search need to restart from 0.  */
#ifdef ISORT
    for (jj = 0; jj < perm_u[j]; jj++) /* perm_u[j] == j */
#else
    for (jj = 0; jj < perm_u[2 * j + 1]; jj++) /* == j */
#endif
    {
        /* Reinitilize the pointers to the beginning of the 
	 * k-th column/row of L/U factors.
	 * usub[] - index array for panel U(k,:)
	 */
        // printf("iukp %d \n",*iukp );
        *jb = usub[*iukp]; /* Global block number of block U(k,jj). */
        // printf("jb %d \n",*jb );
        *nsupc = SuperSize(*jb);
        // printf("nsupc %d \n",*nsupc );
        *iukp += UB_DESCRIPTOR;   /* Start fstnz of block U(k,j). */
        *rukp += usub[*iukp - 1]; /* Jump # of nonzeros in block U(k,jj);
				     Move to block U(k,jj+1) in nzval[] */
        *iukp += *nsupc;
    }

    /* Set the pointers to the beginning of U block U(k,j) */
    *jb = usub[*iukp];     /* Global block number of block U(k,j). */
    *ljb = LBj(*jb, grid); /* Local block number of U(k,j). */
    *nsupc = SuperSize(*jb);
    *iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
}

/*
 * Count the maximum size of U(kk,:) that I own locally.
 * September 28, 2016.
 * Modified December 4, 2018.
 */
int_t num_full_cols_U(
    int_t kk, int_t **Ufstnz_br_ptr, int_t *xsup,
    gridinfo_t *grid, int_t *perm_u,
    int_t *ldu /* max. segment size of nonzero columns in U(kk,:) */
)
{
    int_t lk = LBi(kk, grid);
    int_t *usub = Ufstnz_br_ptr[lk];

    if (usub == NULL)
        return 0; /* code */

    int_t iukp = BR_HEADER; /* Skip header; Pointer to index[] of U(k,:) */
    int_t rukp = 0;         /* Pointer to nzval[] of U(k,:) */
    int_t nub = usub[0];    /* Number of blocks in the block row U(k,:) */

    int_t klst = FstBlockC(kk + 1);
    int_t iukp0 = iukp;
    int_t rukp0 = rukp;
    int_t jb, ljb;
    int_t nsupc;
    int_t full = 1;
    int_t full_Phi = 1;
    int_t temp_ncols = 0;
    int_t segsize;

    *ldu = 0;

    for (int_t j = 0; j < nub; ++j)
    {

        /* Sherry -- no need to search from beginning ?? */
        arrive_at_ublock(
            j, &iukp, &rukp, &jb, &ljb, &nsupc,
            iukp0, rukp0, usub, perm_u, xsup, grid);
        for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
        {
            segsize = klst - usub[jj];
            if (segsize)
                ++temp_ncols;
            if (segsize > *ldu)
                *ldu = segsize;
        }
    }
    return temp_ncols;
}

int_t estimate_bigu_size(
      int_t nsupers,
      int_t **Ufstnz_br_ptr, /* point to U index[] array */
      Glu_persist_t *Glu_persist,
      gridinfo_t* grid, int_t* perm_u, 
      int_t *max_ncols /* Output: Max. number of columns among all U(k,:).
			  This is used for allocating GEMM V buffer.  */
			 )
{
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW(iam, grid);
    int_t mycol = MYCOL(iam, grid);

    int_t *xsup = Glu_persist->xsup;

    int_t ncols = 0; /* Count local number of nonzero columns */
    int_t ldu = 0;   /* Count max. segment size in one row U(k,:) */
    int_t my_max_ldu = 0;
    int_t max_ldu = 0;

    /* Initialize perm_u */
    for (int i = 0; i < nsupers; ++i)
        perm_u[i] = i;

    for (int lk = myrow; lk < nsupers; lk += Pr)
    { /* Go through my block rows */
        ncols = SUPERLU_MAX(ncols, num_full_cols_U(lk, Ufstnz_br_ptr,
                                                   xsup, grid, perm_u, &ldu));
        my_max_ldu = SUPERLU_MAX(ldu, my_max_ldu);
    }
#if 0
	my_max_ldu = my_max_ldu*8;  //YL: 8 is a heuristic number  
#endif
	
    /* Need U buffer size large enough to hold all U(k,:) transferred from
       other processes. */
    MPI_Allreduce(&my_max_ldu, &max_ldu, 1, mpi_int_t, MPI_MAX, grid->cscp.comm);
    MPI_Allreduce(&ncols, max_ncols, 1, mpi_int_t, MPI_MAX, grid->cscp.comm);

#if (PRNTlevel >= 1)
    if (iam == 0)
    {
        printf("max_ncols %d,  max_ldu %d, bigu_size " IFMT "\n",
               (int)*max_ncols, (int)max_ldu, max_ldu * (*max_ncols));
        fflush(stdout);
    }
#endif

    return (max_ldu * (*max_ncols));
}

void quickSort(int_t *a, int_t l, int_t r, int_t dir)
{
    int_t j;

    if (l < r)
    {
        // divide and conquer
        j = partition(a, l, r, dir);
        quickSort(a, l, j - 1, dir);
        quickSort(a, j + 1, r, dir);
    }
}

int_t partition(int_t *a, int_t l, int_t r, int_t dir)
{
    int_t pivot, i, j, t;
    pivot = a[l];
    i = l;
    j = r + 1;

    if (dir == 0)
    {
        while (1)
        {
            do
                ++i;
            while (a[i] <= pivot && i <= r);
            do
                --j;
            while (a[j] > pivot);
            if (i >= j)
                break;
            t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        t = a[l];
        a[l] = a[j];
        a[j] = t;
        return j;
    }
    else if (dir == 1)
    {
        while (1)
        {
            do
                ++i;
            while (a[i] >= pivot && i <= r);
            do
                --j;
            while (a[j] < pivot);
            if (i >= j)
                break;
            t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        t = a[l];
        a[l] = a[j];
        a[j] = t;
        return j;
    }
    return 0;
}

void quickSortM(int_t *a, int_t l, int_t r, int_t lda, int_t dir, int_t dims)
{
    int_t j;

    if (l < r)
    {
        // printf("dims: %5d",dims);
        // fflush(stdout);

        // divide and conquer
        j = partitionM(a, l, r, lda, dir, dims);
        quickSortM(a, l, j-1, lda, dir, dims);
        quickSortM(a, j+1, r, lda, dir, dims);
    }
}

int_t partitionM(int_t *a, int_t l, int_t r, int_t lda, int_t dir, int_t dims)
{
    int_t pivot, i, j, t, dd;
    pivot = a[l];
    i = l;
    j = r + 1;

    if (dir == 0)
    {
        while (1)
        {
            do
                ++i;
            while (a[i] <= pivot && i <= r);
            do
                --j;
            while (a[j] > pivot);
            if (i >= j)
                break;
            for (dd = 0; dd < dims; dd++)
            {
                t = a[i + lda * dd];
                a[i + lda * dd] = a[j + lda * dd];
                a[j + lda * dd] = t;
            }
        }
        for (dd = 0; dd < dims; dd++)
        {
            t = a[l + lda * dd];
            a[l + lda * dd] = a[j + lda * dd];
            a[j + lda * dd] = t;
        }
        return j;
    }
    else if (dir == 1)
    {
        while (1)
        {
            do
                ++i;
            while (a[i] >= pivot && i <= r);
            do
                --j;
            while (a[j] < pivot);
            if (i >= j)
                break;
            for (dd = 0; dd < dims; dd++)
            {
                t = a[i + lda * dd];
                a[i + lda * dd] = a[j + lda * dd];
                a[j + lda * dd] = t;
            }
        }
        for (dd = 0; dd < dims; dd++)
        {
            t = a[l + lda * dd];
            a[l + lda * dd] = a[j + lda * dd];
            a[j + lda * dd] = t;
        }
        return j;
    }

    return 0;
} /* partitionM */

int_t **getTreePerm(int_t *myTreeIdxs, int_t *myZeroTrIdxs,
                    int_t *nodeCount, int_t **nodeList,
                    int_t *perm_c_supno, int_t *iperm_c_supno,
                    gridinfo3d_t *grid3d)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t **treePerm = SUPERLU_MALLOC(sizeof(int_t *) * maxLvl);
    for (int_t lvl = 0; lvl < maxLvl; lvl++)
    {
        // treePerm[lvl] = NULL;
        int_t treeId = myTreeIdxs[lvl];
        treePerm[lvl] = getPermNodeList(nodeCount[treeId], nodeList[treeId],
                                        perm_c_supno, iperm_c_supno);
    }
    return treePerm;
}

int_t *getMyNodeCounts(int_t maxLvl, int_t *myTreeIdxs, int_t *gNodeCount)
{
    int_t *myNodeCount = INT_T_ALLOC(maxLvl);
    for (int i = 0; i < maxLvl; ++i)
    {
        myNodeCount[i] = gNodeCount[myTreeIdxs[i]];
    }
    return myNodeCount;
}

/*chekc a vector vec of len across different process grids*/
int_t checkIntVector3d(int_t *vec, int_t len, gridinfo3d_t *grid3d)
{
    int_t nP = grid3d->zscp.Np;
    int_t myGrid = grid3d->zscp.Iam;
    int_t *buf = intMalloc_dist(len);

    if (!myGrid)
    {
        for (int_t p = 1; p < nP; ++p)
        {
            MPI_Status status;
            MPI_Recv(buf, len, mpi_int_t, p, p, grid3d->zscp.comm, &status);

            for (int_t i = 0; i < len; ++i)
            {
                /* code */
                if (buf[i] != vec[i])
                {
                    /* code */
                    printf("Error occured at (%d) Loc %d \n", (int)p, (int)i);
                    exit(0);
                }
            }
        }
    }
    else
    {
        MPI_Send(vec, len, mpi_int_t, 0, myGrid, grid3d->zscp.comm);
    }

    return 0;
}

/**
 * reduce the states from all the two grids before prinitng it out
 * See the defenition of enum PhaseType in superlu_enum_const.h
 */
int_t reduceStat(PhaseType PHASE,
                 SuperLUStat_t *stat, gridinfo3d_t *grid3d)
{
    flops_t *ops = stat->ops;

    flops_t flopcnt;
    MPI_Reduce(&ops[PHASE], &flopcnt, 1, MPI_FLOAT, MPI_SUM, 0, grid3d->zscp.comm);

    if (!grid3d->zscp.Iam)
    {
        ops[PHASE] = flopcnt;
    }

    return 0;
}

/*---- end from 3D code p3dcomm.c ----*/

#define GPU_ACC
#ifdef GPU_ACC

#define GPUMM_MIN_K 64  // minimum size of k formatrix multiplication on GPUs
#define GPUMM_MIN_MN 256*256     //minimum size of M\times N for matrix multiplication offload on GPUs

/*
 * Divide GEMM on GPU into multiple streams, each having sufficent work.
 */
void
gemm_division_cpu_gpu(
    superlu_dist_options_t *options,
/* output */
    int* num_streams_used, /* number of CUDA streams to be used,
			      it is <= nstreams   */
    int* stream_end_col,   /* array holding last column blk for each stream partition */
    int * ncpu_blks,       /* Number of CPU dgemm blks (output) */
    /*input */
    int nbrow,             /* number of row in A matrix */
    int ldu,               /* number of k in dgemm */
    int nstreams,          /* maximum possible GPU streams */
    int* full_u_cols,      /* array containing prefix sum of GPU workload */
    int num_blks,           /* Number of block cloumns (workload) on GPU */
    int_t gemmBufferSize       /*gemm buffer size*/
)
{
    int Ngem = sp_ienv_dist(7, options);  /*get_mnk_dgemm ();*/
    int min_gpu_col = get_gpublas_nb (); /* default 64 */
    int superlu_acc_offload = get_acc_offload();
    int ncols = full_u_cols[num_blks - 1];
    // int ncolsExcludingFirst =full_u_cols[num_blks - 1]


    /* Early return, when number of columns is smaller than threshold,
       or superlu_acc_offload == 0, then everything should be done on CPU. 
       Test condition GPU Flops ~ nbrow*ldu*cols < Ngem */
    if ( 
        (ldu < GPUMM_MIN_K)       // inner dimension is sufficient to hide latency
     || (nbrow*ncols < GPUMM_MIN_MN) // product of MN is sufficient
     || (ncols*nbrow*ldu < Ngem )
	 || (num_blks==1) || (nstreams==0)
     || nbrow*ncols > gemmBufferSize
	 || (superlu_acc_offload==0) )
    {
        *num_streams_used = 0;
        *ncpu_blks = num_blks;
        return;

    }

    for (int i = 0; i < nstreams; ++i)
    {
        stream_end_col[i] = num_blks;
    }

    *num_streams_used = 0;
    *ncpu_blks = 0;
    
    /* Find first block where count > Ngem */
    int i;
    for (i = 0; i < num_blks - 1; ++i)  /*I can use binary search here */
    {
        if (full_u_cols[i + 1] > Ngem / (nbrow * ldu))
            break;
    }
    *ncpu_blks = i + 1;

    int_t cols_remain =
        full_u_cols[num_blks - 1] - full_u_cols[*ncpu_blks - 1];

    if (cols_remain > 0)
    {
        *num_streams_used = 1;  /* now at least one stream would be used */

        int_t FP_MIN = 200000 / (nbrow * ldu); // >= 200000 flops per GPU stream
        int_t cols_per_stream = SUPERLU_MAX (min_gpu_col, cols_remain / nstreams);
        cols_per_stream = SUPERLU_MAX (cols_per_stream, FP_MIN);

        int_t cutoff = cols_per_stream + full_u_cols[*ncpu_blks - 1];
        for (int_t i = 0; i < nstreams; ++i)
        {
            stream_end_col[i] = num_blks;
        }
        int j = *ncpu_blks;
        for (int i = 0; i < nstreams - 1; ++i)
        {
            int_t st = (i == 0) ? (*ncpu_blks) : stream_end_col[i - 1];
	        // ^ starting block column of next stream

            for (j = st; j < num_blks - 1; ++j)
            {
                if (full_u_cols[j + 1] > cutoff)
                {
                    cutoff = cols_per_stream + full_u_cols[j];
                    stream_end_col[i] = j + 1;
                    *num_streams_used += 1;
                    j++;
                    break;  // block column j starts a new stream
                }
            } // end for j ...
        } // end for i ... streams

    }
	
} /* gemm_division_cpu_gpu */


#endif  /* defined GPU_ACC */

/* The following are moved from superlu_gpu.cu */

int getnGPUStreams()
{
    // Disabling multiple gpu streams -- bug with multiple streams in 3D code?
    #if 0
	return 1;
    #else 
	char *ttemp;
	ttemp = getenv ("SUPERLU_NUM_GPU_STREAMS");

	if (ttemp)
		return atoi (ttemp);
	else
		return 1;
    #endif 
}

int get_mpi_process_per_gpu ()
{
    char *ttemp;
    ttemp = getenv ("SUPERLU_MPI_PROCESS_PER_GPU");

    if (ttemp)
      return atol (ttemp);
    else
      {
	//printf("SUPERLU_MPI_PROCESS_PER_GPU is not set; Using default 1 \n");
	return 1;
      }
}

