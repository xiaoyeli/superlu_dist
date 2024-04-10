

#include <stdlib.h>  // For NULL
#include <mpi.h>
#include "superlu_ddefs.h"

#undef LOG_FUNC_ENTER
#define LOG_FUNC_ENTER() printf("\033[1;32mEntering function %s at %s:%d\033[0m\n", __func__, __FILE__, __LINE__)

/**
 * @brief Validates the input parameters for a given problem.
 *
 * This function checks the input parameters for a given problem and sets the
 * error code in the 'info' variable accordingly. If there is an error, it
 * prints an error message and returns.
 *
 * @param[in] options Pointer to the options structure containing Fact, RowPerm, ColPerm, and IterRefine values.
 * @param[in] A Pointer to the matrix A structure containing nrow, ncol, Stype, Dtype, and Mtype values.
 * @param[in] ldb The leading dimension of the array B.
 * @param[in] nrhs The number of right-hand sides.
 * @param[in] grid Pointer to the grid structure.
 * @param[out] info Pointer to an integer variable that stores the error code.
 */
void validateInput_pdgssvx3d(superlu_dist_options_t *options, SuperMatrix *A,
     int ldb, int nrhs, gridinfo3d_t *grid3d, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    NRformat_loc *Astore = A->Store;
    int Fact = options->Fact;
    if (Fact < 0 || Fact > FACTORED)
        *info = -1;
    else if (options->RowPerm < 0 || options->RowPerm > MY_PERMR)
        *info = -1;
    else if (options->ColPerm < 0 || options->ColPerm > MY_PERMC)
        *info = -1;
    else if (options->IterRefine < 0 || options->IterRefine > SLU_EXTRA)
        *info = -1;
    else if (options->IterRefine == SLU_EXTRA)
    {
        *info = -1;
        fprintf(stderr,
                "Extra precise iterative refinement yet to support.");
    }
    else if (A->nrow != A->ncol || A->nrow < 0 || A->Stype != SLU_NR_loc || A->Dtype != SLU_D || A->Mtype != SLU_GE)
        *info = -2;
    else if (ldb < Astore->m_loc)
        *info = -5;
    else if (nrhs < 0)
    {
        *info = -6;
    }
    if (*info)
    {
        int i = -(*info);
        pxerr_dist("pdgssvx3d", grid, -(*info));
        return;
    }
}


void dscaleRows(int_t m_loc, int_t fst_row, int_t *rowptr, double *a, double *R) {
    int_t irow = fst_row;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
			    a[i] *= R[irow];    /* Scale rows. */
        }
        ++irow;
    }
}

void dscaleColumns(int_t m_loc, int_t *rowptr, int_t *colind, double *a, double *C) {
    int_t icol;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            icol = colind[i];
            a[i] *= C[icol];          /* Scale columns. */
        }
    }
}

void dscaleBoth(int_t m_loc, int_t fst_row, int_t *rowptr,
    int_t *colind, double *a, double *R, double *C) {
    int_t irow = fst_row;
    int_t icol;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            icol = colind[i];
            a[i] *= R[irow] * C[icol]; /* Scale rows and cols. */
        }
        ++irow;
    }
}

void dscalePrecomputed(SuperMatrix *A, dScalePermstruct_t *ScalePermstruct) {
    NRformat_loc *Astore = (NRformat_loc *)A->Store;
    int_t m_loc = Astore->m_loc;
    int_t fst_row = Astore->fst_row;
    double *a = (double *)Astore->nzval;
    int_t *rowptr = Astore->rowptr;
    int_t *colind = Astore->colind;
    double *R = ScalePermstruct->R;
    double *C = ScalePermstruct->C;
    switch (ScalePermstruct->DiagScale) {
    case NOEQUIL:
        break;
    case ROW:
        dscaleRows(m_loc, fst_row, rowptr, a, R);
        break;
    case COL:
        dscaleColumns(m_loc, rowptr, colind, a, C);
        break;
    case BOTH:
        dscaleBoth(m_loc, fst_row, rowptr, colind, a, R, C);
        break;
    default:
        break;
    }
}

void dscaleFromScratch(
    SuperMatrix *A, dScalePermstruct_t *ScalePermstruct,
    gridinfo_t *grid, int *rowequ, int *colequ, int *iinfo)
{
    NRformat_loc *Astore = (NRformat_loc *)A->Store;
    int_t m_loc = Astore->m_loc;
    int_t fst_row = Astore->fst_row;
    double *a = (double *)Astore->nzval;
    int_t *rowptr = Astore->rowptr;
    int_t *colind = Astore->colind;
    double *R = ScalePermstruct->R;
    double *C = ScalePermstruct->C;
    double rowcnd, colcnd, amax;
    // int_t iinfo;
    char equed[1];
    int iam = grid->iam;

    pdgsequ(A, R, C, &rowcnd, &colcnd, &amax, iinfo, grid);

    if (*iinfo > 0) {
#if (PRNTlevel >= 1)
        fprintf(stderr, "The " IFMT "-th %s of A is exactly zero\n", *iinfo <= m_loc ? *iinfo : *iinfo - m_loc, *iinfo <= m_loc ? "row" : "column");
#endif
    } else if (*iinfo < 0) {
        return;
    }

    pdlaqgs(A, R, C, rowcnd, colcnd, amax, equed);

    if      (strncmp(equed, "R", 1) == 0) { ScalePermstruct->DiagScale = ROW; *rowequ = 1; *colequ = 0; }
    else if (strncmp(equed, "C", 1) == 0) { ScalePermstruct->DiagScale = COL; *rowequ = 0; *colequ = 1; }
    else if (strncmp(equed, "B", 1) == 0) { ScalePermstruct->DiagScale = BOTH; *rowequ = 1; *colequ = 1; }
    else                                  { ScalePermstruct->DiagScale = NOEQUIL; *rowequ = 0; *colequ = 0; }

#if (PRNTlevel >= 1)
    if (iam == 0) {
        printf(".. equilibrated? *equed = %c\n", *equed);
        fflush(stdout);
    }
#endif
}

void dscaleMatrixDiagonally(fact_t Fact, dScalePermstruct_t *ScalePermstruct,
                           SuperMatrix *A, SuperLUStat_t *stat, gridinfo_t *grid,
                            int *rowequ, int *colequ, int *iinfo)
{
    int iam = grid->iam;

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter equil");
#endif

    double t_start = SuperLU_timer_();

    if (Fact == SamePattern_SameRowPerm) {
        dscalePrecomputed(A, ScalePermstruct);
    } else {
        dscaleFromScratch(A, ScalePermstruct, grid, rowequ, colequ, iinfo);
    }

    stat->utime[EQUIL] = SuperLU_timer_() - t_start;

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit equil");
#endif
}

/**
 * Finds row permutations using the MC64 algorithm in a distributed manner.
 *
 * @param grid The grid info object, which includes the current node's information and MPI communicator.
 * @param job The type of job to be done.
 * @param m The number of rows in the sparse matrix.
 * @param n The number of columns in the sparse matrix.
 * @param nnz The number of non-zero elements in the sparse matrix.
 * @param colptr The column pointer array of the sparse matrix (CSC format).
 * @param rowind The row index array of the sparse matrix (CSC format).
 * @param a_GA The non-zero values of the sparse matrix.
 * @param Equil The equilibration flag.
 * @param perm_r The output permutation array for the rows.
 * @param R1 The output row scaling factors.
 * @param C1 The output column scaling factors.
 * @param iinfo The output status code.
 */
void dfindRowPerm_MC64(gridinfo_t* grid, int_t job,
                      int_t m, int_t n,
                      int_t nnz,
                      int_t* colptr,
                      int_t* rowind,
                      double* a_GA,
                      int_t Equil,
                      int_t* perm_r,
                      double* R1,
                      double* C1,
                      int* iinfo) {
    #if ( DEBUGlevel>=1 )
    LOG_FUNC_ENTER();
    #endif
    // Check input parameters
    if (colptr == NULL || rowind == NULL || a_GA == NULL ||
        perm_r == NULL ) {
        fprintf(stderr, "Error: NULL input parameter.\n");
        return;
    }

    int root = 0;

    // If the current node is the root node, perform the permutation computation
    if (grid->iam == root) {
        *iinfo = dldperm_dist(job, m, nnz, colptr, rowind, a_GA, perm_r, R1, C1);
    }

    // Broadcast the status code to all other nodes in the communicator
    MPI_Bcast(iinfo, 1, MPI_INT, root, grid->comm);

    // If the computation was successful
    if (*iinfo == 0) {
        // Broadcast the resulting permutation array to all other nodes
        MPI_Bcast(perm_r, m, mpi_int_t, root, grid->comm);

        // If job == 5 and Equil == true, broadcast the scaling factors as well
        if (job == 5 && Equil) {
            MPI_Bcast(R1, m, MPI_DOUBLE, root, grid->comm);
            MPI_Bcast(C1, n, MPI_DOUBLE, root, grid->comm);
        }
    }
}


/**
 * This function scales a distributed matrix.
 *

 * @param[in]   rowequ  A flag indicating whether row should be equalized.
 * @param[in]   colequ  A flag indicating whether column should be equalized.
 * @param[in]   m       Number of rows in the matrix.
 * @param[in]   n       Number of columns in the matrix.
 * @param[in]   m_loc   The local row dimension.
 * @param[in]   rowptr  Pointer to the array holding row pointers.
 * @param[in]   colind  Pointer to the array holding column indices.
 * @param[in]   fst_row The first row of the local block.
 * @param[in,out] a     Pointer to the array holding the values of the matrix.
 * @param[in,out] R     Pointer to the row scaling factors.
 * @param[in,out] C     Pointer to the column scaling factors.
 * @param[in,out] R1    Pointer to the array holding the new row scaling factors.
 * @param[in,out] C1    Pointer to the array holding the new column scaling factors.
 */
void dscale_distributed_matrix(int rowequ, int colequ, int_t m, int_t n,
 int_t m_loc, int_t *rowptr, int_t *colind, int_t fst_row, double *a,
  double *R, double *C, double *R1, double *C1)
{
    #if ( DEBUGlevel>=1 )
    LOG_FUNC_ENTER();
    #endif
    // Scale the row and column factors
    for (int i = 0; i < n; ++i) {
        R1[i] = exp(R1[i]);
        C1[i] = exp(C1[i]);
    }

    // Scale the elements of the matrix
    int rowIndex = fst_row;
    for (int j = 0; j < m_loc; ++j) {
        for (int i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            int columnIndex = colind[i];
            a[i] *= R1[rowIndex] * C1[columnIndex];

#if 0
// this is not support as dmin, dsum and dprod are not used later in pdgssvx3d
#if (PRNTlevel >= 2)
            if (perm_r[irow] == icol)
            {
                /* New diagonal */
                if (job == 2 || job == 3)
                    dmin = SUPERLU_MIN(dmin, fabs(a[i]));
                else if (job == 4)
                    dsum += fabs(a[i]);
                else if (job == 5)
                    dprod *= fabs(a[i]);
            }
#endif
#endif
        }
        ++rowIndex;
    }

    // Scale the row factors
    for (int i = 0; i < m; ++i)
        R[i] = (rowequ) ? R[i] * R1[i] : R1[i];

    // Scale the column factors
    for (int i = 0; i < n; ++i)
        C[i] = (colequ) ? C[i] * C1[i] : C1[i];
}


/**
 * Performs a permutation operation on the rows of a sparse matrix (CSC format).
 *
 * @param m The number of rows in the sparse matrix.
 * @param n The number of columns in the sparse matrix.
 * @param colptr The column pointer array of the sparse matrix (CSC format).
 * @param rowind The row index array of the sparse matrix (CSC format).
 * @param perm_r The permutation array for the rows.
 */
void dpermute_global_A(int_t m, int_t n, int_t *colptr, int_t *rowind, int_t *perm_r) {
    // Check input parameters
    if (colptr == NULL || rowind == NULL || perm_r == NULL) {
        fprintf(stderr, "Error: NULL input parameter to: dpermute_global_A()\n");
        return;
    }

    // Iterate through each column
    for (int_t j = 0; j < n; ++j) {
        // For each column, iterate through its non-zero elements
        for (int_t i = colptr[j]; i < colptr[j + 1]; ++i) {
            // Get the original row index
            int_t irow = rowind[i];
            // Assign the new row index from the permutation array
            rowind[i] = perm_r[irow];
        }
    }
}


/**
 * @brief Performs a set of operations on distributed matrices including finding row permutations, scaling, and permutation of global A.
 * The operations depend on job and iinfo parameters.
 *
 * @param[in]     options                SuperLU options.
 * @param[in]     Fact                   Factored form of the matrix.
 * @param[in,out] ScalePermstruct        Scaling and Permutation structure.
 * @param[in,out] LUstruct               LU decomposition structure.
 * @param[in]     m                      Number of rows in the matrix.
 * @param[in]     n                      Number of columns in the matrix.
 * @param[in]     grid                   Grid information for distributed computation.
 * @param[in,out] A                      SuperMatrix A to be operated upon.
 * @param[in,out] GA                     SuperMatrix GA to be operated upon.
 * @param[out]    stat                   SuperLU statistics object to record factorization statistics.
 * @param[in]     job                    The type of job to be done.
 * @param[in]     Equil                  The equilibration flag.
 * @param[in]     rowequ                 Flag indicating whether rows of the matrix should be equalized.
 * @param[in]     colequ                 Flag indicating whether columns of the matrix should be equalized.
 * @param[out]    iinfo                  The output status code.
 *
 * @note The functions dfindRowPerm_MC64, dscale_distributed_matrix and dpermute_global_A are called in this function.
 */
void dperform_LargeDiag_MC64(
    superlu_dist_options_t *options, fact_t Fact,
    dScalePermstruct_t *ScalePermstruct, dLUstruct_t *LUstruct,
    int_t m, int_t n, gridinfo_t *grid,
    SuperMatrix *A, SuperMatrix *GA, SuperLUStat_t *stat, int_t job,
    int Equil, int *rowequ, int *colequ, int *iinfo) {
    double *R1 = NULL;
    double *C1 = NULL;

    int_t *perm_r = ScalePermstruct->perm_r;
    int_t *perm_c = ScalePermstruct->perm_c;
    int_t *etree = LUstruct->etree;
    double *R = ScalePermstruct->R;
    double *C = ScalePermstruct->C;
    int iam = grid->iam;


    NRformat_loc *Astore = (NRformat_loc *)A->Store;
    int_t nnz_loc = (Astore)->nnz_loc;
    int_t m_loc = (Astore)->m_loc;
    int_t fst_row = (Astore)->fst_row;
    double *a = (double *)(Astore)->nzval;
    int_t *rowptr = (Astore)->rowptr;
    int_t *colind = (Astore)->colind;

    NCformat *GAstore = (NCformat *)GA->Store;
    int_t *colptr = (GAstore)->colptr;
    int_t *rowind = (GAstore)->rowind;
    int_t nnz = (GAstore)->nnz;
    double *a_GA = (double *)(GAstore)->nzval;

    if (job == 5) {
        R1 = doubleMalloc_dist(m);
        if (!R1)
            ABORT("SUPERLU_MALLOC fails for R1[]");
        C1 = doubleMalloc_dist(n);
        if (!C1)
            ABORT("SUPERLU_MALLOC fails for C1[]");
    }

    // int iinfo;
    dfindRowPerm_MC64(grid, job, m, n,
    nnz,
    colptr,
    rowind,
     a_GA, Equil, perm_r, R1, C1, iinfo);

    if (*iinfo && job == 5) {
        SUPERLU_FREE(R1);
        SUPERLU_FREE(C1);
    }
#if (PRNTlevel >= 2)
    double dmin = damch_dist("Overflow");
    double dsum = 0.0;
    double dprod = 1.0;
#endif
    if (*iinfo == 0) {
        if (job == 5) {
            /* Scale the distributed matrix further.
									   A <-- diag(R1)*A*diag(C1)            */
            if(Equil)
            {
                dscale_distributed_matrix( *rowequ, *colequ, m, n, m_loc, rowptr, colind, fst_row, a, R, C, R1, C1);
                ScalePermstruct->DiagScale = BOTH;
                *rowequ = *colequ = 1;
            } /* end if Equil */
            dpermute_global_A( m, n, colptr, rowind, perm_r);
            SUPERLU_FREE(R1);
            SUPERLU_FREE(C1);
        } else {
            dpermute_global_A( m, n, colptr, rowind, perm_r);
        }
    }
    else
    { /* if iinfo != 0 */
        for (int_t i = 0; i < m; ++i)
            perm_r[i] = i;
    }
#if (PRNTlevel >= 2)
#warning following is not supported
    if (job == 2 || job == 3)
    {
        if (!iam)
            printf("\tsmallest diagonal %e\n", dmin);
    }
    else if (job == 4)
    {
        if (!iam)
            printf("\tsum of diagonal %e\n", dsum);
    }
    else if (job == 5)
    {
        if (!iam)
            printf("\t product of diagonal %e\n", dprod);
    }
#endif
} /* dperform_LargeDiag_MC64 */


void dperform_row_permutation(
    superlu_dist_options_t *options,
    fact_t Fact,
    dScalePermstruct_t *ScalePermstruct, dLUstruct_t *LUstruct,
    int_t m, int_t n,
    gridinfo_t *grid,
    SuperMatrix *A,
    SuperMatrix *GA,
    SuperLUStat_t *stat,
    int job,
    int Equil,
    int *rowequ,
    int *colequ,
    int *iinfo)
{
    #if ( DEBUGlevel>=1 )
    LOG_FUNC_ENTER();
    #endif
    int_t *perm_r = ScalePermstruct->perm_r;
    /* Get NC format data from SuperMatrix GA */
    NCformat* GAstore = (NCformat *)GA->Store;
    int_t* colptr = GAstore->colptr;
    int_t* rowind = GAstore->rowind;
    int_t nnz = GAstore->nnz;
    double* a_GA = (double *)GAstore->nzval;

    int iam = grid->iam;
    /* ------------------------------------------------------------
			   Find the row permutation for A.
    ------------------------------------------------------------ */
    double t;

    if (options->RowPerm != NO)
    {
        t = SuperLU_timer_();

        if (Fact != SamePattern_SameRowPerm)
        {
            if (options->RowPerm == MY_PERMR)
            {
                applyRowPerm(colptr, rowind, perm_r, n);
            }
            else if (options->RowPerm == LargeDiag_MC64)
            {

                dperform_LargeDiag_MC64(
                options, Fact,
                ScalePermstruct, LUstruct,
                m, n, grid,
                A, GA, stat, job,
                Equil, rowequ, colequ, iinfo);
            }
            else // LargeDiag_HWPM
            {
#ifdef HAVE_COMBBLAS
                d_c2cpp_GetHWPM(A, grid, ScalePermstruct);
#else
                if (iam == 0)
                {
                    printf("CombBLAS is not available\n");
                    fflush(stdout);
                }
#endif
            }

            t = SuperLU_timer_() - t;
            stat->utime[ROWPERM] = t;
#if (PRNTlevel >= 1)
            if (!iam)
            {
                printf(".. LDPERM job %d\t time: %.2f\n", job, t);
                fflush(stdout);
            }
#endif
        }
    }
    else // options->RowPerm == NOROWPERM / NATURAL
    {
        for (int i = 0; i < m; ++i)
            perm_r[i] = i;
    }

    #if (DEBUGlevel >= 2)
	if (!grid->iam)
		PrintInt10("perm_r", m, perm_r);
    #endif
}


/**
 * @brief This function computes the norm of a matrix A.
 * @param notran A flag which determines the norm type to be calculated.
 * @param A The input matrix for which the norm is computed.
 * @param grid The gridinfo_t object that contains the information of the grid.
 * @return Returns the computed norm of the matrix A.
 *
 * the iam process is the root (iam=0), it prints the computed norm to the standard output.
 */
double dcomputeA_Norm(int notran, SuperMatrix *A, gridinfo_t *grid) {
    char norm;
    double anorm;

    /* Compute norm(A), which will be used to adjust small diagonal. */
    if (notran)
        norm = '1';
    else
        norm = 'I';

    anorm = pdlangs(&norm, A, grid);

#if (PRNTlevel >= 1)
    if (!grid->iam) {
        printf(".. anorm %e\n", anorm);
        fflush(stdout);
    }
#endif

    return anorm;
}

void dallocScalePermstruct_RC(dScalePermstruct_t * ScalePermstruct, int_t m, int_t n) {
    /* Allocate storage if not done so before. */
	switch (ScalePermstruct->DiagScale) {
		case NOEQUIL:
			if (!(ScalePermstruct->R = (double *)doubleMalloc_dist(m)))
				ABORT("Malloc fails for R[].");
			if (!(ScalePermstruct->C = (double *)doubleMalloc_dist(n)))
				ABORT("Malloc fails for C[].");
			break;
		case ROW:
			if (!(ScalePermstruct->C = (double *)doubleMalloc_dist(n)))
				ABORT("Malloc fails for C[].");
			break;
		case COL:
			if (!(ScalePermstruct->R = (double *)doubleMalloc_dist(m)))
				ABORT("Malloc fails for R[].");
			break;
		default:
			break;
	}
}


#ifdef REFACTOR_SYMBOLIC
/**
 * @brief Distributes the permuted matrix into L and U storage.
 *
 * @param[in] options           Pointer to the options structure.
 * @param[in] n                 Order of the input matrix.
 * @param[in] A                 Pointer to the input matrix.
 * @param[in] ScalePermstruct   Pointer to the scaling and permutation structures.
 * @param[in] Glu_freeable      Pointer to the LU data structures that can be deallocated.
 * @param[out] LUstruct         Pointer to the LU data structures.
 * @param[in] grid              Pointer to the process grid.
 * @return Memory usage in bytes (0 if successful).
 */
int dDistributePermutedMatrix(const superlu_dist_options_t *options,
                             const int_t n,
                             const SuperMatrix *A,
                             const dScalePermstruct_t *ScalePermstruct,
                             const Glu_freeable_t *Glu_freeable,
                             LUstruct_t *LUstruct,
                             const gridinfo_t *grid);

#endif // REFACTOR_SYMBOLIC


#ifdef REFACTOR_DistributePermutedMatrix

#endif // REFACTOR_DistributePermutedMatrix


