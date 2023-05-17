
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
void validateInput_ssvx3d(superlu_dist_options_t *options, SuperMatrix *A,int ldb, int nrhs, gridinfo3d_t *grid3d, int *info)
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