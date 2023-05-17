
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

void scaleRows(int_t m_loc, int_t fst_row, int_t *rowptr, double *a, double *R) {
    int_t irow = fst_row;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            a[i] *= R[irow];
        }
        ++irow;
    }
}

void scaleColumns(int_t m_loc, int_t *rowptr, int_t *colind, double *a, double *C) {
    int_t icol;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            icol = colind[i];
            a[i] *= C[icol];
        }
    }
}

void scaleBoth(int_t m_loc, int_t fst_row, int_t *rowptr, int_t *colind, double *a, double *R, double *C) {
    int_t irow = fst_row;
    int_t icol;
    for (int_t j = 0; j < m_loc; ++j) {
        for (int_t i = rowptr[j]; i < rowptr[j + 1]; ++i) {
            icol = colind[i];
            a[i] *= R[irow] * C[icol];
        }
        ++irow;
    }
}




void scalePrecomputed(SuperMatrix *A,
    dScalePermstruct_t *ScalePermstruct) 
{
    NRformat_loc *Astore = (NRformat_loc *)A->Store;
    int_t m_loc = Astore->m_loc;
    int_t fst_row = Astore->fst_row;
    double *a = (double *)Astore->nzval;
    int_t *rowptr = Astore->rowptr;
    int_t *colind = Astore->colind;
    double *R = ScalePermstruct->R;
    double *C = ScalePermstruct->C;
    switch (ScalePermstruct->DiagScale) {
    case ROW:
        scaleRows(m_loc, fst_row, rowptr, a, R);
        break;
    case COL:
        scaleColumns(m_loc, rowptr, colind, a, C);
        break;
    case BOTH:
        scaleBoth(m_loc, fst_row, rowptr, colind, a, R, C);
        break;
    default:
        break;
    }
}

void scaleFromScratch(
    SuperMatrix *A, dScalePermstruct_t *ScalePermstruct,  gridinfo_t *grid, int_t *rowequ, int_t *colequ)  
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
    int_t iinfo;
    char equed[1];
    int iam = grid->iam;

    pdgsequ(A, R, C, &rowcnd, &colcnd, &amax, &iinfo, grid);

    if (iinfo != 0) {
        if (iinfo > 0) fprintf(stderr, "The " IFMT "-th %s of A is exactly zero\n", iinfo <= m_loc ? iinfo : iinfo - m_loc, iinfo <= m_loc ? "row" : "column");
        return;
    }

    pdlaqgs(A, R, C, rowcnd, colcnd, amax, equed);

    if      (strncmp(equed, "R", 1) == 0) { ScalePermstruct->DiagScale = ROW; *rowequ = 1; *colequ = 0; }
    else if (strncmp(equed, "C", 1) == 0) { ScalePermstruct->DiagScale = COL; *rowequ = 0; *colequ = 1; }
    else if (strncmp(equed, "B", 1) == 0) { ScalePermstruct->DiagScale = BOTH; *rowequ = 1; *colequ = 1; }
    else                                  { ScalePermstruct->DiagScale = NOEQUIL; *rowequ = 0; *colequ = 0; }
    // else if (strncmp(equed, "C", 1) == 0) ScalePermstruct->DiagScale = COL;
    // else if (strncmp(equed, "B", 1) == 0) ScalePermstruct->DiagScale = BOTH;
    // else                                  ScalePermstruct->DiagScale = NOEQUIL;

    if (iam == 0) printf(".. equilibrated? *equed = %c\n", *equed);
}





void scaleMatrixDiagonally(fact_t Fact, dScalePermstruct_t *ScalePermstruct, 
                           SuperMatrix *A, SuperLUStat_t *stat, gridinfo_t *grid, int_t *rowequ, int_t *colequ) 
{
    
    
    NRformat_loc *Astore = (NRformat_loc *)A->Store;
    int_t m_loc = Astore->m_loc;
    int_t fst_row = Astore->fst_row;
    double *a = (double *)Astore->nzval;
    int_t *rowptr = Astore->rowptr;
    int_t *colind = Astore->colind;
    double *R = ScalePermstruct->R;
    double *C = ScalePermstruct->C;
    

    double t_start = SuperLU_timer_();

    if (Fact == SamePattern_SameRowPerm) {
        scalePrecomputed(A, ScalePermstruct);
    } else {
        scaleFromScratch(A, ScalePermstruct, grid, rowequ, colequ);
    }

    stat->utime[EQUIL] = SuperLU_timer_() - t_start;
}