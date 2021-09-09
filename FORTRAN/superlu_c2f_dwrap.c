

/*! @file 
 * \brief C interface functions for the Fortran90 wrapper.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 2012
 * April 5, 2015
 * May 12, 2021
 */

#include "superlu_ddefs.h"
#include "superlu_FCnames.h"


/* kind of integer to hold a pointer.
   Be sure to be consistent with that in superlupara.f90 */
#if 0
typedef int fptr;  /* 32-bit */
#else
typedef long long int fptr;  /* 64-bit */
#endif

/* functions that create memory for a struct and return a handle */

void f_dcreate_ScalePerm_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(dScalePermstruct_t));
}

void f_dcreate_LUstruct_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(dLUstruct_t));
}

void f_dcreate_SOLVEstruct_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(dSOLVEstruct_t));
}

/* wrappers for SuperLU functions */

void f_dScalePermstructInit(int *m, int *n, fptr *ScalePermstruct)
{
   dScalePermstructInit(*m, *n, (dScalePermstruct_t *) *ScalePermstruct);
}

void f_dScalePermstructFree(fptr *ScalePermstruct)
{
   dScalePermstructFree((dScalePermstruct_t *) *ScalePermstruct);
}

void f_dLUstructInit(int *m, int *n, fptr *LUstruct)
{
   extern void dLUstructInit(const int_t, dLUstruct_t *);

   dLUstructInit(*m, (dLUstruct_t *) *LUstruct);
}

void f_dLUstructFree(fptr *LUstruct)
{
   extern void dLUstructFree(dLUstruct_t *);

   dLUstructFree((dLUstruct_t *) *LUstruct);
}

void f_dDestroy_LU_SOLVE_struct(fptr *options, int *n, fptr *grid,
                               fptr *LUstruct, fptr *SOLVEstruct)
{
    superlu_dist_options_t *opt = (superlu_dist_options_t *) *options;
    dDestroy_LU(*n, (gridinfo_t *) *grid, (dLUstruct_t *) *LUstruct);
    dLUstructFree((dLUstruct_t *) *LUstruct);
    if ( opt->SolveInitialized ) {
        dSolveFinalize(opt, (dSOLVEstruct_t *) *SOLVEstruct);
    }
}

void f_dDestroy_LU_SOLVE_struct_3d(fptr *options, int *n, fptr *grid,
		                  fptr *LUstruct, fptr *SOLVEstruct)
{
    gridinfo3d_t *grid3d = (gridinfo3d_t *) *grid;
    superlu_dist_options_t *opt = (superlu_dist_options_t *) *options;
    dLUstruct_t *LUstruct_ptr = (dLUstruct_t *) *LUstruct;
    
    if ( grid3d->zscp.Iam == 0 ) { // process layer 0
	dDestroy_LU(*n, &(grid3d->grid2d), LUstruct_ptr);
    	dSolveFinalize(opt, (dSOLVEstruct_t *) *SOLVEstruct);
    } else { // process layers not equal 0
        dDeAllocLlu_3d(*n, LUstruct_ptr, grid3d);
        dDeAllocGlu_3d(LUstruct_ptr);
    }
    
    dLUstructFree(LUstruct_ptr);
}

void f_dDestroy_A3d_gathered_on_2d(fptr *SOLVEstruct, fptr *grid3d)
{
    dDestroy_A3d_gathered_on_2d((dSOLVEstruct_t *) *SOLVEstruct,
                                      (gridinfo3d_t *) *grid3d);
}


void f_dCreate_CompRowLoc_Mat_dist(fptr *A, int *m, int *n, int *nnz_loc,
				   int *m_loc, int *fst_row, double *nzval,
				   int_t *colind, int_t *rowptr, int *stype,
				   int *dtype, int *mtype)
{
#if 1
    double *C_nzval = nzval;
    int_t *C_colind = colind;
    int_t *C_rowptr = rowptr;
#else
    /* make a copy of matrix A that is internal to the C side */
    double *C_nzval = doubleMalloc_dist(*nnz_loc);
    int_t *C_colind = intMalloc_dist(*nnz_loc);
    int_t *C_rowptr = intMalloc_dist(*m_loc + 1);
    int i;
    
    for (i = 0; i < *nnz_loc; ++i) {
        C_nzval[i] = nzval[i];
        C_colind[i] = colind[i];
    }
    for (i = 0; i <= *m_loc; ++i) {
        C_rowptr[i] = rowptr[i];
    }
#endif

    dCreate_CompRowLoc_Matrix_dist((SuperMatrix *) *A, *m, *n, *nnz_loc, *m_loc,
                                  *fst_row, C_nzval, C_colind, C_rowptr,
                                  (Stype_t) *stype, (Dtype_t) *dtype,
                                  (Mtype_t) *mtype);
}

void f_dSolveFinalize(fptr *options, fptr *SOLVEstruct)
{
   dSolveFinalize((superlu_dist_options_t *) *options,
                  (dSOLVEstruct_t *) *SOLVEstruct);
}

void f_pdgssvx(fptr *options, fptr *A, fptr *ScalePermstruct, double *B,
               int *ldb, int *nrhs, fptr *grid, fptr *LUstruct,
               fptr *SOLVEstruct, double *berr, fptr *stat, int *info)
{
    pdgssvx((superlu_dist_options_t *) *options, (SuperMatrix *) *A,
	    (dScalePermstruct_t *) *ScalePermstruct, B, *ldb, *nrhs,
	    (gridinfo_t *) *grid, (dLUstruct_t *) *LUstruct,
	    (dSOLVEstruct_t *) *SOLVEstruct, berr,
	    (SuperLUStat_t *) *stat, info);

    PStatPrint((superlu_dist_options_t *) *options, (SuperLUStat_t *) *stat,
	       (gridinfo_t *) *grid);
}

void f_pdgssvx3d(fptr *options, fptr *A, fptr *ScalePermstruct,
                 double *B, int *ldb, int *nrhs,
                 fptr *grid, fptr *LUstruct, fptr *SOLVEstruct,
                 double *berr, fptr *stat, int *info)
{
    gridinfo3d_t *grid3d = (gridinfo3d_t *) *grid;
    pdgssvx3d((superlu_dist_options_t *) *options, (SuperMatrix *) *A,
	      (dScalePermstruct_t *) *ScalePermstruct, B, *ldb, *nrhs,
	      grid3d, (dLUstruct_t *) *LUstruct,
	      (dSOLVEstruct_t *) *SOLVEstruct, berr,
	      (SuperLUStat_t *) *stat, info);

    if ( grid3d->zscp.Iam == 0 ) {
	PStatPrint((superlu_dist_options_t *) *options,
		   (SuperLUStat_t *) *stat, &(grid3d->grid2d));
    }
}

/* Create the distributed matrix */

void f_dcreate_matrix_x_b(char *fname, fptr *A, int *m, int *n, int_t *nnz,
		           int *nrhs, double *b, int *ldb,
		           double *xtrue, int *ldx, fptr *grid)
{
    extern int c2f_dcreate_matrix_x_b(char *fname, int nrhs, int nprocs,
    	                   MPI_Comm, SuperMatrix *A, int *m_g, int *n_g,
			   int_t *nnz_g, double *rhs, int *ldb,
			   double *x, int *ldx);
    extern void f_get_gridinfo(fptr *grid, int *iam, int *nprow, int *npcol);

    int iam, nprocs;
    int nprow, npcol;
    MPI_Comm slucomm = ((gridinfo_t *) *grid)->comm;
    f_get_gridinfo(grid, &iam, &nprow, &npcol);
    nprocs = nprow * npcol;
			   
    c2f_dcreate_matrix_x_b(fname, *nrhs, nprocs, slucomm,
    	                   (SuperMatrix *) *A, m, n, nnz, b, ldb, xtrue, ldx);
}

void f_dcreate_matrix_x_b_3d(char *fname, fptr *A, int *m, int *n, int_t *nnz,
		           int *nrhs, double *b, int *ldb,
		           double *xtrue, int *ldx, fptr *grid)
{
    extern int c2f_dcreate_matrix_x_b(char *fname, int nrhs, int nprocs,
    	                   MPI_Comm, SuperMatrix *A, int *m_g, int *n_g,
			   int_t *nnz_g, double *rhs, int *ldb,
			   double *x, int *ldx);
    extern void f_get_gridinfo3d(fptr *grid, int *iam,
                                 int *nprow, int *npcol, int *npdep);

    int iam, nprocs;
    int nprow, npcol, npdep;
    MPI_Comm slucomm = ((gridinfo3d_t *) *grid)->comm;
    f_get_gridinfo3d(grid, &iam, &nprow, &npcol, &npdep);
    nprocs = nprow * npcol * npdep;
			   
    c2f_dcreate_matrix_x_b(fname, *nrhs, nprocs, slucomm,
    	                   (SuperMatrix *) *A, m, n, nnz, b, ldb, xtrue, ldx);
}

