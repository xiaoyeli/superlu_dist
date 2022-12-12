
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

#include "superlu_zdefs.h"
#include "superlu_FCnames.h"


/* kind of integer to hold a pointer.
   Be sure to be consistent with that in superlupara.f90 */
#if 0
typedef int fptr;  /* 32-bit */
#else
typedef long long int fptr;  /* 64-bit */
#endif

/* functions that create memory for a struct and return a handle */

void f_zcreate_ScalePerm_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(zScalePermstruct_t));
}

void f_zcreate_LUstruct_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(zLUstruct_t));
}

void f_zcreate_SOLVEstruct_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(zSOLVEstruct_t));
}

/* wrappers for SuperLU functions */

void f_zScalePermstructInit(int *m, int *n, fptr *ScalePermstruct)
{
   zScalePermstructInit(*m, *n, (zScalePermstruct_t *) *ScalePermstruct);
}

void f_zScalePermstructFree(fptr *ScalePermstruct)
{
   zScalePermstructFree((zScalePermstruct_t *) *ScalePermstruct);
}

void f_zLUstructInit(int *m, int *n, fptr *LUstruct)
{
   extern void zLUstructInit(const int_t, zLUstruct_t *);

   zLUstructInit(*m, (zLUstruct_t *) *LUstruct);
}

void f_zLUstructFree(fptr *LUstruct)
{
   extern void zLUstructFree(zLUstruct_t *);

   zLUstructFree((zLUstruct_t *) *LUstruct);
}

void f_zDestroy_LU_SOLVE_struct(fptr *options, int *n, fptr *grid,
                               fptr *LUstruct, fptr *SOLVEstruct)
{
    superlu_dist_options_t *opt = (superlu_dist_options_t *) *options;
    zDestroy_LU(*n, (gridinfo_t *) *grid, (zLUstruct_t *) *LUstruct);
    zLUstructFree((zLUstruct_t *) *LUstruct);
    if ( opt->SolveInitialized ) {
        zSolveFinalize(opt, (zSOLVEstruct_t *) *SOLVEstruct);
    }
}

void f_zDestroy_LU_SOLVE_struct_3d(fptr *options, int *n, fptr *grid,
		                  fptr *LUstruct, fptr *SOLVEstruct)
{
    gridinfo3d_t *grid3d = (gridinfo3d_t *) *grid;
    superlu_dist_options_t *opt = (superlu_dist_options_t *) *options;
    zLUstruct_t *LUstruct_ptr = (zLUstruct_t *) *LUstruct;
    
    if ( grid3d->zscp.Iam == 0 ) { // process layer 0
	zDestroy_LU(*n, &(grid3d->grid2d), LUstruct_ptr);
    	zSolveFinalize(opt, (zSOLVEstruct_t *) *SOLVEstruct);
    } else { // process layers not equal 0
        zDeAllocLlu_3d(*n, LUstruct_ptr, grid3d);
        zDeAllocGlu_3d(LUstruct_ptr);
    }
    
    zLUstructFree(LUstruct_ptr);
}

void f_zDestroy_A3d_gathered_on_2d(fptr *SOLVEstruct, fptr *grid3d)
{
    zDestroy_A3d_gathered_on_2d((zSOLVEstruct_t *) *SOLVEstruct,
                                      (gridinfo3d_t *) *grid3d);
}


void f_zCreate_CompRowLoc_Mat_dist(fptr *A, int *m, int *n, int *nnz_loc,
				   int *m_loc, int *fst_row, doublecomplex *nzval,
				   int_t *colind, int_t *rowptr, int *stype,
				   int *dtype, int *mtype)
{
#if 1
    doublecomplex *C_nzval = nzval;
    int_t *C_colind = colind;
    int_t *C_rowptr = rowptr;
#else
    /* make a copy of matrix A that is internal to the C side */
    doublecomplex *C_nzval = doublecomplexMalloc_dist(*nnz_loc);
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

    zCreate_CompRowLoc_Matrix_dist((SuperMatrix *) *A, *m, *n, *nnz_loc, *m_loc,
                                  *fst_row, C_nzval, C_colind, C_rowptr,
                                  (Stype_t) *stype, (Dtype_t) *dtype,
                                  (Mtype_t) *mtype);
}

void f_zSolveFinalize(fptr *options, fptr *SOLVEstruct)
{
   zSolveFinalize((superlu_dist_options_t *) *options,
                  (zSOLVEstruct_t *) *SOLVEstruct);
}

void f_pzgssvx(fptr *options, fptr *A, fptr *ScalePermstruct, doublecomplex *B,
               int *ldb, int *nrhs, fptr *grid, fptr *LUstruct,
               fptr *SOLVEstruct, double *berr, fptr *stat, int *info)
{
    pzgssvx((superlu_dist_options_t *) *options, (SuperMatrix *) *A,
	    (zScalePermstruct_t *) *ScalePermstruct, B, *ldb, *nrhs,
	    (gridinfo_t *) *grid, (zLUstruct_t *) *LUstruct,
	    (zSOLVEstruct_t *) *SOLVEstruct, berr,
	    (SuperLUStat_t *) *stat, info);

    PStatPrint((superlu_dist_options_t *) *options, (SuperLUStat_t *) *stat,
	       (gridinfo_t *) *grid);
}

void f_pzgssvx3d(fptr *options, fptr *A, fptr *ScalePermstruct,
                 doublecomplex *B, int *ldb, int *nrhs,
                 fptr *grid, fptr *LUstruct, fptr *SOLVEstruct,
                 double *berr, fptr *stat, int *info)
{
    gridinfo3d_t *grid3d = (gridinfo3d_t *) *grid;
    pzgssvx3d((superlu_dist_options_t *) *options, (SuperMatrix *) *A,
	      (zScalePermstruct_t *) *ScalePermstruct, B, *ldb, *nrhs,
	      grid3d, (zLUstruct_t *) *LUstruct,
	      (zSOLVEstruct_t *) *SOLVEstruct, berr,
	      (SuperLUStat_t *) *stat, info);

    if ( grid3d->zscp.Iam == 0 ) {
	PStatPrint((superlu_dist_options_t *) *options,
		   (SuperLUStat_t *) *stat, &(grid3d->grid2d));
    }
}

/* Create the distributed matrix */

void f_zcreate_matrix_x_b(char *fname, fptr *A, int *m, int *n, int_t *nnz,
		           int *nrhs, doublecomplex *b, int *ldb,
		           doublecomplex *xtrue, int *ldx, fptr *grid)
{
    extern int c2f_zcreate_matrix_x_b(char *fname, int nrhs, int nprocs,
    	                   MPI_Comm, SuperMatrix *A, int *m_g, int *n_g,
			   int_t *nnz_g, doublecomplex *rhs, int *ldb,
			   doublecomplex *x, int *ldx);
    extern void f_get_gridinfo(fptr *grid, int *iam, int *nprow, int *npcol);

    int iam, nprocs;
    int nprow, npcol;
    MPI_Comm slucomm = ((gridinfo_t *) *grid)->comm;
    f_get_gridinfo(grid, &iam, &nprow, &npcol);
    nprocs = nprow * npcol;
			   
    c2f_zcreate_matrix_x_b(fname, *nrhs, nprocs, slucomm,
    	                   (SuperMatrix *) *A, m, n, nnz, b, ldb, xtrue, ldx);
}

void f_zcreate_matrix_x_b_3d(char *fname, fptr *A, int *m, int *n, int_t *nnz,
		           int *nrhs, doublecomplex *b, int *ldb,
		           doublecomplex *xtrue, int *ldx, fptr *grid)
{
    extern int c2f_zcreate_matrix_x_b(char *fname, int nrhs, int nprocs,
    	                   MPI_Comm, SuperMatrix *A, int *m_g, int *n_g,
			   int_t *nnz_g, doublecomplex *rhs, int *ldb,
			   doublecomplex *x, int *ldx);
    extern void f_get_gridinfo3d(fptr *grid, int *iam,
                                 int *nprow, int *npcol, int *npdep);

    int iam, nprocs;
    int nprow, npcol, npdep;
    MPI_Comm slucomm = ((gridinfo3d_t *) *grid)->comm;
    f_get_gridinfo3d(grid, &iam, &nprow, &npcol, &npdep);
    nprocs = nprow * npcol * npdep;
			   
    c2f_zcreate_matrix_x_b(fname, *nrhs, nprocs, slucomm,
    	                   (SuperMatrix *) *A, m, n, nnz, b, ldb, xtrue, ldx);
}

