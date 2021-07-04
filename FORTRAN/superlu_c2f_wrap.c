

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

#include "superlu_defs.h"
#include "superlu_FCnames.h"

/* kind of integer to hold a pointer.
   Be sure to be consistent with that in superlupara.f90 */
#if 0
typedef int fptr;  /* 32-bit */
#else
typedef long long int fptr;  /* 64-bit */
#endif


/* some MPI implementations may require conversion between a Fortran
   communicator and a C communicator.  This routine is used to perform the
   conversion.  It may need different forms for different MPI libraries. */

/* NO_MPI2 should be defined on the compiler command line if the MPI
   library does not provide MPI_Comm_f2c */

MPI_Comm f2c_comm(int *f_comm)
{
#ifndef NO_MPI2

/* MPI 2 provides a standard way of doing this */
   return MPI_Comm_f2c((MPI_Fint)(*f_comm));
#else

/* will probably need some special cases here */
/* when in doubt, just return the input */
   return (MPI_Comm)(*f_comm);
#endif
}


/* functions that create memory for a struct and return a handle */

void f_create_gridinfo_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(gridinfo_t));
}

void f_create_gridinfo3d_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(gridinfo3d_t));
}

void f_create_options_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(superlu_dist_options_t));
}

void f_create_SuperMatrix_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(SuperMatrix));
}

void f_create_SuperLUStat_handle(fptr *handle)
{
   *handle = (fptr) SUPERLU_MALLOC(sizeof(SuperLUStat_t));
}

/* functions that free the memory allocated by the above functions */

void f_destroy_gridinfo_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_options_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_ScalePerm_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_LUstruct_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_SOLVEstruct_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_SuperMatrix_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

void f_destroy_SuperLUStat_handle(fptr *handle)
{
   SUPERLU_FREE((void *)*handle);
}

/* functions that get or set values in a C struct.
   This is not the complete set of structs for which a user might want
   to get/set a component, and there may be missing components. */

void f_get_gridinfo(fptr *grid, int *iam, int *nprow, int *npcol)
{
  *iam=((gridinfo_t *) *grid)->iam;
  *npcol=((gridinfo_t *) *grid)->npcol;
  *nprow=((gridinfo_t *) *grid)->nprow;
}

void f_get_gridinfo3d(fptr *grid, int *iam,
         	      int *nprow, int *npcol, int *npdep)
{
  *iam=((gridinfo3d_t *) *grid)->iam;
  *npcol=((gridinfo3d_t *) *grid)->npcol;
  *nprow=((gridinfo3d_t *) *grid)->nprow;
  *npdep=((gridinfo3d_t *) *grid)->npdep;
}

void f_get_SuperMatrix(fptr *A, int *nrow, int *ncol)
{
   *nrow = ((SuperMatrix *) *A)->nrow;
   *ncol = ((SuperMatrix *) *A)->ncol;
}

void f_set_SuperMatrix(fptr *A, int *nrow, int *ncol)
{
   ((SuperMatrix *) *A)->nrow = *nrow;
   ((SuperMatrix *) *A)->ncol = *ncol;
}

void f_get_CompRowLoc_Matrix(fptr *A, int *m, int *n, int_t *nnz_loc,
			     int *m_loc, int *fst_row)
{
  *m=((SuperMatrix *) *A)->nrow;
  *n=((SuperMatrix *) *A)->ncol;
  *m_loc=((NRformat_loc *) ((SuperMatrix *) *A)->Store)->m_loc;
  *nnz_loc=((NRformat_loc *) ((SuperMatrix *) *A)->Store)->nnz_loc;
  *fst_row=((NRformat_loc *) ((SuperMatrix *) *A)->Store)->fst_row;
}

void f_set_CompRowLoc_Matrix(fptr *A, int *m, int *n, int_t *nnz_loc,
			     int *m_loc, int *fst_row)
{
  NRformat_loc *Astore = ((SuperMatrix *) *A)->Store;

  ((SuperMatrix *) *A)->nrow = *m;
  ((SuperMatrix *) *A)->ncol = *n;
  Astore->m_loc = *m_loc;
  Astore->nnz_loc = *nnz_loc;
  Astore->fst_row = *fst_row;
}

void f_get_superlu_options(fptr *opt, int *Fact, int *Equil, int *ParSymbFact,
                           int *ColPerm, int *RowPerm, int *IterRefine,
			   int *Trans, int *ReplaceTinyPivot,
			   int *SolveInitialized, int *RefineInitialized,
			   int *PrintStat)
{
   *Fact = (int) ((superlu_dist_options_t *) *opt)->Fact;
   *Equil = (int) ((superlu_dist_options_t *) *opt)->Equil;
   *ParSymbFact = (int) ((superlu_dist_options_t *) *opt)->ParSymbFact;
   *ColPerm = (int) ((superlu_dist_options_t *) *opt)->ColPerm;
   *RowPerm = (int) ((superlu_dist_options_t *) *opt)->RowPerm;
   *IterRefine = (int) ((superlu_dist_options_t *) *opt)->IterRefine;
   *Trans = (int) ((superlu_dist_options_t *) *opt)->Trans;
   *ReplaceTinyPivot = (int) ((superlu_dist_options_t *) *opt)->ReplaceTinyPivot;
   *SolveInitialized = (int) ((superlu_dist_options_t *) *opt)->SolveInitialized;
   *RefineInitialized = (int) ((superlu_dist_options_t *) *opt)->RefineInitialized;
   *PrintStat = (int) ((superlu_dist_options_t *) *opt)->PrintStat;
}

void f_set_superlu_options(fptr *opt, int *Fact, int *Equil, int *ParSymbFact,
                           int *ColPerm, int *RowPerm, int *IterRefine,
			   int *Trans, int *ReplaceTinyPivot,
			   int *SolveInitialized, int *RefineInitialized,
			   int *PrintStat)
{
    superlu_dist_options_t *l_options = (superlu_dist_options_t*) *opt;
    l_options->Fact = (fact_t) *Fact;
   ((superlu_dist_options_t *) *opt)->Equil = (yes_no_t) *Equil;
   ((superlu_dist_options_t *) *opt)->ParSymbFact = (yes_no_t) *ParSymbFact;
   ((superlu_dist_options_t *) *opt)->ColPerm = (colperm_t) *ColPerm;
   ((superlu_dist_options_t *) *opt)->RowPerm = (rowperm_t) *RowPerm;
   ((superlu_dist_options_t *) *opt)->IterRefine = (IterRefine_t) *IterRefine;
   ((superlu_dist_options_t *) *opt)->Trans = (trans_t) *Trans;
   ((superlu_dist_options_t *) *opt)->ReplaceTinyPivot = (yes_no_t) *ReplaceTinyPivot;
   ((superlu_dist_options_t *) *opt)->SolveInitialized = (yes_no_t) *SolveInitialized;
   ((superlu_dist_options_t *) *opt)->RefineInitialized = (yes_no_t) *RefineInitialized;
   ((superlu_dist_options_t *) *opt)->PrintStat = (yes_no_t) *PrintStat;
}

/* wrappers for SuperLU functions */

void f_set_default_options(fptr *options)
{
   set_default_options_dist((superlu_dist_options_t *) *options);
}

void f_superlu_gridinit(int *Bcomm, int *nprow, int *npcol, fptr *grid)
{
   superlu_gridinit(f2c_comm(Bcomm), *nprow, *npcol, (gridinfo_t *) *grid);
}

void f_superlu_gridinit3d(int *Bcomm, int *nprow, int *npcol,
   			  int *npdep, fptr *grid)
{
    superlu_gridinit3d(f2c_comm(Bcomm), *nprow, *npcol, *npdep, (gridinfo3d_t *) *grid);
}

void f_superlu_gridmap(int *Bcomm, int *nprow, int *npcol, 
                       int *usermap, int *ldumap, fptr *grid)
{
   superlu_gridmap(f2c_comm(Bcomm), *nprow, *npcol, usermap, *ldumap,
		   (gridinfo_t *) *grid);
}

void f_superlu_gridexit(fptr *grid)
{
   superlu_gridexit((gridinfo_t *) *grid);
}

void f_PStatInit(fptr *stat)
{
   PStatInit((SuperLUStat_t *) *stat);
}

void f_PStatFree(fptr *stat)
{
   PStatFree((SuperLUStat_t *) *stat);
}

void f_Destroy_CompRowLoc_Mat_dist(fptr *A)
{
   Destroy_CompRowLoc_Matrix_dist((SuperMatrix *) *A);
}

void f_Destroy_SuperMat_Store_dist(fptr *A)
{
   Destroy_SuperMatrix_Store_dist((SuperMatrix *) *A);
}

/* Check malloc */

void f_check_malloc(int *iam)
{
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(*iam, "Check Malloc");
#endif
}
