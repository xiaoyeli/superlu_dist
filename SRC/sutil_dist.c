/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Several matrix utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 * October 5, 2021
 */

#include <math.h>
#include "superlu_sdefs.h"

void
sCreate_CompCol_Matrix_dist(SuperMatrix *A, int_t m, int_t n, int_t nnz,
			    float *nzval, int_t *rowind, int_t *colptr,
			    Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NCformat *Astore;

    A->Stype = stype;
    A->Dtype = dtype;
    A->Mtype = mtype;
    A->nrow = m;
    A->ncol = n;
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");
    Astore = (NCformat *) A->Store;
    Astore->nnz = nnz;
    Astore->nzval = nzval;
    Astore->rowind = rowind;
    Astore->colptr = colptr;
}

void
sCreate_CompRowLoc_Matrix_dist(SuperMatrix *A, int_t m, int_t n,
			       int_t nnz_loc, int_t m_loc, int_t fst_row,
			       float *nzval, int_t *colind, int_t *rowptr,
			       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NRformat_loc *Astore;

    A->Stype = stype;
    A->Dtype = dtype;
    A->Mtype = mtype;
    A->nrow = m;
    A->ncol = n;
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat_loc) );
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");
    Astore = (NRformat_loc *) A->Store;
    Astore->nnz_loc = nnz_loc;
    Astore->fst_row = fst_row;
    Astore->m_loc = m_loc;
    Astore->nzval = nzval;
    Astore->colind = colind;
    Astore->rowptr = rowptr;
}

/*! \brief Convert a row compressed storage into a column compressed storage.
 */
void
sCompRow_to_CompCol_dist(int_t m, int_t n, int_t nnz,
                         float *a, int_t *colind, int_t *rowptr,
                         float **at, int_t **rowind, int_t **colptr)
{
    register int i, j, col, relpos;
    int_t *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = (float *) floatMalloc_dist(nnz);
    *rowind = intMalloc_dist(nnz);
    *colptr = intMalloc_dist(n+1);
    marker = intCalloc_dist(n);

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
	for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
	(*colptr)[j+1] = (*colptr)[j] + marker[j];
	marker[j] = (*colptr)[j];
    }

    /* Transfer the matrix into the compressed column storage. */
    for (i = 0; i < m; ++i) {
	for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
	    col = colind[j];
	    relpos = marker[col];
	    (*rowind)[relpos] = i;
	    (*at)[relpos] = a[j];
	    ++marker[col];
	}
    }

    SUPERLU_FREE(marker);
}

/*! \brief Copy matrix A into matrix B. */
void
sCopy_CompCol_Matrix_dist(SuperMatrix *A, SuperMatrix *B)
{
    NCformat *Astore, *Bstore;
    int      ncol, nnz, i;

    B->Stype = A->Stype;
    B->Dtype = A->Dtype;
    B->Mtype = A->Mtype;
    B->nrow  = A->nrow;;
    B->ncol  = ncol = A->ncol;
    Astore   = (NCformat *) A->Store;
    Bstore   = (NCformat *) B->Store;
    Bstore->nnz = nnz = Astore->nnz;
    for (i = 0; i < nnz; ++i)
	((float *)Bstore->nzval)[i] = ((float *)Astore->nzval)[i];
    for (i = 0; i < nnz; ++i) Bstore->rowind[i] = Astore->rowind[i];
    for (i = 0; i <= ncol; ++i) Bstore->colptr[i] = Astore->colptr[i];
}


void sPrint_CompCol_Matrix_dist(SuperMatrix *A)
{
    NCformat     *Astore;
    register int i;
    float       *dp;

    printf("\nCompCol matrix: ");
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    Astore = (NCformat *) A->Store;
    printf("nrow %lld, ncol %lld, nnz %lld\n", (long long) A->nrow,
	    (long long) A->ncol, (long long) Astore->nnz);
    if ( (dp = (float *) Astore->nzval) != NULL ) {
        printf("nzval:\n");
        for (i = 0; i < Astore->nnz; ++i) printf("%f  ", dp[i]);
    }
    printf("\nrowind:\n");
    for (i = 0; i < Astore->nnz; ++i)
        printf("%lld  ", (long long) Astore->rowind[i]);
    printf("\ncolptr:\n");
    for (i = 0; i <= A->ncol; ++i)
        printf("%lld  ", (long long) Astore->colptr[i]);
    printf("\nend CompCol matrix.\n");
}

void sPrint_Dense_Matrix_dist(SuperMatrix *A)
{
    DNformat     *Astore;
    register int i;
    float       *dp;

    printf("\nDense matrix: ");
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    Astore = (DNformat *) A->Store;
    dp = (float *) Astore->nzval;
    printf("nrow %lld, ncol %lld, lda %lld\n",
        (long long) A->nrow, (long long) A->ncol, (long long) Astore->lda);
    printf("\nnzval: ");
    for (i = 0; i < A->nrow; ++i) printf("%f  ", dp[i]);
    printf("\nend Dense matrix.\n");
}

int sPrint_CompRowLoc_Matrix_dist(SuperMatrix *A)
{
    NRformat_loc  *Astore;
    int_t  nnz_loc, m_loc;
    float  *dp;

    printf("\n==== CompRowLoc matrix: ");
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    Astore = (NRformat_loc *) A->Store;
    printf("nrow %ld, ncol %ld\n",
            (long int) A->nrow, (long int) A->ncol);
    nnz_loc = Astore->nnz_loc; m_loc = Astore->m_loc;
    printf("nnz_loc %ld, m_loc %ld, fst_row %ld\n", (long int) nnz_loc,
            (long int) m_loc, (long int) Astore->fst_row);
    PrintInt10("rowptr", m_loc+1, Astore->rowptr);
    PrintInt10("colind", nnz_loc, Astore->colind);
    if ( (dp = (float *) Astore->nzval) != NULL )
        Printfloat5("nzval", nnz_loc, dp);
    printf("==== end CompRowLoc matrix\n");
    return 0;
}

int file_sPrint_CompRowLoc_Matrix_dist(FILE *fp, SuperMatrix *A)
{
    NRformat_loc     *Astore;
    int_t  nnz_loc, m_loc;
    float       *dp;

    fprintf(fp, "\n==== CompRowLoc matrix: ");
    fprintf(fp, "Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    Astore = (NRformat_loc *) A->Store;
    fprintf(fp, "nrow %ld, ncol %ld\n", (long int) A->nrow, (long int) A->ncol);
    nnz_loc = Astore->nnz_loc; m_loc = Astore->m_loc;
    fprintf(fp, "nnz_loc %ld, m_loc %ld, fst_row %ld\n", (long int) nnz_loc,
            (long int) m_loc, (long int) Astore->fst_row);
    file_PrintInt10(fp, "rowptr", m_loc+1, Astore->rowptr);
    file_PrintInt10(fp, "colind", nnz_loc, Astore->colind);
    if ( (dp = (float *) Astore->nzval) != NULL )
        file_Printfloat5(fp, "nzval", nnz_loc, dp);
    fprintf(fp, "==== end CompRowLoc matrix\n");
    return 0;
}

void
sCreate_Dense_Matrix_dist(SuperMatrix *X, int_t m, int_t n, float *x,
			  int_t ldx, Stype_t stype, Dtype_t dtype,
			  Mtype_t mtype)
{
    DNformat    *Xstore;

    X->Stype = stype;
    X->Dtype = dtype;
    X->Mtype = mtype;
    X->nrow = m;
    X->ncol = n;
    X->Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );
    if ( !(X->Store) ) ABORT("SUPERLU_MALLOC fails for X->Store");
    Xstore = (DNformat *) X->Store;
    Xstore->lda = ldx;
    Xstore->nzval = (float *) x;
}

void
sCopy_Dense_Matrix_dist(int_t M, int_t N, float *X, int_t ldx,
			float *Y, int_t ldy)
{
/*! \brief
 *
 * <pre>
 *  Purpose
 *  =======
 *
 *  Copies a two-dimensional matrix X to another matrix Y.
 * </pre>
 */
    int    i, j;

    for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i)
            Y[i + j*ldy] = X[i + j*ldx];
}

void
sCreate_SuperNode_Matrix_dist(SuperMatrix *L, int_t m, int_t n, int_t nnz,
			      float *nzval, int_t *nzval_colptr,
			      int_t *rowind, int_t *rowind_colptr,
			      int_t *col_to_sup, int_t *sup_to_col,
			      Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    SCformat *Lstore;

    L->Stype = stype;
    L->Dtype = dtype;
    L->Mtype = mtype;
    L->nrow = m;
    L->ncol = n;
    L->Store = (void *) SUPERLU_MALLOC( sizeof(SCformat) );
    if ( !(L->Store) ) ABORT("SUPERLU_MALLOC fails for L->Store");
    Lstore = L->Store;
    Lstore->nnz = nnz;
    Lstore->nsuper = col_to_sup[n];
    Lstore->nzval = nzval;
    Lstore->nzval_colptr = nzval_colptr;
    Lstore->rowind = rowind;
    Lstore->rowind_colptr = rowind_colptr;
    Lstore->col_to_sup = col_to_sup;
    Lstore->sup_to_col = sup_to_col;

}

/**** The following utilities are added per request of SUNDIALS ****/

/*! \brief Clone: Allocate memory for a new matrix B, which is of the same type
 *  and shape as A.
 *  The clone operation would copy all the non-pointer structure members like
 *  nrow, ncol, Stype, Dtype, Mtype from A and allocate a new nested Store
 *  structure. It would also copy nnz_loc, m_loc, fst_row from A->Store
 *  into B->Store. It does not copy the matrix entries, row pointers,
 *  or column indices.
 */
void sClone_CompRowLoc_Matrix_dist(SuperMatrix *A, SuperMatrix *B)
{
    NRformat_loc  *Astore, *Bstore;

    B->Stype = A->Stype;
    B->Dtype = A->Dtype;
    B->Mtype = A->Mtype;
    B->nrow  = A->nrow;;
    B->ncol  = A->ncol;
    Astore   = (NRformat_loc *) A->Store;
    B->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat_loc) );
    if ( !(B->Store) ) ABORT("SUPERLU_MALLOC fails for B->Store");
    Bstore = (NRformat_loc *) B->Store;

    Bstore->nnz_loc = Astore->nnz_loc;
    Bstore->m_loc = Astore->m_loc;
    Bstore->fst_row = Astore->fst_row;
    if ( !(Bstore->nzval = (float *) floatMalloc_dist(Bstore->nnz_loc)) )
	ABORT("floatMalloc_dist fails for Bstore->nzval");
    if ( !(Bstore->colind = (int_t *) intMalloc_dist(Bstore->nnz_loc)) )
	ABORT("intMalloc_dist fails for Bstore->colind");
    if ( !(Bstore->rowptr = (int_t *) intMalloc_dist(Bstore->m_loc + 1)) )
	ABORT("intMalloc_dist fails for Bstore->rowptr");

    return;
}

/* \brief Copy: copies all entries, row pointers, and column indices of
 *  a matrix into another matrix of the same type,
 *  B_{i,j}=A_{i,j}, for i,j=1,...,n
 */
void sCopy_CompRowLoc_Matrix_dist(SuperMatrix *A, SuperMatrix *B)
{
    NRformat_loc  *Astore, *Bstore;

    Astore = (NRformat_loc *) A->Store;
    Bstore = (NRformat_loc *) B->Store;

    memcpy(Bstore->nzval, Astore->nzval, Astore->nnz_loc * sizeof(float));
    memcpy(Bstore->colind, Astore->colind, Astore->nnz_loc * sizeof(int_t));
    memcpy(Bstore->rowptr, Astore->rowptr, (Astore->m_loc+1) * sizeof(int_t));

    return;
}

/*! \brief Sets all entries of a matrix to zero, A_{i,j}=0, for i,j=1,..,n */
void sZero_CompRowLoc_Matrix_dist(SuperMatrix *A)
{
    float zero = 0.0;
    NRformat_loc  *Astore = A->Store;
    float *aval;
    int_t i;

    aval = (float *) Astore->nzval;
    for (i = 0; i < Astore->nnz_loc; ++i) aval[i] = zero;

    return;
}

/*! \brief Scale and add I: scales a matrix and adds an identity.
 *  A_{i,j} = c * A_{i,j} + \delta_{i,j} for i,j=1,...,n and
 *  \delta_{i,j} is the Kronecker delta.
 */
void sScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *A, float c)
{
    float one = 1.0;
    NRformat_loc  *Astore = A->Store;
    float *aval = (float *) Astore->nzval;
    int i, j;
    float temp;

    for (i = 0; i < Astore->m_loc; ++i) { /* Loop through each row */
        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
            if ( (Astore->fst_row + i) == Astore->colind[j] ) {  /* diagonal */
                temp = aval[j] * c;
                aval[j] = temp + one;
            } else {
                aval[j] *= c;
	   }
        }
    }

    return;
}

/*! \brief Scale and add: adds a scalar multiple of one matrix to another.
 *  A_{i,j} = c * A_{i,j} + B_{i,j}$ for i,j=1,...,n
 */
void sScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *A, SuperMatrix *B, float c)
{
    NRformat_loc  *Astore = A->Store;
    NRformat_loc  *Bstore = B->Store;
    float *aval = (float *) Astore->nzval, *bval = (float *) Bstore->nzval;
    int_t i;
    float temp;

    for (i = 0; i < Astore->nnz_loc; ++i) { /* Loop through each nonzero */
        aval[i] = c * aval[i] + bval[i];
    }

    return;
}
/**** end utilities added for SUNDIALS ****/

/**** end additions for SUNDIALS ****/

/*! \brief Find max(abs(L(i,j)))
 */
float sMaxAbsLij(int iam, int n, Glu_persist_t *Glu_persist,
		 sLUstruct_t *LUstruct, gridinfo_t *grid)
{
    register int extra, gb, j, lb, nsupc, nsupr, ncb;
    register int_t k, mycol, r;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;
    int nsupers = Glu_persist->supno[n-1] + 1;
    float lmax = 0.0, lmax_loc = 0.0;

    ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (j = 0; j < nsupc; ++j) {
                for (r = 0; r < nsupr; ++r) {
		    lmax_loc = SUPERLU_MAX(lmax_loc, fabs(nzval[r + j*nsupr]));
		}
            }
	}
    } /* end for */

    /* Reduce max(abs(Uij)) from all processes, to process 0 */
    MPI_Reduce (&lmax_loc, &lmax, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm);

    /* Reduce sum of the row counts from each process. */

    return (lmax);
} /* sMaxAbsLij */

/*! \brief Find max(abs(U(i,j)))
 */
float sMaxAbsUij(int iam, int n, Glu_persist_t *Glu_persist,
		 sLUstruct_t *LUstruct, gridinfo_t *grid)
{
    sLocalLU_t *Llu = LUstruct->Llu;
    register int c, extra, jb, k, lb, len, nb, nrb, nsupc;
    register int myrow, r, j, nsupers;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;
    float umax = 0.0, umax_loc = 0.0;

    nsupers = Glu_persist->supno[n-1] + 1;
    nrb = nsupers / grid->nprow;
    extra = nsupers % grid->nprow;
    myrow = MYROW( iam, grid );
    if ( myrow < extra ) ++nrb;

    // Sherry: Can also compute the maximum row count ...
    for (lb = 0; lb < nrb; ++lb) {
	index = Llu->Ufstnz_br_ptr[lb];
	if ( index ) { /* Not an empty block row */
	    nzval = Llu->Unzval_br_ptr[lb];
	    nb = index[0]; /* number of blocks */
	    r  = 0;
	    for (c = 0, k = BR_HEADER; c < nb; ++c) {
	        jb = index[k];    /* block number */
		len = index[k+1]; /* number of nonzeros in the block */
		nsupc = SuperSize( jb );
		for (j = r; j < r + len; ++j) 
		    umax_loc = SUPERLU_MAX(umax_loc, fabs(nzval[j]));
		k += UB_DESCRIPTOR + nsupc;
		r += len;
	    }
	}
    }
    
    /* Reduce max(abs(Uij)) from all processes, to process 0 */
    MPI_Reduce (&umax_loc, &umax, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm);

    /* Reduce sum of the row counts from each process. */

    return (umax);

} /* sMaxAbsUij */


/*! \brief Allocate storage in ScalePermstruct */
void sScalePermstructInit(const int_t m, const int_t n,
                         sScalePermstruct_t *ScalePermstruct)
{
    ScalePermstruct->DiagScale = NOEQUIL;
    if ( !(ScalePermstruct->perm_r = intMalloc_dist(m)) )
        ABORT("Malloc fails for perm_r[].");
    if ( !(ScalePermstruct->perm_c = intMalloc_dist(n)) )
        ABORT("Malloc fails for perm_c[].");
}

/*! \brief Deallocate ScalePermstruct */
void sScalePermstructFree(sScalePermstruct_t *ScalePermstruct)
{
    SUPERLU_FREE(ScalePermstruct->perm_r);
    SUPERLU_FREE(ScalePermstruct->perm_c);
    switch ( ScalePermstruct->DiagScale ) {
      case ROW:
        SUPERLU_FREE(ScalePermstruct->R);
        break;
      case COL:
        SUPERLU_FREE(ScalePermstruct->C);
        break;
      case BOTH:
        SUPERLU_FREE(ScalePermstruct->R);
        SUPERLU_FREE(ScalePermstruct->C);
        break;
      default: break;
    }
}

/*
 * The following are from 3D code p3dcomm.c
 */

int sAllocGlu_3d(int_t n, int_t nsupers, sLUstruct_t * LUstruct)
{
    /*broadcasting Glu_persist*/
    LUstruct->Glu_persist->xsup  = intMalloc_dist(nsupers+1); //INT_T_ALLOC(nsupers+1);
    LUstruct->Glu_persist->supno = intMalloc_dist(n); //INT_T_ALLOC(n);
    return 0;
}

// Sherry added
/* Free the replicated data on 3D process layer that is not grid-0 */
int sDeAllocGlu_3d(sLUstruct_t * LUstruct)
{
    SUPERLU_FREE(LUstruct->Glu_persist->xsup);
    SUPERLU_FREE(LUstruct->Glu_persist->supno);
    return 0;
}

/* Free the replicated data on 3D process layer that is not grid-0 */
int sDeAllocLlu_3d(int_t n, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{
    int i, nbc, nbr, nsupers;
    sLocalLU_t *Llu = LUstruct->Llu;

    nsupers = (LUstruct->Glu_persist)->supno[n-1] + 1;

    nbc = CEILING(nsupers, grid3d->npcol);
    for (i = 0; i < nbc; ++i) 
	if ( Llu->Lrowind_bc_ptr[i] ) {
	    SUPERLU_FREE (Llu->Lrowind_bc_ptr[i]);
	    SUPERLU_FREE (Llu->Lnzval_bc_ptr[i]);
	}
    SUPERLU_FREE (Llu->Lrowind_bc_ptr);
    SUPERLU_FREE (Llu->Lnzval_bc_ptr);

    nbr = CEILING(nsupers, grid3d->nprow);
    for (i = 0; i < nbr; ++i)
	if ( Llu->Ufstnz_br_ptr[i] ) {
	    SUPERLU_FREE (Llu->Ufstnz_br_ptr[i]);
	    SUPERLU_FREE (Llu->Unzval_br_ptr[i]);
	}
    SUPERLU_FREE (Llu->Ufstnz_br_ptr);
    SUPERLU_FREE (Llu->Unzval_br_ptr);

    /* The following can be freed after factorization. */
    SUPERLU_FREE(Llu->ToRecv);
    SUPERLU_FREE(Llu->ToSendD);
    for (i = 0; i < nbc; ++i) SUPERLU_FREE(Llu->ToSendR[i]);
    SUPERLU_FREE(Llu->ToSendR);
    return 0;
} /* sDeAllocLlu_3d */

/**** Other utilities ****/
void
sGenXtrue_dist(int_t n, int_t nrhs, float *x, int_t ldx)
{
    int  i, j;
    double exponent, tau; /* See TOMS paper on ItRef (LAWN165); testing code: 
			     Codes/UCB-itref-xblas-etc/xiaoye/itref/driver.c  */
    double r;
    float xmax = 0.0, xmin = 1.0e6;

#if 0    
    // NEED TO REMOVE THE FOLLOWING
    for (j = 0; j < nrhs; ++j) {
	for (i = 0; i < n; ++i) {
	  x[i + j*ldx] = 1.0;
	}
    }
    return;
    /////////////////////////////
#endif
    
    exponent = (double)rand() / (double)((unsigned)RAND_MAX + 1); /* uniform in [0,1) */
#if 1
    tau = pow(2.0, 12.0 * exponent);  // 24.0
#else
    tau = 5.0;
#endif
    //printf("new dGenXtrue, tau %e\n", tau);
    
    r = (double)rand() / (double)((unsigned)RAND_MAX + 1); /* uniform in [0,1) */
    r = r + 0.5; /* uniform in (0.5, 1.5) */

    for (j = 0; j < nrhs; ++j) {
        xmax = 0.0, xmin = 1.0e6;
	for (i = 0; i < n; ++i) {
#if 1
	  x[i + j*ldx] = pow(tau, - ((double)i / (n-1))) * r;

	  //if (i % 2) x[i + j*ldx] = 1.0; else x[i + j*ldx] = -1.0;
#else
	  x[i + j*ldx] = rand() / (double)((unsigned)RAND_MAX + 1); /* uniform in [0,1) */
#endif
	  xmax = SUPERLU_MAX( xmax, x[i + j*ldx] );
	  xmin = SUPERLU_MIN( xmin, x[i + j*ldx] );
	}
#if ( PRNTlevel>=1 )
	int iam;
	MPI_Comm_rank(MPI_COMM_WORLD, &iam);
	if (iam==0) {
	  printf(".. sGenXtrue: xmax %e, xmin %e\n", xmax, xmin);
	  fflush(stdout);
	}
#endif	
    }
}

/*! \brief Let rhs[i] = sum of i-th row of A, so the solution vector is all 1's
 */
void
sFillRHS_dist(char *trans, int_t nrhs, float *x, int_t ldx,
	      SuperMatrix *A, float *rhs, int_t ldb)
{
    float one = 1.0;
    float zero = 0.0;

    sp_sgemm_dist(trans, nrhs, one, A, x, ldx, zero, rhs, ldb);

}

/*! \brief Fills a float precision array with a given value.
 */
void
sfill_dist(float *a, int_t alen, float dval)
{
    register int_t i;
    for (i = 0; i < alen; i++) a[i] = dval;
}



/*! \brief Check the inf-norm of the error vector
 */
void sinf_norm_error_dist(int_t n, int_t nrhs, float *x, int_t ldx,
			  float *xtrue, int_t ldxtrue,
                          gridinfo_t *grid)
{
    double err, xnorm;
    float *x_work, *xtrue_work;
    int i, j;

    for (j = 0; j < nrhs; j++) {
      x_work = &x[j*ldx];
      xtrue_work = &xtrue[j*ldxtrue];
      err = xnorm = 0.0;
      for (i = 0; i < n; i++) {
	err = SUPERLU_MAX(err, fabs(x_work[i] - xtrue_work[i]));
	xnorm = SUPERLU_MAX(xnorm, fabs(x_work[i]));
      }
      err = err / xnorm;
      printf("\tRHS %2d: ||X-Xtrue||/||X|| = %e\n", j, err);
    }
}

void Printfloat5(char *name, int_t len, float *x)
{
    register int i;

    printf("%10s:", name);
    for (i = 0; i < len; ++i) {
	if ( i % 5 == 0 ) printf("\n[%d-%d] ",  i, i+4);
	printf("%14e", x[i]);
    }
    printf("\n\n");
}

int file_Printfloat5(FILE *fp, char *name, int_t len, float *x)
{
    register int_t i;

    fprintf(fp, "%10s:", name);
    for (i = 0; i < len; ++i) {
	if ( i % 5 == 0 ) fprintf(fp, "\n[%ld-%ld] ", (long int) i, (long int) i+4);
	fprintf(fp, "%14e", x[i]);
    }
    fprintf(fp, "\n");
    return 0;
}

/*! \brief Print the blocks in the factored matrix L.
 */
void sPrintLblocks(int iam, int_t nsupers, gridinfo_t *grid,
		  Glu_persist_t *Glu_persist, sLocalLU_t *Llu)
{
    register int c, extra, gb, j, lb, nsupc, nsupr, len, nb, ncb;
    register int_t k, mycol, r;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;

    printf("\n[%d] L BLOCKS IN COLUMN-MAJOR ORDER -->\n", iam);
    ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nb = index[0];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    printf("[%d] block column %d (local # %d), nsupc %d, # row blocks %d\n",
		   iam, gb, lb, nsupc, nb);
	    for (c = 0, k = BC_HEADER, r = 0; c < nb; ++c) {
		len = index[k+1];
		printf("[%d] row-block %d: block # " IFMT "\tlength %d\n",
		       iam, c, index[k], len);
		PrintInt10("lsub", len, &index[k+LB_DESCRIPTOR]);
		for (j = 0; j < nsupc; ++j) {
		    Printfloat5("nzval", len, &nzval[r + j*nsupr]);
		}
		k += LB_DESCRIPTOR + len;
		r += len;
	    }
	}
	printf("(%d)", iam);
 	PrintInt32("ToSendR[]", grid->npcol, Llu->ToSendR[lb]);
	PrintInt32("fsendx_plist[]", grid->nprow, Llu->fsendx_plist[lb]);
    }
    printf("nfrecvx %d\n", Llu->nfrecvx);
    k = CEILING( nsupers, grid->nprow );
    PrintInt32("fmod", k, Llu->fmod);

} /* SPRINTLBLOCKS */


/*! \brief Sets all entries of matrix L to zero.
 */
void sZeroLblocks(int iam, int n, gridinfo_t *grid, sLUstruct_t *LUstruct)
{
    float zero = 0.0;
    register int extra, gb, j, lb, nsupc, nsupr, ncb;
    register int k, mycol, r;
    sLocalLU_t *Llu = LUstruct->Llu;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;
    int nsupers = Glu_persist->supno[n-1] + 1;

    ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (j = 0; j < nsupc; ++j) {
                for (r = 0; r < nsupr; ++r) {
                    nzval[r + j*nsupr] = zero;
		}
            }
	}
    }
} /* end sZeroLblocks */


/*! \brief Dump the factored matrix L using matlab triple-let format
 */
void sDumpLblocks(int iam, int_t nsupers, gridinfo_t *grid,
		  Glu_persist_t *Glu_persist, sLocalLU_t *Llu)
{
    register int c, extra, gb, j, i, lb, nsupc, nsupr, len, nb, ncb;
    int k, mycol, r, n, nmax;
    int_t nnzL;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;
	char filename[256];
	FILE *fp, *fopen();

	// assert(grid->npcol*grid->nprow==1);

	// count nonzeros in the first pass
	nnzL = 0;
	n = 0;
    ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nb = index[0];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (c = 0, k = BC_HEADER, r = 0; c < nb; ++c) {
		len = index[k+1];

		for (j = 0; j < nsupc; ++j) {
		for (i=0; i<len; ++i){

		if(index[k+LB_DESCRIPTOR+i]+1>=xsup[gb]+j+1){
			nnzL ++;
			nmax = SUPERLU_MAX(n,index[k+LB_DESCRIPTOR+i]+1);
			n = nmax;
		}

		}
		}
		k += LB_DESCRIPTOR + len;
		r += len;
	    }
	}
    }
	MPI_Allreduce(MPI_IN_PLACE,&nnzL,1,mpi_int_t,MPI_SUM,grid->comm);
	MPI_Allreduce(MPI_IN_PLACE,&n,1,mpi_int_t,MPI_MAX,grid->comm);

	snprintf(filename, sizeof(filename), "%s-%d", "L", iam);
    printf("Dumping L factor to --> %s\n", filename);
 	if ( !(fp = fopen(filename, "w")) ) {
			ABORT("File open failed");
		}

	if(grid->iam==0){
		fprintf(fp, "%d %d " IFMT "\n", n,n,nnzL);
	}

     ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nb = index[0];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (c = 0, k = BC_HEADER, r = 0; c < nb; ++c) {
		len = index[k+1];

		for (j = 0; j < nsupc; ++j) {
		for (i=0; i<len; ++i){
			fprintf(fp, IFMT IFMT " %e\n", index[k+LB_DESCRIPTOR+i]+1, xsup[gb]+j+1, (double)iam);
#if 0
			fprintf(fp, IFMT IFMT " %e\n", index[k+LB_DESCRIPTOR+i]+1, xsup[gb]+j+1, nzval[r +i+ j*nsupr]);
#endif
		}
		}
		k += LB_DESCRIPTOR + len;
		r += len;
	    }
	}
    }
 	fclose(fp);

} /* sDumpLblocks */


/*! \brief Print the blocks in the factored matrix U.
 */
void sPrintUblocks(int iam, int_t nsupers, gridinfo_t *grid,
		  Glu_persist_t *Glu_persist, sLocalLU_t *Llu)
{
    register int c, extra, jb, k, lb, len, nb, nrb, nsupc;
    register int_t myrow, r;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;

    printf("\n[%d] U BLOCKS IN ROW-MAJOR ORDER -->\n", iam);
    nrb = nsupers / grid->nprow;
    extra = nsupers % grid->nprow;
    myrow = MYROW( iam, grid );
    if ( myrow < extra ) ++nrb;
    for (lb = 0; lb < nrb; ++lb) {
	index = Llu->Ufstnz_br_ptr[lb];
	if ( index ) { /* Not an empty row */
	    nzval = Llu->Unzval_br_ptr[lb];
	    nb = index[0];
	    printf("[%d] block row " IFMT " (local # %d), # column blocks %d\n",
		   iam, lb*grid->nprow+myrow, lb, nb);
	    r  = 0;
	    for (c = 0, k = BR_HEADER; c < nb; ++c) {
		jb = index[k];
		len = index[k+1];
		printf("[%d] col-block %d: block # %d\tlength " IFMT "\n",
		       iam, c, jb, index[k+1]);
		nsupc = SuperSize( jb );
		PrintInt10("fstnz", nsupc, &index[k+UB_DESCRIPTOR]);
		Printfloat5("nzval", len, &nzval[r]);
		k += UB_DESCRIPTOR + nsupc;
		r += len;
	    }

	    printf("[%d] ToSendD[] %d\n", iam, Llu->ToSendD[lb]);
	}
    }
} /* end sPrintUlocks */

/*! \brief Sets all entries of matrix U to zero.
 */
void sZeroUblocks(int iam, int n, gridinfo_t *grid, sLUstruct_t *LUstruct)
{
    float zero = 0.0;
    register int i, extra, lb, len, nrb;
    register int myrow, r;
    sLocalLU_t *Llu = LUstruct->Llu;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    float *nzval;
    int nsupers = Glu_persist->supno[n-1] + 1;

    nrb = nsupers / grid->nprow;
    extra = nsupers % grid->nprow;
    myrow = MYROW( iam, grid );
    if ( myrow < extra ) ++nrb;
    for (lb = 0; lb < nrb; ++lb) {
	index = Llu->Ufstnz_br_ptr[lb];
	if ( index ) { /* Not an empty row */
	    nzval = Llu->Unzval_br_ptr[lb];
	    len = index[1];  // number of entries in nzval[];
	    for (i = 0; i < len; ++i) {
	        nzval[i] = zero;
	    }
	}
    }
} /* end sZeroUlocks */

int
sprint_gsmv_comm(FILE *fp, int_t m_loc, psgsmv_comm_t *gsmv_comm,
                 gridinfo_t *grid)
{
  int_t procs = grid->nprow*grid->npcol;
  fprintf(fp, "TotalIndSend " IFMT "\tTotalValSend " IFMT "\n", gsmv_comm->TotalIndSend,
	  gsmv_comm->TotalValSend);
  file_PrintInt10(fp, "extern_start", m_loc, gsmv_comm->extern_start);
  file_PrintInt10(fp, "ind_tosend", gsmv_comm->TotalIndSend, gsmv_comm->ind_tosend);
  file_PrintInt10(fp, "ind_torecv", gsmv_comm->TotalValSend, gsmv_comm->ind_torecv);
  file_PrintInt10(fp, "ptr_ind_tosend", procs+1, gsmv_comm->ptr_ind_tosend);
  file_PrintInt10(fp, "ptr_ind_torecv", procs+1, gsmv_comm->ptr_ind_torecv);
  file_PrintInt32(fp, "SendCounts", procs, gsmv_comm->SendCounts);
  file_PrintInt32(fp, "RecvCounts", procs, gsmv_comm->RecvCounts);
  return 0;
}


void
sGenXtrueRHS(int nrhs, SuperMatrix *A, Glu_persist_t *Glu_persist,
	    gridinfo_t *grid, float **xact, int *ldx, float **b, int *ldb)
{
    int_t gb, gbrow, i, iam, irow, j, lb, lsup, myrow, n, nlrows,
          nsupr, nsupers, rel;
    int_t *supno, *xsup, *lxsup;
    float *x, *bb;
    NCformat *Astore;
    float   *aval;

    n = A->ncol;
    *ldb = 0;
    supno = Glu_persist->supno;
    xsup = Glu_persist->xsup;
    nsupers = supno[n-1] + 1;
    iam = grid->iam;
    myrow = MYROW( iam, grid );
    Astore = (NCformat *) A->Store;
    aval = Astore->nzval;
    lb = CEILING( nsupers, grid->nprow ) + 1;
    if ( !(lxsup = intMalloc_dist(lb)) )
	ABORT("Malloc fails for lxsup[].");

    lsup = 0;
    nlrows = 0;
    for (j = 0; j < nsupers; ++j) {
	i = PROW( j, grid );
	if ( myrow == i ) {
	    nsupr = SuperSize( j );
	    *ldb += nsupr;
	    lxsup[lsup++] = nlrows;
	    nlrows += nsupr;
	}
    }
    *ldx = n;
    if ( !(x = floatMalloc_dist(((size_t)*ldx) * nrhs)) )
	ABORT("Malloc fails for x[].");
    if ( !(bb = floatCalloc_dist(*ldb * nrhs)) )
	ABORT("Calloc fails for bb[].");
    for (j = 0; j < nrhs; ++j)
	for (i = 0; i < n; ++i) x[i + j*(*ldx)] = 1.0;

    /* Form b = A*x. */
    for (j = 0; j < n; ++j)
	for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
	    irow = Astore->rowind[i];
	    gb = supno[irow];
	    gbrow = PROW( gb, grid );
	    if ( myrow == gbrow ) {
		rel = irow - xsup[gb];
		lb = LBi( gb, grid );
		bb[lxsup[lb] + rel] += aval[i] * x[j];
	    }
	}

    /* Memory allocated but not freed: xact, b */
    *xact = x;
    *b = bb;

    SUPERLU_FREE(lxsup);

#if ( PRNTlevel>=2 )
    for (i = 0; i < grid->nprow*grid->npcol; ++i) {
	if ( iam == i ) {
	    printf("\n(%d)\n", iam);
	    Printfloat5("rhs", *ldb, *b);
	}
	MPI_Barrier( grid->comm );
    }
#endif

} /* GENXTRUERHS */

/* g5.rua
          b = A*x    y = L\b
   0      1	     1.0000
   1      0	     0.2500
   2      1	     1.0000
   3      2	     2.0000
   4      1	     1.7500
   5      1	     1.8917
   6      0	     1.1879
   7      2	     2.0000
   8      2	     2.0000
   9      1	     1.0000
   10     1	     1.7500
   11     0	          0
   12     1	     1.8750
   13     2	     2.0000
   14     1	     1.0000
   15     0	     0.2500
   16     1	     1.7667
   17     0	     0.6419
   18     1	     2.2504
   19     0	     1.1563
   20     0	     0.9069
   21     0	     1.4269
   22     1	     2.7510
   23     1	     2.2289
   24     0	     2.4332

   g6.rua
       b=A*x  y=L\b
    0    0         0
    1    1    1.0000
    2    1    1.0000
    3    2    2.5000
    4    0         0
    5    2    2.0000
    6    1    1.0000
    7    1    1.7500
    8    1    1.0000
    9    0    0.2500
   10    0    0.5667
   11    1    2.0787
   12    0    0.8011
   13    1    1.9838
   14    1    1.0000
   15    1    1.0000
   16    2    2.5000
   17    0    0.8571
   18    0         0
   19    1    1.0000
   20    0    0.2500
   21    1    1.0000
   22    2    2.0000
   23    1    1.7500
   24    1    1.8917
   25    0    1.1879
   26    0    0.8011
   27    1    1.9861
   28    1    2.0199
   29    0    1.3620
   30    0    0.6136
   31    1    2.3677
   32    0    1.1011
   33    0    1.5258
   34    0    1.7628
   35    0    2.1658
*/
