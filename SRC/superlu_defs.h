/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Definitions which are precision-neutral
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 *
 * Modified:
 *     February 20, 2008
 *     October 11, 2014
 *     September 18, 2018  version 6.0
 *     February 8, 2019    version 6.1.1
 *     November 12, 2019   version 6.2.0
 *     October 23, 2020    version 6.4.0
 *     May 12, 2021        version 7.0.0
 *     October 5, 2021     version 7.1.0
 *     October 18, 2021    version 7.1.1
 *     December 12, 2021   version 7.2.0
 *     May 22, 2022        version 8.0.0
 * </pre>
 */

#ifndef __SUPERLU_DEFS /* allow multiple inclusions */
#define __SUPERLU_DEFS

/*
 * File name:	superlu_defs.h
 * Purpose:     Definitions which are precision-neutral
 */
#ifdef _CRAY
    #include <fortran.h>
#endif

#ifdef _OPENMP
   #include <omp.h>
#endif


#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
// #include <stdatomic.h>
#include <math.h>
#include <stdint.h>
//#include <malloc.h>  Sherry: not available on Mac OS
// /* Following is for vtune */
// #if 0
// #include <ittnotify.h>
// #define USE_VTUNE
// #endif
#if ( VTUNE>=1 )
#include <ittnotify.h>			 
#endif

/*************************************************************************
 * Constants
 **************************************************************************/
/*
 * You can support older version of SuperLU_DIST.
 * At compile-time, you can catch the new release as:
 *   #ifdef SUPERLU_DIST_MAIN_VERSION == 5
 *       use the new interface
 *   #else
 *       use the old interface
 *   #endif
 * Versions 4.x and earlier do not include a #define'd version numbers.
 */
#define SUPERLU_DIST_MAJOR_VERSION     8
#define SUPERLU_DIST_MINOR_VERSION     0
#define SUPERLU_DIST_PATCH_VERSION     0
#define SUPERLU_DIST_RELEASE_DATE      "May 22, 2022"

#include "superlu_dist_config.h"

#ifdef HAVE_CUDA
#define GPU_ACC
//#include "cublas_utils.h"
#endif

#ifdef HAVE_HIP
#ifndef GPU_ACC
#define GPU_ACC
#endif
#endif

#ifdef GPU_ACC
#include "gpu_api_utils.h"
#endif


/* Define my integer size int_t */
#ifdef _CRAY
  typedef short int_t;
  /*#undef int   Revert back to int of default size. */
  #define mpi_int_t   MPI_SHORT
#elif defined (_LONGINT)
  typedef int64_t int_t;
  #define mpi_int_t   MPI_LONG_LONG_INT
  #define IFMT "%lld"
#else /* Default */
  typedef int int_t;
  #define mpi_int_t   MPI_INT
  #define IFMT "%8d"
#endif


/* MPI C complex datatype */
#define SuperLU_MPI_COMPLEX         MPI_C_COMPLEX 
#define SuperLU_MPI_DOUBLE_COMPLEX  MPI_C_DOUBLE_COMPLEX

/* MPI_Datatype cannot be used in C typedef
typedef MPI_C_COMPLEX         SuperLU_MPI_COMPLEX;
typedef MPI_C_DOUBLE_COMPLEX  SuperLU_MPI_DOUBLE_COMPLEX;
*/

#include "superlu_FortranCInterface.h"
#include "superlu_FCnames.h"
#include "superlu_enum_consts.h"
#include "supermatrix.h"
#include "util_dist.h"
#include "psymbfact.h"


#define MAX_SUPER_SIZE 512   /* Sherry: moved from superlu_gpu.cu */


#define ISORT     /* NOTE: qsort() has bug on Mac */

/*********************************************************************** 
 * Constants
 ***********************************************************************/
/* 
 * For each block column of L, the index[] array contains both the row 
 * subscripts and the integers describing the size of the blocks.
 * The organization of index[] looks like:
 *
 *     [ BLOCK COLUMN HEADER (size BC_HEADER)
 *           number of blocks 
 *           number of row subscripts, i.e., LDA of nzval[]
 *       BLOCK 0                                        <----
 *           BLOCK DESCRIPTOR (of size LB_DESCRIPTOR)  |
 *               block number (global)                      |
 *               number of full rows in the block           |
 *           actual row subscripts                          |
 *       BLOCK 1                                            | Repeat ...
 *           BLOCK DESCRIPTOR                               | number of blocks
 *               block number (global)                      | 
 *               number of full rows in the block           |
 *           actual row subscripts                          |
 *       .                                                  |
 *       .                                                  |
 *       .                                              <----
 *     ]
 *
 * For each block row of U, the organization of index[] looks like:
 *
 *     [ BLOCK ROW HEADER (of size BR_HEADER)
 *           number of blocks 
 *           number of entries in nzval[]
 *           number of entries in index[]
 *       BLOCK 0                                        <----
 *           BLOCK DESCRIPTOR (of size UB_DESCRIPTOR)  |
 *               block number (global)                      |
 *               number of nonzeros in the block            |
 *           actual fstnz subscripts                        |
 *       BLOCK 1                                            | Repeat ...
 *           BLOCK DESCRIPTOR                               | number of blocks
 *               block number (global)                      |
 *               number of nonzeros in the block            |
 *           actual fstnz subscripts                        |
 *       .                                                  |
 *       .                                                  |
 *       .                                              <----
 *     ]
 *
 */
#define BC_HEADER      2
#define LB_DESCRIPTOR  2
#define BR_HEADER      3
#define UB_DESCRIPTOR  2
#define NBUFFERS       5

/*
 * Communication tags
 */
/* Return the mpi_tag assuming 5 pairs of communications and MPI_TAG_UB >= 5 *
 * for each supernodal column "num", the five communications are:            *
 * 0,1: for sending L to "right"                                             *
 * 2,3: for sending off-diagonal blocks of U "down"                          *
 * 4  : for sending the diagonal blcok down (in pxgstrf2)                    */
//#define SLU_MPI_TAG(id,num) ( (5*(num)+id) % tag_ub )

    /* For numeric factorization. */
#if 0
#define NTAGS    10000
#else
#define NTAGS    INT_MAX
#endif
#define UjROW    10
#define UkSUB    11
#define UkVAL    12
#define LkSUB    13
#define LkVAL    14
#define LkkDIAG  15
    /* For triangular solves. */
#define XK_H     2  /* The header preceding each X block. */
#define LSUM_H   2  /* The header preceding each MOD block. */
#define GSUM     20 
#define Xk       21
#define Yk       22
#define LSUM     23    

 
static const int BC_L=1;	/* MPI tag for x in L-solve*/	
static const int RD_L=2;	/* MPI tag for lsum in L-solve*/	
static const int BC_U=3;	/* MPI tag for x in U-solve*/
static const int RD_U=4;	/* MPI tag for lsum in U-solve*/	

/* 
 * Communication scopes
 */
#define COMM_ALL      100
#define COMM_COLUMN   101
#define COMM_ROW      102

/*
 * Matrix distribution for sparse matrix-vector multiplication
 */
#define SUPER_LINEAR     11
#define SUPER_BLOCK      12

/*
 * No of marker arrays used in the symbolic factorization, each of size n
 */
#define NO_MARKER     3



/***********************************************************************
 * Macros
 ***********************************************************************/
#define IAM(comm)    { int rank; MPI_Comm_rank ( comm, &rank ); rank};
#define MYROW(iam,grid) ( (iam) / grid->npcol )
#define MYCOL(iam,grid) ( (iam) % grid->npcol )
#define BlockNum(i)     ( supno[i] )
#define FstBlockC(bnum) ( xsup[bnum] )
#define SuperSize(bnum) ( xsup[bnum+1]-xsup[bnum] )
#define LBi(bnum,grid)  ( (bnum)/grid->nprow )/* Global to local block rowwise */
#define LBj(bnum,grid)  ( (bnum)/grid->npcol )/* Global to local block columnwise*/
#define PROW(bnum,grid) ( (bnum) % grid->nprow )
#define PCOL(bnum,grid) ( (bnum) % grid->npcol )
#define PNUM(i,j,grid)  ( (i)*grid->npcol + j ) /* Process number at coord(i,j) */
#define CEILING(a,b)    ( ((a)%(b)) ? ((a)/(b) + 1) : ((a)/(b)) )
    /* For triangular solves */
#define RHS_ITERATE(i)                    \
        for (i = 0; i < nrhs; ++i)
#define X_BLK(i)                          \
        ilsum[i] * nrhs + (i+1) * XK_H
#define LSUM_BLK(i)                       \
        ilsum[i] * nrhs + (i+1) * LSUM_H

#define SuperLU_timer_  SuperLU_timer_dist_
#define LOG2(x)   (log10((double) x) / log10(2.0))

#if ( VAMPIR>=1 ) 
#define VT_TRACEON    VT_traceon()
#define VT_TRACEOFF   VT_traceoff()
#else
#define VT_TRACEON 
#define VT_TRACEOFF
#endif

/* Support Windows */
#ifndef SUPERLU_DIST_EXPORT
#if MSVC
#ifdef SUPERLU_DIST_EXPORTS
#define SUPERLU_DIST_EXPORT __declspec(dllexport)
#else
#define SUPERLU_DIST_EXPORT __declspec(dllimport)
#endif /* SUPERLU_DIST_EXPORTS */
#else
#define SUPERLU_DIST_EXPORT
#endif /* MSVC */
#endif /* SUPERLU_DIST_EXPORT */


/*
 * CONSTANTS in MAGMA
 */
#ifndef MAGMA_CONST
#define MAGMA_CONST

// #define DIM_X  32
// #define DIM_Y  16

#define DIM_X  16
#define DIM_Y  16


#define BLK_M  DIM_X*4
#define BLK_N  DIM_Y*4
#define BLK_K 2048/(BLK_M)

#define DIM_XA  DIM_X
#define DIM_YA  DIM_Y
#define DIM_XB  DIM_X
#define DIM_YB  DIM_Y

#define NWARP  DIM_X*DIM_Y/32

// // // // // // #define TILE_SIZE  32


#define THR_M ( BLK_M / DIM_X )
#define THR_N ( BLK_N / DIM_Y )

#define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]
#define fma(A, B, C) C += (A*B)
#endif
/*---- end MAGMA ----*/
    
#ifdef __cplusplus
extern "C" {
#endif


#ifndef max
    #define cmax(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __cplusplus
  }
#endif


/***********************************************************************
 * New data types
 ***********************************************************************/

/* 
 *   Define the 2D mapping of matrix blocks to process grid.
 *
 *   Process grid:
 *     Processes are numbered (0 : P-1).
 *     P = Pr x Pc, where Pr, Pc are the number of process rows and columns.
 *     (pr,pc) is the coordinate of IAM; 0 <= pr < Pr, 0 <= pc < Pc.
 *
 *   Matrix blocks:
 *     Matrix is partitioned according to supernode partitions, both
 *     column and row-wise. 
 *     The k-th block columns (rows) contains columns (rows) (s:t), where
 *             s=xsup[k], t=xsup[k+1]-1.
 *     Block A(I,J) contains
 *             rows from (xsup[I]:xsup[I+1]-1) and
 *             columns from (xsup[J]:xsup[J+1]-1)
 *
 *  Mapping of matrix entry (i,j) to matrix block (I,J):
 *     (I,J) = ( supno[i], supno[j] )
 *
 *  Mapping of matrix block (I,J) to process grid (pr,pc):
 *     (pr,pc) = ( MOD(I,NPROW), MOD(J,NPCOL) )
 *  
 *  (xsup[nsupers],supno[n]) are replicated on all processors.
 *
 */

/*-- Communication subgroup */
typedef struct {
    MPI_Comm comm;        /* MPI communicator */
    int Np;               /* number of processes */
    int Iam;              /* my process number */
} superlu_scope_t;

/*-- 2D process grid definition */
typedef struct {
    MPI_Comm comm;        /* MPI communicator */
    superlu_scope_t rscp; /* process scope in rowwise, horizontal directon */
    superlu_scope_t cscp; /* process scope in columnwise, vertical direction */
    int iam;              /* my process number in this grid */
    int_t nprow;          /* number of process rows */
    int_t npcol;          /* number of process columns */
} gridinfo_t;

/*-- 3D process grid definition */
typedef struct {
    MPI_Comm comm;        /* MPI communicator */
    superlu_scope_t rscp; /* row scope */
    superlu_scope_t cscp; /* column scope */
    superlu_scope_t zscp; /* scope in third dimension */
    gridinfo_t grid2d;    /* for using 2D functions */
    int iam;              /* my process number in this grid */
    int_t nprow;          /* number of process rows */
    int_t npcol;          /* number of process columns */
    int_t npdep;          /* number of replication factor in Z-dimension  */
    int rankorder;        /* = 0: Z-major ( default )
			   *    e.g. 1x3x4 grid: layer0 layer1 layer2 layer3
			   *                     0      3      6      9
			   *                     1      4      7      10      
			   *                     2      5      8      11
			   * = 1: XY-major (need set env. var.: SUPERLU_RANKORDER=XY)
			   *    e.g. 1x3x4 grid: layer0 layer1 layer2 layer3
			   *                     0      1      2      3
			   *                     4      5      6      7
			   *                     8      9     10     11
			   */
} gridinfo3d_t;


/*
 *-- The structures are determined by SYMBFACT and used thereafter.
 *
 * (xsup,supno) describes mapping between supernode and column:
 *	xsup[s] is the leading column of the s-th supernode.
 *      supno[i] is the supernode no to which column i belongs;
 *	e.g.   supno 0 1 2 2 3 3 3 4 4 4 4 4   (n=12)
 *	        xsup 0 1 2 4 7 12
 *	Note: dfs will be performed on supernode rep. relative to the new 
 *	      row pivoting ordering
 *
 * This is allocated during symbolic factorization SYMBFACT.
 */
typedef struct {
    int_t     *xsup;
    int_t     *supno;
} Glu_persist_t;

/*
 *-- The structures are determined by SYMBFACT and used by DDISTRIBUTE.
 * 
 * (xlsub,lsub): lsub[*] contains the compressed subscript of
 *	rectangular supernodes; xlsub[j] points to the starting
 *	location of the j-th column in lsub[*]. Note that xlsub 
 *	is indexed by column.
 *	Storage: original row subscripts
 *
 *      During the course of sparse LU factorization, we also use
 *	(xlsub,lsub) for the purpose of symmetric pruning. For each
 *	supernode {s,s+1,...,t=s+r} with first column s and last
 *	column t, the subscript set
 *		lsub[j], j=xlsub[s], .., xlsub[s+1]-1
 *	is the structure of column s (i.e. structure of this supernode).
 *	It is used for the storage of numerical values.
 *	Furthermore,
 *		lsub[j], j=xlsub[t], .., xlsub[t+1]-1
 *	is the structure of the last column t of this supernode.
 *	It is for the purpose of symmetric pruning. Therefore, the
 *	structural subscripts can be rearranged without making physical
 *	interchanges among the numerical values.
 *
 *	However, if the supernode has only one column, then we
 *	only keep one set of subscripts. For any subscript interchange
 *	performed, similar interchange must be done on the numerical
 *	values.
 *
 *	The last column structures (for pruning) will be removed
 *	after the numercial LU factorization phase.
 *
 * (xusub,usub): xusub[i] points to the starting location of column i
 *      in usub[]. For each U-segment, only the row index of first nonzero
 *      is stored in usub[].
 *
 *      Each U column consists of a number of full segments. Each full segment
 *      starts from a leading nonzero, running up to the supernode (block)
 *      boundary. (Recall that the column-wise supernode partition is also
 *      imposed on the rows.) Because the segment is full, we don't store all
 *      the row indices. Instead, only the leading nonzero index is stored.
 *      The rest can be found together with xsup/supno pair.
 *      For example, 
 *          usub[xsub[j+1]] - usub[xsub[j]] = number of segments in column j.
 *          for any i in usub[], 
 *              supno[i]         = block number in which i belongs to
 *  	        xsup[supno[i]+1] = first row of the next block
 *              The nonzeros of this segment are: 
 *                  i, i+1 ... xsup[supno[i]+1]-1 (only i is stored in usub[])
 *
 */
typedef struct {
    int_t     *lsub;     /* compressed L subscripts */
    int_t     *xlsub;
    int_t     *usub;     /* compressed U subscripts */
    int_t     *xusub;
    int_t     nzlmax;    /* current max size of lsub */
    int_t     nzumax;    /*    "    "    "      usub */
    LU_space_t MemModel; /* 0 - system malloc'd; 1 - user provided */
    //int_t     *llvl;     /* keep track of level in L for level-based ILU */
    //int_t     *ulvl;     /* keep track of level in U for level-based ILU */
    int64_t nnzLU;   /* number of nonzeros in L+U*/
} Glu_freeable_t;

#if 0 // Sherry: move to precision-dependent file
/* 
 *-- The structure used to store matrix A of the linear system and
 *   several vectors describing the transformations done to matrix A.
 *
 * A      (SuperMatrix*)
 *        Matrix A in A*X=B, of dimension (A->nrow, A->ncol).
 *        The number of linear equations is A->nrow. The type of A can be:
 *        Stype = SLU_NC; Dtype = SLU_D; Mtype = SLU_GE.
 *         
 * DiagScale  (DiagScale_t)
 *        Specifies the form of equilibration that was done.
 *        = NOEQUIL: No equilibration.
 *        = ROW:  Row equilibration, i.e., A was premultiplied by diag(R).
 *        = COL:  Column equilibration, i.e., A was postmultiplied by diag(C).
 *        = BOTH: Both row and column equilibration, i.e., A was replaced 
 *                 by diag(R)*A*diag(C).
 *
 * R      double*, dimension (A->nrow)
 *        The row scale factors for A.
 *        If DiagScale = ROW or BOTH, A is multiplied on the left by diag(R).
 *        If DiagScale = NOEQUIL or COL, R is not defined.
 *
 * C      double*, dimension (A->ncol)
 *        The column scale factors for A.
 *        If DiagScale = COL or BOTH, A is multiplied on the right by diag(C).
 *        If DiagScale = NOEQUIL or ROW, C is not defined.
 *         
 * perm_r (int*) dimension (A->nrow)
 *        Row permutation vector which defines the permutation matrix Pr,
 *        perm_r[i] = j means row i of A is in position j in Pr*A.
 *
 * perm_c (int*) dimension (A->ncol)
 *	  Column permutation vector, which defines the 
 *        permutation matrix Pc; perm_c[i] = j means column i of A is 
 *        in position j in A*Pc.
 *
 */
typedef struct {
    DiagScale_t DiagScale;
    double *R;
    double *C; 
    int_t  *perm_r;
    int_t  *perm_c;
} ScalePermstruct_t;
#endif

/*-- Data structure for redistribution of B and X --*/
typedef struct {
    int  *B_to_X_SendCnt;
    int  *X_to_B_SendCnt;
    int  *ptr_to_ibuf, *ptr_to_dbuf;

    /* the following are needed in the hybrid solver PDSLin */	
    int *X_to_B_iSendCnt;
    int *X_to_B_vSendCnt;
    int    *disp_ibuf;
    int_t  *send_ibuf;
    void   *send_dbuf;

    int_t  x2b, b2x;
    int_t  *send_ibuf2;
    int_t  *recv_ibuf2;
    void   *send_dbuf2;
    void   *recv_dbuf2;
} pxgstrs_comm_t;

/* 
 *-- This contains the options used to control the solution process.
 *
 * Fact   (fact_t)
 *        Specifies whether or not the factored form of the matrix
 *        A is supplied on entry, and if not, how the matrix A should
 *        be factorizaed.
 *        = DOFACT: The matrix A will be factorized from scratch, and the
 *             factors will be stored in L and U.
 *        = SamePattern: The matrix A will be factorized assuming
 *             that a factorization of a matrix with the same sparsity
 *             pattern was performed prior to this one. Therefore, this
 *             factorization will reuse column permutation vector 
 *             ScalePermstruct->perm_c and the column elimination tree
 *             LUstruct->etree.
 *        = SamePattern_SameRowPerm: The matrix A will be factorized
 *             assuming that a factorization of a matrix with the same
 *             sparsity	pattern and similar numerical values was performed
 *             prior to this one. Therefore, this factorization will reuse
 *             both row and column scaling factors R and C, both row and
 *             column permutation vectors perm_r and perm_c, and the
 *             data structure set up from the previous symbolic factorization.
 *        = FACTORED: On entry, L, U, perm_r and perm_c contain the 
 *              factored form of A. If DiagScale is not NOEQUIL, the matrix
 *              A has been equilibrated with scaling factors R and C.
 *
 * Equil  (yes_no_t)
 *        Specifies whether to equilibrate the system (scale A's row and
 *        columns to have unit norm).
 *
 * DiagInv (yes_no_t)
 *        Specifies whether to invert the diagonal blocks of the LU
 *        triangular matrices.
 *
 * ColPerm (colperm_t)
 *        Specifies what type of column permutation to use to reduce fill.
 *        = NATURAL: use the natural ordering 
 *        = MMD_ATA: use minimum degree ordering on structure of A'*A
 *        = MMD_AT_PLUS_A: use minimum degree ordering on structure of A'+A
 *        = COLAMD: use approximate minimum degree column ordering
 *        = MY_PERMC: use the ordering specified by the user
 *         
 * Trans  (trans_t)
 *        Specifies the form of the system of equations:
 *        = NOTRANS: A * X = B        (No transpose)
 *        = TRANS:   A**T * X = B     (Transpose)
 *        = CONJ:    A**H * X = B     (Transpose)
 *
 * IterRefine (IterRefine_t)
 *        Specifies whether to perform iterative refinement.
 *        = NO: no iterative refinement
 *        = SINGLE: perform iterative refinement in single precision
 *        = DOUBLE: perform iterative refinement in double precision
 *        = EXTRA: perform iterative refinement in extra precision
 *
 * DiagPivotThresh (double, in [0.0, 1.0]) (only for serial SuperLU)
 *        Specifies the threshold used for a diagonal entry to be an
 *        acceptable pivot.
 *
 * SymmetricMode (yest_no_t) (only for serial SuperLU)
 *        Specifies whether to use symmetric mode. Symmetric mode gives 
 *        preference to diagonal pivots, and uses an (A'+A)-based column
 *        permutation algorithm.
 *
 * PivotGrowth (yes_no_t)  (only for serial SuperLU)
 *        Specifies whether to compute the reciprocal pivot growth.
 *
 * ConditionNumber (ues_no_t) (only for serial SuperLU)
 *        Specifies whether to compute the reciprocal condition number.
 *
 * RowPerm (rowperm_t) (only for SuperLU_DIST or ILU in serial SuperLU)
 *        Specifies whether to permute rows of the original matrix.
 *        = NO: not to permute the rows
 *        = LargeDiag: make the diagonal large relative to the off-diagonal
 *        = MY_PERMR: use the permutation given by the user
 *
 * ILU_DropRule (int)  (only for serial SuperLU)
 *        Specifies the dropping rule:
 *	  = DROP_BASIC:   Basic dropping rule, supernodal based ILUTP(tau).
 *	  = DROP_PROWS:   Supernodal based ILUTP(p,tau), p = gamma * nnz(A)/n.
 *	  = DROP_COLUMN:  Variant of ILUTP(p,tau), for j-th column,
 *			      p = gamma * nnz(A(:,j)).
 *	  = DROP_AREA:    Variation of ILUTP, for j-th column, use
 *			      nnz(F(:,1:j)) / nnz(A(:,1:j)) to control memory.
 *	  = DROP_DYNAMIC: Modify the threshold tau during factorizaion:
 *			  If nnz(L(:,1:j)) / nnz(A(:,1:j)) > gamma
 *				  tau_L(j) := MIN(tau_0, tau_L(j-1) * 2);
 *			  Otherwise
 *				  tau_L(j) := MAX(tau_0, tau_L(j-1) / 2);
 *			  tau_U(j) uses the similar rule.
 *			  NOTE: the thresholds used by L and U are separate.
 *	  = DROP_INTERP:  Compute the second dropping threshold by
 *	                  interpolation instead of sorting (default).
 *  		          In this case, the actual fill ratio is not
 *			  guaranteed to be smaller than gamma.
 *   	  Note: DROP_PROWS, DROP_COLUMN and DROP_AREA are mutually exclusive.
 *	  ( Default: DROP_BASIC | DROP_AREA )
 *
 * ILU_DropTol (double) (only for serial SuperLU)
 *        numerical threshold for dropping.
 *
 * ILU_FillFactor (double) (only for serial SuperLU)
 *        Gamma in the secondary dropping.
 *
 * ILU_Norm (norm_t)  (only for serial SuperLU)
 *        Specify which norm to use to measure the row size in a
 *        supernode: infinity-norm, 1-norm, or 2-norm.
 *
 * ILU_FillTol (double) (only for serial SuperLU)
 *        numerical threshold for zero pivot perturbation.
 *
 * ILU_MILU (milu_t)  (only for serial SuperLU)
 *        Specifies which version of MILU to use.
 *
 * ILU_MILU_Dim (double) 
 *        Dimension of the PDE if available.
 *
 * ReplaceTinyPivot (yes_no_t) (only for SuperLU_DIST)
 *        Specifies whether to replace the tiny diagonals by
 *        sqrt(epsilon)*||A|| during LU factorization.
 *
 * SolveInitialized (yes_no_t) (only for SuperLU_DIST)
 *        Specifies whether the initialization has been performed to the
 *        triangular solve.
 *
 * RefineInitialized (yes_no_t) (only for SuperLU_DIST)
 *        Specifies whether the initialization has been performed to the
 *        sparse matrix-vector multiplication routine needed in iterative
 *        refinement.
 *
 * num_lookaheads (int) (only for SuperLU_DIST)
 *        Specifies the number of levels in the look-ahead factorization
 *
 * lookahead_etree (yes_no_t) (only for SuperLU_DIST)
 *        Specifies whether to use the elimination tree computed from the 
 *        serial symbolic factorization to perform scheduling.
 *
 * SymPattern (yes_no_t) (only for SuperLU_DIST)
 *        Gives the scheduling algorithm a hint whether the matrix
 *        would have symmetric pattern.
 *
 */
typedef struct {
    fact_t        Fact;
    yes_no_t      Equil;
    yes_no_t      DiagInv;
    colperm_t     ColPerm;
    trans_t       Trans;
    IterRefine_t  IterRefine;
    double        DiagPivotThresh;
    yes_no_t      SymmetricMode;
    yes_no_t      PivotGrowth;
    yes_no_t      ConditionNumber;
    rowperm_t     RowPerm;
    int 	  ILU_DropRule;
    double	  ILU_DropTol;    /* threshold for dropping */
    double	  ILU_FillFactor; /* gamma in the secondary dropping */
    norm_t	  ILU_Norm;       /* infinity-norm, 1-norm, or 2-norm */
    double	  ILU_FillTol;    /* threshold for zero pivot perturbation */
    milu_t	  ILU_MILU;
    double	  ILU_MILU_Dim;   /* Dimension of PDE (if available) */
    yes_no_t      ParSymbFact;
    yes_no_t      ReplaceTinyPivot; /* used in SuperLU_DIST */
    yes_no_t      SolveInitialized;
    yes_no_t      RefineInitialized;
    yes_no_t      PrintStat;
    //int           nnzL, nnzU;      /* used to store nnzs for now       */
    yes_no_t      lookahead_etree; /* use etree computed from the
				      serial symbolic factorization */
    int num_lookaheads;  /* num of levels in look-ahead      */
    int superlu_relax;   /* max. allowed relaxed supernode size; see sp_ienv(2) */
    int superlu_maxsup;  /* max. allowed supernode size; see sp_ienv(3) */
    char superlu_rankorder[4]; /* Z-major or XY-majir order in 3D grid */
    char superlu_lbs[4]; /* etree load balancing strategy in 3D algorithm */
    int superlu_n_gemm; /* one of GEMM offload criteria; see sp_ienv(7) */
    int superlu_max_buffer_size; /* max. buffer size on GPU; see sp_ienv(8) */
    int superlu_num_gpu_streams; /* number of GPU streams; see sp_ienv(9) */
    int superlu_acc_offload; /* whether to offload work to GPU; see sp_ienv(10) */
    yes_no_t      SymPattern;      /* symmetric factorization          */
    yes_no_t      Use_TensorCore;  /* Use Tensor Core or not  */
    yes_no_t      Algo3d;          /* use 3D factorization/solve algorithms */
} superlu_dist_options_t;

typedef struct {
    float for_lu;
    float total;
    int expansions;
    int64_t nnzL, nnzU;
} superlu_dist_mem_usage_t;

/*-- Auxiliary data type used in PxGSTRS/PxGSTRS1. */
typedef struct {
    int_t lbnum;  /* Row block number (local).      */
    int_t indpos; /* Starting position in Uindex[]. */
} Ucb_indptr_t;

/* 
 *-- The new structures added in the hybrid GPU + OpenMP + MPI code.
 */
typedef struct {
    int_t rukp;
    int_t iukp;
    int_t jb;
    int_t full_u_cols;
    int_t eo; /* order of elimination. For 3D algorithm */
    int_t ncols;
    int_t StCol;
} Ublock_info_t;

typedef struct {
    int_t lptr;
    int_t ib;
    int_t eo; /* order of elimination, for 3D code */
    int_t nrows;
    int_t FullRow;
    int_t StRow;
} Remain_info_t;

typedef struct
{
    int id, key;
    void *next;
} etree_node;

struct superlu_pair
{
    int ind;
    int val;
};

/**--------**/

/*==== For 3D code ====*/

typedef struct
{
  int_t nub;
  int_t klst;
  int_t ldu;
  int_t* usub;
  //double *uval;
} uPanelInfo_t;

typedef struct
{
  int_t *lsub;
  //double *lusup;
  void *lusup;
  int_t luptr0;
  int_t nlb;  //number of l blocks                                                      
  int_t nsupr;
} lPanelInfo_t;

typedef struct
{
    Ublock_info_t* Ublock_info;
    Remain_info_t*  Remain_info;
    uPanelInfo_t* uPanelInfo;
    lPanelInfo_t* lPanelInfo;
} packLUInfo_t;


/* HyP_t is the data structure to assist HALO offload of Schur-complement. */
typedef struct
{
  Remain_info_t *lookAhead_info, *Remain_info;
  Ublock_info_t *Ublock_info, *Ublock_info_Phi;

  int_t first_l_block_acc , first_u_block_acc;
  int_t last_offload ;
  int_t *Lblock_dirty_bit, * Ublock_dirty_bit;
  void *lookAhead_L_buff, *Remain_L_buff;
  int_t lookAheadBlk;  /* number of blocks in look-ahead window */
  int_t RemainBlk ;    /* number of blocks outside look-ahead window */
  int_t  num_look_aheads, nsupers;
  int_t ldu, ldu_Phi;
  int_t num_u_blks, num_u_blks_Phi;

  int_t jj_cpu;
  void *bigU_Phi;
  void *bigU_host;
  int_t Lnbrow;
  int_t Rnbrow;

  int_t buffer_size;
  int_t bigu_size;
  int offloadCondition;
  int superlu_acc_offload;
  int nGPUStreams;
} HyP_t;



/* return the mpi_tag assuming 5 pairs of communications and MPI_TAG_UB >= 5 *
 * for each supernodal column, the five communications are:                  *
 * 0,1: for sending L to "right"                                             *
 * 2,3: for sending off-diagonal blocks of U "down"                          *
 * 4  : for sending the diagonal blcok down (in pxgstrf2)                    */
// int tag_ub;
// #define SLU_MPI_TAG(id,num) ( (5*(num)+id) % tag_ub )

// #undef SLU_MPI_TAG
/*defining my own MPI tags */
/* return the mpi_tag assuming 5 pairs of communications and MPI_TAG_UB >= 5 *
 * for each supernodal column, the five communications are:                  *
 * 0,1: for sending L to "right"                                             *
 * 2,3: for sending off-diagonal blocks of U "down"                          *
 * 4  : for sending the diagonal blcok down (in pxgstrf2)                    *
 * 5  : for sending the diagonal L block right () : added by piyush */
#define SLU_MPI_TAG(id,num) ( (6*(num)+id) % tag_ub )

/*structs for quick look up */
typedef struct
{
    int_t luptrj;
    int_t lptrj;
    int_t lib;
} local_l_blk_info_t;
 
typedef struct
{
    int_t iuip;
    int_t ruip;
    int_t ljb;
} local_u_blk_info_t;


//global variable 
// extern double CPU_CLOCK_RATE;

typedef struct
{
    int_t *perm_c_supno;
    int_t *iperm_c_supno;
}  perm_array_t;

typedef struct
{   
    int_t* factored;
    int_t* factored_D;
    int_t* factored_L;
    int_t* factored_U;
    int_t* IrecvPlcd_D;
    int_t* IbcastPanel_L;         /*I bcast and recv placed for the k-th L panel*/
    int_t* IbcastPanel_U;         /*I bcast and recv placed for the k-th U panel*/
    int_t* numChildLeft;            /*number of children left to be factored*/
    int_t* gpuLUreduced;          /*New for GPU acceleration*/
}factStat_t;

typedef struct
{
    int_t next_col;
    int_t next_k;
    int_t kljb;
    int_t kijb;
    int_t copyL_kljb;
    int_t copyU_kljb;
    int_t l_copy_len;
    int_t u_copy_len;
    int_t *kindexL;
    int_t *kindexU;
    int_t mkrow;
    int_t mkcol;
    int_t ksup_size;
} d2Hreduce_t;

typedef struct{
	int_t numChild;
	int_t numDescendents;
	int_t left;
	int_t right;
	int_t extra;
	int_t* childrenList;
	int_t depth; 		// distance from the top
	double weight; 		// weight of the supernode
	double iWeight; 	// weight of the whole subtree below
	double scuWeight; 	// weight of schur complement update = max|n_k||L_k||U_k|
} treeList_t;

typedef struct 
{
	int_t numLvl;  // number of level in tree;
	int_t* eTreeTopLims; // boundaries of each level  of size 
	int_t* myIperm;		// Iperm for my tree size nsupers;
	
} treeTopoInfo_t;

typedef struct 
{
	int_t* setree; 		// global supernodal elimination tree
	int_t* numChildLeft;
} gEtreeInfo_t; 

typedef enum treePartStrat{
	ND,				// nested dissection ordering or natural ordering
	GD 				// greedy load balance stregy
}treePartStrat;

typedef struct
{
	/* data */
	int_t nNodes; 			// total number of nodes
	int_t* nodeList;		// list of nodes, should be in order of factorization
#if 0 // Sherry: the following array is used on rForest_t. ???
	int_t* treeHeads;
#endif
                            /*topological information about the tree*/
	int_t numLvl;  			// number of Topological levels in the forest
	int_t numTrees; 		// number of tree in the forest
	treeTopoInfo_t  topoInfo; //
#if 0  // Sherry fix: the following two structures are in treeTopoInfo_t. ???
	int_t* eTreeTopLims; 	// boundaries of each level  of size 
	int_t* myIperm;			// Iperm for my tree size nsupers;
#endif

	/*information about load balance*/
	double weight;		// estimated cost 
	double cost; 		// measured cost

} sForest_t;

typedef struct
{
    /* data */
    MPI_Request* L_diag_blk_recv_req;
    MPI_Request* L_diag_blk_send_req;
    MPI_Request* U_diag_blk_recv_req;
    MPI_Request* U_diag_blk_send_req;
    MPI_Request* recv_req;
    MPI_Request* recv_requ;
    MPI_Request* send_req;
    MPI_Request* send_requ;
} commRequests_t;

typedef struct
{
    int_t *iperm_c_supno;
    int_t *iperm_u;
    int_t *perm_u;
    int *indirect;
    int *indirect2;
} factNodelists_t;

typedef struct
{
    int* msgcnt;
    int* msgcntU;
} msgs_t;

typedef struct xtrsTimer_t
{
    double trsDataSendXY;
    double trsDataSendZ;
    double trsDataRecvXY;
    double trsDataRecvZ;
    double t_pdReDistribute_X_to_B;
    double t_pdReDistribute_B_to_X;
    double t_forwardSolve;
    double tfs_compute;
    double tfs_comm;
    double t_backwardSolve;
    double tbs_compute;
    double tbs_comm;
    double tbs_tree[2*MAX_3D_LEVEL];
    double tfs_tree[2*MAX_3D_LEVEL];
    
    // counters for communication and computation volume 
    
    int_t trsMsgSentXY;
    int_t trsMsgSentZ;
    int_t trsMsgRecvXY;
    int_t trsMsgRecvZ;
    
    double ppXmem;		// perprocess X-memory
} xtrsTimer_t;

/*==== end For 3D code ====*/

/*====================*/

/***********************************************************************
 * Function prototypes
 ***********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

extern void   superlu_gridinit(MPI_Comm, int, int, gridinfo_t *);
extern void   superlu_gridmap(MPI_Comm, int, int, int [], int, gridinfo_t *);
extern void   superlu_gridexit(gridinfo_t *);
extern void   superlu_gridinit3d(MPI_Comm Bcomm,  int nprow, int npcol, int npdep,
				 gridinfo3d_t *grid) ;
extern void   superlu_gridexit3d(gridinfo3d_t *grid);

extern void   set_default_options_dist(superlu_dist_options_t *);
extern void   print_options_dist(superlu_dist_options_t *);
extern void   print_sp_ienv_dist(superlu_dist_options_t *);
extern void   Destroy_CompCol_Matrix_dist(SuperMatrix *);
extern void   Destroy_SuperNode_Matrix_dist(SuperMatrix *);
extern void   Destroy_SuperMatrix_Store_dist(SuperMatrix *);
extern void   Destroy_CompCol_Permuted_dist(SuperMatrix *);
extern void   Destroy_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void   Destroy_CompRow_Matrix_dist(SuperMatrix *);
extern void   sp_colorder (superlu_dist_options_t*, SuperMatrix*, int_t*, int_t*,
			   SuperMatrix*);
extern int    sp_symetree_dist(int_t *, int_t *, int_t *, int_t, int_t *);
extern int    sp_coletree_dist (int_t *, int_t *, int_t *, int_t, int_t, int_t *);
extern void   get_perm_c_dist(int_t, int_t, SuperMatrix *, int_t *);
extern void   at_plus_a_dist(const int_t, const int_t, int_t *, int_t *,
			     int_t *, int_t **, int_t **);
extern int    genmmd_dist_(int_t *, int_t *, int_t *a, 
			   int_t *, int_t *, int_t *, int_t *, 
			   int_t *, int_t *, int_t *, int_t *, int_t *);
extern void  bcast_tree(void *, int, MPI_Datatype, int, int,
			gridinfo_t *, int, int *);
extern int_t symbfact(superlu_dist_options_t *, int, SuperMatrix *, int_t *,
                      int_t *, Glu_persist_t *, Glu_freeable_t *);
extern int_t symbfact_SubInit(superlu_dist_options_t *options,
			      fact_t, void *, int_t, int_t, int_t, int_t,
			      Glu_persist_t *, Glu_freeable_t *);
extern int_t symbfact_SubXpand(int_t, int_t, int_t, MemType, int_t *,
			       Glu_freeable_t *);
extern int_t symbfact_SubFree(Glu_freeable_t *);
extern void    countnz_dist (const int_t, int_t *, int_t *, int_t *,
			     Glu_persist_t *, Glu_freeable_t *);
extern int64_t fixupL_dist (const int_t, const int_t *, Glu_persist_t *,
				  Glu_freeable_t *);
extern int_t   *TreePostorder_dist (int_t, int_t *);
extern float   smach_dist(char *);
extern double  dmach_dist(char *);
extern void    *superlu_malloc_dist (size_t);
extern void    superlu_free_dist (void*);
extern int   *int32Malloc_dist (int);
extern int   *int32Calloc_dist (int);
extern int_t   *intMalloc_dist (int_t);
extern int_t   *intCalloc_dist (int_t);
extern int   mc64id_dist(int *);
extern void  arrive_at_ublock (int_t, int_t *, int_t *, int_t *,
			       int_t *, int_t *, int_t, int_t, 
			       int_t *, int_t *, int_t *, gridinfo_t *);
extern int_t estimate_bigu_size (int_t, int_t **, Glu_persist_t *,
				 gridinfo_t *, int_t *, int_t*);

/* Auxiliary routines */
extern double SuperLU_timer_ ();
extern void   superlu_abort_and_exit_dist(char *);
extern int    sp_ienv_dist (int, superlu_dist_options_t *);
extern void   ifill_dist (int_t *, int_t, int_t);
extern void   super_stats_dist (int_t, int_t *);
extern void  get_diag_procs(int_t, Glu_persist_t *, gridinfo_t *, int_t *,
			    int_t **, int_t **);
extern int_t QuerySpace_dist(int_t, int_t, Glu_freeable_t *, superlu_dist_mem_usage_t *);
extern int   xerr_dist (char *, int *);
extern void  pxerr_dist (char *, gridinfo_t *, int_t);
extern void  PStatInit(SuperLUStat_t *);
extern void  PStatFree(SuperLUStat_t *);
extern void  PStatPrint(superlu_dist_options_t *, SuperLUStat_t *, gridinfo_t *);
extern void  log_memory(int64_t, SuperLUStat_t *);
extern void  print_memorylog(SuperLUStat_t *, char *);
extern int   superlu_dist_GetVersionNumber(int *, int *, int *);
extern void  quickSort( int_t*, int_t, int_t, int_t);
extern void  quickSortM( int_t*, int_t, int_t, int_t, int_t, int_t);
extern int_t partition( int_t*, int_t, int_t, int_t);
extern int_t partitionM( int_t*, int_t, int_t, int_t, int_t, int_t);

/* Prototypes for parallel symbolic factorization */
extern float symbfact_dist
(superlu_dist_options_t *, int,  int,
 SuperMatrix *, int_t *, int_t *,  int_t *, int_t *,
 Pslu_freeable_t *, MPI_Comm *, MPI_Comm *,  superlu_dist_mem_usage_t *);

/* Get the column permutation using parmetis */
extern float get_perm_c_parmetis 
(SuperMatrix *, int_t *, int_t *, int, int, 
 int_t **, int_t **, gridinfo_t *, MPI_Comm *);

/* Auxiliary routines for memory expansions used during
   the parallel symbolic factorization routine */

extern int_t psymbfact_LUXpandMem
(int, int_t, int_t, int_t, int_t, int, int, int, 
 Pslu_freeable_t *, Llu_symbfact_t *,  vtcsInfo_symbfact_t *, psymbfact_stat_t *);

extern int_t psymbfact_LUXpand
(int_t, int_t, int_t, int_t, int_t *, int_t, int_t, int_t, int_t, 
 Pslu_freeable_t *, Llu_symbfact_t *,  vtcsInfo_symbfact_t *, psymbfact_stat_t *);

extern int_t psymbfact_LUXpand_RL
(int_t, int_t, int_t, int_t, int_t, int_t,
 Pslu_freeable_t *, Llu_symbfact_t *, vtcsInfo_symbfact_t *, psymbfact_stat_t *);

extern int_t psymbfact_prLUXpand
(int_t,  int_t, int, Llu_symbfact_t *, psymbfact_stat_t *);

#ifdef ISORT
extern void isort (int_t N, int_t *ARRAY1, int_t *ARRAY2);
extern void isort1 (int_t N, int_t *ARRAY);
#else
int superlu_sort_perm (const void *arg1, const void *arg2)
{
    const int_t *val1 = (const int_t *) arg1;
    const int_t *val2 = (const int_t *) arg2;
    return (*val2 < *val1);
}
#endif

#ifdef GPU_ACC   /* GPU related */
extern void gemm_division_cpu_gpu (superlu_dist_options_t *,
				   int *, int *, int *, int,
				   int, int, int *, int, int_t);
extern int_t get_gpublas_nb ();
extern int_t get_num_gpu_streams ();
extern int getnGPUStreams();
extern int get_mpi_process_per_gpu ();
/*to print out various statistics from GPU activities*/
extern void printGPUStats(int nsupers, SuperLUStat_t *stat, gridinfo3d_t*);
#endif

extern double estimate_cpu_time(int m, int n , int k);

extern int get_thread_per_process();
extern int_t get_max_buffer_size ();
extern int_t get_min (int_t *, int_t);
extern int compare_pair (const void *, const void *);
extern int_t static_partition (struct superlu_pair *, int_t, int_t *, int_t,
			       int_t *, int_t *, int);
extern int get_acc_offload();
    

/* Routines for debugging */
extern void  print_panel_seg_dist(int_t, int_t, int_t, int_t, int_t *, int_t *);
extern void  check_repfnz_dist(int_t, int_t, int_t, int_t *);
extern int_t CheckZeroDiagonal(int_t, int_t *, int_t *, int_t *);
extern int   check_perm_dist(char *what, int_t n, int_t *perm);
extern void  PrintDouble5(char *, int_t, double *);
extern void  PrintInt10(char *, int_t, int_t *);
extern void  PrintInt32(char *, int, int *);
extern int   file_PrintInt10(FILE *, char *, int_t, int_t *);
extern int   file_PrintInt32(FILE *, char *, int, int *);
extern int   file_PrintLong10(FILE *, char *, int_t, int_t *);

/* Routines for Async_tree communication*/

#ifndef __SUPERLU_ASYNC_TREE /* allow multiple inclusions */
#define __SUPERLU_ASYNC_TREE
typedef struct
{
    MPI_Request sendRequests_[2];
    MPI_Comm comm_;
    int myRoot_;
    int destCnt_;
    int myDests_[2];
    int myRank_;
    int msgSize_;
    int tag_;
    yes_no_t empty_;
    MPI_Datatype type_;
} C_Tree;

#ifndef DEG_TREE
#define DEG_TREE 2
#endif

#endif

extern void C_RdTree_Create(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision);
extern void C_RdTree_Nullify(C_Tree* tree);
extern yes_no_t C_RdTree_IsRoot(C_Tree* tree);
extern void C_RdTree_forwardMessageSimple(C_Tree* Tree, void* localBuffer, int msgSize);
extern void C_RdTree_waitSendRequest(C_Tree* Tree);

extern void C_BcTree_Create(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision);
extern void C_BcTree_Nullify(C_Tree* tree);
extern yes_no_t C_BcTree_IsRoot(C_Tree* tree);
extern void C_BcTree_forwardMessageSimple(C_Tree* tree, void* localBuffer, int msgSize);
extern void C_BcTree_waitSendRequest(C_Tree* tree);

/*==== For 3D code ====*/

extern void DistPrint(char* function_name,  double value, char* Units, gridinfo_t* grid);
extern void DistPrint3D(char* function_name,  double value, char* Units, gridinfo3d_t* grid3d);
extern void treeImbalance3D(gridinfo3d_t *grid3d, SCT_t* SCT);
extern void SCT_printComm3D(gridinfo3d_t *grid3d, SCT_t* SCT);

// permutation from superLU default
extern int_t* getPerm_c_supno(int_t nsupers, superlu_dist_options_t *,
			      int_t *etree, Glu_persist_t *Glu_persist, 
			      int_t** Lrowind_bc_ptr, int_t** Ufstnz_br_ptr,
			      gridinfo_t *);

/* Manipulate counters */
extern void SCT_init(SCT_t*);
extern void SCT_print(gridinfo_t *grid, SCT_t* SCT);
extern void SCT_print3D(gridinfo3d_t *grid3d, SCT_t* SCT);
extern void SCT_free(SCT_t*);

extern treeList_t* setree2list(int_t nsuper, int_t* setree );
extern int  free_treelist(int_t nsuper, treeList_t* treeList);

// int_t calcTreeWeight(int_t nsupers, treeList_t* treeList, int_t* xsup);
extern int_t calcTreeWeight(int_t nsupers, int_t*setree, treeList_t* treeList, int_t* xsup);
extern int_t getDescendList(int_t k, int_t*dlist,  treeList_t* treeList);
extern int_t getCommonAncestorList(int_t k, int_t* alist,  int_t* seTree, treeList_t* treeList);
extern int_t getCommonAncsCount(int_t k, treeList_t* treeList);
extern int_t* getPermNodeList(int_t nnode, 	// number of nodes
			      int_t* nlist, int_t* perm_c_sup,int_t* iperm_c_sup);
extern int_t* getEtreeLB(int_t nnodes, int_t* perm_l, int_t* gTopOrder);
extern int_t* getSubTreeRoots(int_t k, treeList_t* treeList);
// int_t* treeList2perm(treeList_t* , ..);
extern int_t* merg_perms(int_t nperms, int_t* nnodes, int_t** perms);
// returns a concatenated permutation for three permutation arrays

extern int_t* getGlobal_iperm(int_t nsupers, int_t nperms, int_t** perms,
			      int_t* nnodes);
extern int_t log2i(int_t index);
extern int_t *supernodal_etree(int_t nsuper, int_t * etree, int_t* supno, int_t *xsup);
extern int_t testSubtreeNodelist(int_t nsupers, int_t numList, int_t** nodeList, int_t* nodeCount);
extern int_t testListPerm(int_t nodeCount, int_t* nodeList, int_t* permList, int_t* gTopLevel);

/*takes supernodal elimination tree and for each 
  supernode calculates "level" in elimination tree*/
extern int_t* topological_ordering(int_t nsuper, int_t* setree);
extern int_t* Etree_LevelBoundry(int_t* perm,int_t* tsort_etree, int_t nsuper);

/*calculated boundries of the topological levels*/
extern int_t* calculate_num_children(int_t nsuper, int_t* setree);
extern void Print_EtreeLevelBoundry(int_t *Etree_LvlBdry, int_t max_level, int_t nsuper);
extern void print_etree_leveled(int_t *setree,  int_t* tsort_etree, int_t nsuper);
extern void print_etree(int_t *setree, int_t* iperm, int_t nsuper);
extern int_t printFileList(char* sname, int_t nnodes, int_t*dlist, int_t*setree);
int* getLastDepBtree( int_t nsupers, treeList_t* treeList);

/*returns array R with of size maxLevel with either 0 or 1
	R[i] = 1; then Tree[level-i] is set to zero= to only 
	accumulate the results   */
extern int_t* getReplicatedTrees( gridinfo3d_t* grid3d);

/*returns indices in gNodeList of trees that belongs to my layer*/
extern int_t* getGridTrees( gridinfo3d_t* grid3d);


/*returns global nodelist*/
extern int_t** getNodeList(int_t maxLvl, int_t* setree, int_t* nnodes,
			int_t* treeHeads, treeList_t* treeList);

/* calculate number of nodes in subtrees starting from treeHead[i]*/
extern int_t* calcNumNodes(int_t maxLvl,  int_t* treeHeads, treeList_t* treeList);

/*Returns list of (last) node of the trees */
extern int_t* getTreeHeads(int_t maxLvl, int_t nsupers, treeList_t* treeList);

extern int_t* getMyIperm(int_t nnodes, int_t nsupers, int_t* myPerm);

extern int_t* getMyTopOrder(int_t nnodes, int_t* myPerm, int_t* myIperm, int_t* setree );

extern int_t* getMyEtLims(int_t nnodes, int_t* myTopOrder);


extern treeTopoInfo_t getMyTreeTopoInfo(int_t nnodes, int_t  nsupers,
                                 int_t* myPerm,int_t* setree);

extern sForest_t**  getNestDissForests( int_t maxLvl, int_t nsupers, int_t*setree, treeList_t* treeList);

extern int_t** getTreePermForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t*  sForests,
				  int_t* perm_c_supno, int_t* iperm_c_supno,
				  gridinfo3d_t* grid3d);
extern int_t** getTreePermFr( int_t* myTreeIdxs,
			      sForest_t**  sForests, gridinfo3d_t* grid3d);
extern int_t* getMyNodeCountsFr(int_t maxLvl, int_t* myTreeIdxs,
				sForest_t**  sForests);
extern int_t** getNodeListFr(int_t maxLvl, sForest_t**  sForests);
extern int_t*  getNodeCountsFr(int_t maxLvl, sForest_t**  sForests);
// int_t* getNodeToForstMap(int_t nsupers, sForest_t**  sForests, gridinfo3d_t* grid3d);
extern int* getIsNodeInMyGrid(int_t nsupers, int_t maxLvl, int_t* myNodeCount, int_t** treePerm);
extern void printForestWeightCost(sForest_t**  sForests, SCT_t* SCT, gridinfo3d_t* grid3d);
extern sForest_t**  getGreedyLoadBalForests( int_t maxLvl, int_t nsupers, int_t* setree, treeList_t* treeList);
extern sForest_t**  getForests( int_t maxLvl, int_t nsupers, int_t*setree, treeList_t* treeList);

    /* from trfAux.h */
extern int_t getBigUSize(superlu_dist_options_t *, int_t nsupers,
			 gridinfo_t *grid, int_t **Lrowind_bc_ptr);
extern void getSCUweight(int_t nsupers, treeList_t* treeList, int_t* xsup,
			 int_t** Lrowind_bc_ptr, int_t** Ufstnz_br_ptr,
			 gridinfo3d_t * grid3d);
extern int Wait_LUDiagSend(int_t k, MPI_Request *U_diag_blk_send_req,
			   MPI_Request *L_diag_blk_send_req,
			   gridinfo_t *grid, SCT_t *SCT);

extern int getNsupers(int n, Glu_persist_t *Glu_persist);
extern int set_tag_ub();
extern int getNumThreads(int);
extern int_t num_full_cols_U(int_t kk, int_t **Ufstnz_br_ptr, int_t *xsup,
			     gridinfo_t *, int_t *, int_t *);

#if 0 // Sherry: conflicting with existing routine
extern int_t estimate_bigu_size(int_t nsupers, int_t ldt, int_t**Ufstnz_br_ptr,
				Glu_persist_t *, gridinfo_t*, int_t* perm_u);
#endif

extern int_t* getFactPerm(int_t);
extern int_t* getFactIperm(int_t*, int_t);

extern int_t initCommRequests(commRequests_t* comReqs, gridinfo_t * grid);
extern int_t initFactStat(int_t nsupers, factStat_t* factStat);
extern int   freeFactStat(factStat_t* factStat);
extern int_t initFactNodelists(int_t, int_t, int_t, factNodelists_t*);
extern int   freeFactNodelists(factNodelists_t* fNlists);
extern int_t initMsgs(msgs_t* msgs);
extern int_t getNumLookAhead(superlu_dist_options_t*);
extern commRequests_t** initCommRequestsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid);
extern int   freeCommRequestsArr(int_t mxLeafNode, commRequests_t** comReqss);

extern msgs_t** initMsgsArr(int_t numLA);
extern int      freeMsgsArr(int_t numLA, msgs_t **msgss);

extern int_t Trs2_InitUblock_info(int_t klst, int_t nb, Ublock_info_t *,
                                  int_t *usub, Glu_persist_t *, SuperLUStat_t*);

    /* from sec_structs.h */
extern int Cmpfunc_R_info (const void * a, const void * b);
extern int Cmpfunc_U_info (const void * a, const void * b);
extern int sort_R_info( Remain_info_t* Remain_info, int n );
extern int sort_U_info( Ublock_info_t* Ublock_info, int n );
extern int sort_R_info_elm( Remain_info_t* Remain_info, int n );
extern int sort_U_info_elm( Ublock_info_t* Ublock_info, int n );

    /* from pdgstrs.h */
extern void printTRStimer(xtrsTimer_t *xtrsTimer, gridinfo3d_t *grid3d);
extern void initTRStimer(xtrsTimer_t *xtrsTimer, gridinfo_t *grid);

    /* from p3dcomm.c */
extern int_t** getTreePerm( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                     int_t* nodeCount, int_t** nodeList,
                     int_t* perm_c_supno, int_t* iperm_c_supno,
                     gridinfo3d_t* grid3d);
extern int_t* getMyNodeCounts(int_t maxLvl, int_t* myTreeIdxs, int_t* gNodeCount);
extern int_t checkIntVector3d(int_t* vec, int_t len,  gridinfo3d_t* grid3d);
extern int_t reduceStat(PhaseType PHASE, SuperLUStat_t *stat, gridinfo3d_t * grid3d);

    /* from communication_aux.h */
extern int_t Wait_LSend(int_t k, gridinfo_t *grid, int **ToSendR,
			MPI_Request *s, SCT_t*);
extern int_t Wait_USend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t Check_LRecv(MPI_Request*, int* msgcnt);
extern int_t Wait_UDiagBlockSend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t Wait_LDiagBlockSend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t Wait_UDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t Test_UDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t Wait_LDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t Test_LDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t LDiagBlockRecvWait( int_t k, int_t* factored_U, MPI_Request *, gridinfo_t *);

/*=====================*/

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_DEFS */
