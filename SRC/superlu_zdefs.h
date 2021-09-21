/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief  Distributed SuperLU data types and function prototypes
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology
 * November 1, 2007
 * April 5, 2015
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * May 10, 2019 version 7.0.0
 * </pre>
 */

#ifndef __SUPERLU_ZDEFS /* allow multiple inclusions */
#define __SUPERLU_ZDEFS

/*
 * File name:	superlu_zdefs.h
 * Purpose:     Distributed SuperLU data types and function prototypes
 * History:
 */

#include "superlu_defs.h"
#include "dcomplex.h"

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
} zScalePermstruct_t;

#if 0 // Sherry: move to superlu_defs.h 
/*-- Auxiliary data type used in PxGSTRS/PxGSTRS1. */
typedef struct {
    int_t lbnum;  /* Row block number (local).      */
    int_t indpos; /* Starting position in Uindex[]. */
} Ucb_indptr_t;
#endif

/*
 * On each processor, the blocks in L are stored in compressed block
 * column format, the blocks in U are stored in compressed block row format.
 */
#define MAX_LOOKAHEADS 50
typedef struct {
    int_t   **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
    doublecomplex **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    doublecomplex **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr */
    int_t   *Unnz; /* number of nonzeros per block column in U*/
	int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */
    doublecomplex  **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    doublecomplex  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *LRtree_ptr;       /* size ceil(NSUPERS/Pr)                */
    C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *URtree_ptr;       /* size ceil(NSUPERS/Pr)			*/
#if 0
    int_t   *Lsub_buf;        /* Buffer for the remote subscripts of L */
    doublecomplex  *Lval_buf;        /* Buffer for the remote nonzeros of L   */
    int_t   *Usub_buf;        /* Buffer for the remote subscripts of U */
    doublecomplex  *Uval_buf;        /* Buffer for the remote nonzeros of U   */
#endif
    int_t   *Lsub_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote subscripts of L*/
    doublecomplex  *Lval_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote nonzeros of L  */
    int_t   *Usub_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote subscripts of U */
    doublecomplex  *Uval_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote nonzeros of U   */
    doublecomplex  *ujrow;           /* used in panel factorization.          */
    int_t   bufmax[NBUFFERS]; /* Maximum buffer size across all MPI ranks:
			       *  0 : maximum size of Lsub_buf[]
			       *  1 : maximum size of Lval_buf[]
			       *  2 : maximum size of Usub_buf[]
			       *  3 : maximum size of Uval_buf[]
			       *  4 : maximum size of tempv[LDA]
			       */

    /*-- Record communication schedule for factorization. --*/
    int   *ToRecv;          /* Recv from no one (0), left (1), and up (2).*/
    int   *ToSendD;         /* Whether need to send down block row.       */
    int   **ToSendR;        /* List of processes to send right block col. */

    /*-- Record communication schedule for forward/back solves. --*/
    int_t   *fmod;            /* Modification count for L-solve            */
    int_t   **fsendx_plist;   /* Column process list to send down Xk       */
    int_t   *frecv;           /* Modifications to be recv'd in proc row    */
    int_t   nfrecvx;          /* Number of Xk I will receive in L-solve    */
    int_t   nfsendx;          /* Number of Xk I will send in L-solve       */
    int_t   *bmod;            /* Modification count for U-solve            */
    int_t   **bsendx_plist;   /* Column process list to send down Xk       */
    int_t   *brecv;           /* Modifications to be recv'd in proc row    */
    int_t   nbrecvx;          /* Number of Xk I will receive in U-solve    */
    int_t   nbsendx;          /* Number of Xk I will send in U-solve       */
    int_t   *mod_bit;         /* Flag contribution from each row blocks    */

    /*-- Auxiliary arrays used for forward/back solves. --*/
    int_t   *ilsum;           /* Starting position of each supernode in lsum
				 (local)  */
    int_t   ldalsum;          /* LDA of lsum (local) */
    int_t   SolveMsgSent;     /* Number of actual messages sent in LU-solve */
    int_t   SolveMsgVol;      /* Volume of messages sent in the solve phase */


    /*********************/
    /* The following variables are used in the hybrid solver */

    /*-- Counts to be used in U^{-T} triangular solve. -- */
    int_t UT_SOLVE;
    int_t L_SOLVE;
    int_t FRECV;
    int_t ut_ldalsum;        /* LDA of lsum (local) */
    int_t *ut_ilsum;         /* ilsum in column-wise                        */
    int_t *utmod;            /* Modification count for Ut-solve.            */
    int_t **ut_sendx_plist;  /* Row process list to send down Xk            */
    int_t *utrecv;           /* Modifications to be recev'd in proc column. */
    int_t n_utsendx;         /* Number of Xk I will receive                 */
    int_t n_utrecvx;         /* Number of Xk I will send                    */
    int_t n_utrecvmod;
    int_t nroot;
    int_t *ut_modbit;
    int_t *Urbs;
    Ucb_indptr_t **Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */

    /* some additional counters for L solve */
    int_t n;
    int_t nleaf;
    int_t nfrecvmod;
    int_t inv; /* whether the diagonal block is inverted*/
} zLocalLU_t;


typedef struct {
    int_t *etree;
    Glu_persist_t *Glu_persist;
    zLocalLU_t *Llu;
    char dt;
} zLUstruct_t;


/*-- Data structure for communication during matrix-vector multiplication. */
typedef struct {
    int_t *extern_start;
    int_t *ind_tosend;    /* X indeices to be sent to other processes */
    int_t *ind_torecv;    /* X indeices to be received from other processes */
    int_t *ptr_ind_tosend;/* Printers to ind_tosend[] (Size procs)
			     (also point to val_torecv) */
    int_t *ptr_ind_torecv;/* Printers to ind_torecv[] (Size procs)
			     (also point to val_tosend) */
    int   *SendCounts;    /* Numbers of X indices to be sent
			     (also numbers of X values to be received) */
    int   *RecvCounts;    /* Numbers of X indices to be received
			     (also numbers of X values to be sent) */
    doublecomplex *val_tosend;   /* X values to be sent to other processes */
    doublecomplex *val_torecv;   /* X values to be received from other processes */
    int_t TotalIndSend;   /* Total number of indices to be sent
			     (also total number of values to be received) */
    int_t TotalValSend;   /* Total number of values to be sent.
			     (also total number of indices to be received) */
} pzgsmv_comm_t;

/*-- Data structure holding the information for the solution phase --*/
typedef struct {
    int_t *row_to_proc;
    int_t *inv_perm_c;
    int_t num_diag_procs, *diag_procs, *diag_len;
    pzgsmv_comm_t *gsmv_comm; /* communication metadata for SpMV,
         	       		      required by IterRefine.          */
    pxgstrs_comm_t *gstrs_comm;  /* communication metadata for SpTRSV. */
    int_t *A_colind_gsmv; /* After pzgsmv_init(), the global column
                             indices of A are translated into the relative
                             positions in the gathered x-vector.
                             This is re-used in repeated calls to pzgsmv() */
    int_t *xrow_to_proc; /* used by PDSLin */
} zSOLVEstruct_t;

#if 0 

/*==== For 3D code ====*/

// new structures for pdgstrf_4_8 

typedef struct
{
    int_t nub;
    int_t klst;
    int_t ldu;
    int_t* usub;
    doublecomplex* uval;
} uPanelInfo_t;

typedef struct
{
    int_t *lsub;
    doublecomplex *lusup;
    int_t luptr0;
    int_t nlb;  //number of l blocks
    int_t nsupr;
} lPanelInfo_t;

 

/* HyP_t is the data structure to assist HALO offload of Schur-complement. */
typedef struct
{
    Remain_info_t *lookAhead_info, *Remain_info;
    Ublock_info_t *Ublock_info, *Ublock_info_Phi;
    
    int_t first_l_block_acc , first_u_block_acc;
    int_t last_offload ;
    int_t *Lblock_dirty_bit, * Ublock_dirty_bit;
    doublecomplex *lookAhead_L_buff, *Remain_L_buff;
    int_t lookAheadBlk;  /* number of blocks in look-ahead window */
    int_t RemainBlk ;    /* number of blocks outside look-ahead window */
    int_t  num_look_aheads, nsupers;
    int_t ldu, ldu_Phi;
    int_t num_u_blks, num_u_blks_Phi;

    int_t jj_cpu;
    doublecomplex *bigU_Phi;
    doublecomplex *bigU_host;
    int_t Lnbrow;
    int_t Rnbrow;

    int_t buffer_size;
    int_t bigu_size;
    int_t offloadCondition;
    int_t superlu_acc_offload;
    int_t nCudaStreams;
} HyP_t;

typedef struct 
{
    int_t * Lsub_buf ;
    doublecomplex * Lval_buf ;
    int_t * Usub_buf ;
    doublecomplex * Uval_buf ;
} zLUValSubBuf_t;

int_t scuStatUpdate(
    int_t knsupc,
    HyP_t* HyP, 
    SCT_t* SCT,
    SuperLUStat_t *stat
    );

typedef struct
{
    gEtreeInfo_t gEtreeInfo;
    int_t* iperm_c_supno;
    int_t* myNodeCount;
    int_t* myTreeIdxs;
    int_t* myZeroTrIdxs;
    int_t** treePerm;
    sForest_t** sForests;
    int_t* supernode2treeMap;
    zLUValSubBuf_t  *LUvsb;
} trf3Dpartition_t;

typedef struct
{
    doublecomplex *bigU;
    doublecomplex *bigV;
} scuBufs_t;

typedef struct
{   
    doublecomplex* BlockLFactor;
    doublecomplex* BlockUFactor;
} diagFactBufs_t;

typedef struct
{
    Ublock_info_t* Ublock_info;
    Remain_info_t*  Remain_info;
    uPanelInfo_t* uPanelInfo;
    lPanelInfo_t* lPanelInfo;
} packLUInfo_t;

#endif
/*=====================*/

/***********************************************************************
 * Function prototypes
 ***********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


/* Supernodal LU factor related */
extern void
zCreate_CompCol_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, doublecomplex *,
			    int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_CompRowLoc_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, int_t,
			       int_t, doublecomplex *, int_t *, int_t *,
			       Stype_t, Dtype_t, Mtype_t);
extern void
zCompRow_to_CompCol_dist(int_t, int_t, int_t, doublecomplex *, int_t *, int_t *,
                         doublecomplex **, int_t **, int_t **);
extern int
pzCompRow_loc_to_CompCol_global(int_t, SuperMatrix *, gridinfo_t *,
	 		        SuperMatrix *);
extern void
zCopy_CompCol_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void
zCreate_Dense_Matrix_dist(SuperMatrix *, int_t, int_t, doublecomplex *, int_t,
			  Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_SuperNode_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, doublecomplex *,
			      int_t *, int_t *, int_t *, int_t *, int_t *,
			      Stype_t, Dtype_t, Mtype_t);
extern void
zCopy_Dense_Matrix_dist(int_t, int_t, doublecomplex *, int_t,
                        doublecomplex *, int_t);

extern void    zallocateA_dist (int_t, int_t, doublecomplex **, int_t **, int_t **);
extern void    zGenXtrue_dist (int_t, int_t, doublecomplex *, int_t);
extern void    zFillRHS_dist (char *, int_t, doublecomplex *, int_t,
                              SuperMatrix *, doublecomplex *, int_t);
extern int     zcreate_matrix(SuperMatrix *, int, doublecomplex **, int *,
			      doublecomplex **, int *, FILE *, gridinfo_t *);
extern int     zcreate_matrix_rb(SuperMatrix *, int, doublecomplex **, int *,
			      doublecomplex **, int *, FILE *, gridinfo_t *);
extern int     zcreate_matrix_dat(SuperMatrix *, int, doublecomplex **, int *,
			      doublecomplex **, int *, FILE *, gridinfo_t *);
extern int 	   zcreate_matrix_postfix(SuperMatrix *, int, doublecomplex **, int *,
				  doublecomplex **, int *, FILE *, char *, gridinfo_t *);

extern void   zScalePermstructInit(const int_t, const int_t, 
                                      zScalePermstruct_t *);
extern void   zScalePermstructFree(zScalePermstruct_t *);

/* Driver related */
extern void    zgsequ_dist (SuperMatrix *, double *, double *, double *,
			    double *, double *, int_t *);
extern double  zlangs_dist (char *, SuperMatrix *);
extern void    zlaqgs_dist (SuperMatrix *, double *, double *, double,
			    double, double, char *);
extern void    pzgsequ (SuperMatrix *, double *, double *, double *,
			double *, double *, int_t *, gridinfo_t *);
extern double  pzlangs (char *, SuperMatrix *, gridinfo_t *);
extern void    pzlaqgs (SuperMatrix *, double *, double *, double,
			double, double, char *);
extern int     pzPermute_Dense_Matrix(int_t, int_t, int_t [], int_t[],
				      doublecomplex [], int, doublecomplex [], int, int,
				      gridinfo_t *);

extern int     sp_ztrsv_dist (char *, char *, char *, SuperMatrix *,
			      SuperMatrix *, doublecomplex *, int *);
extern int     sp_zgemv_dist (char *, doublecomplex, SuperMatrix *, doublecomplex *,
			      int, doublecomplex, doublecomplex *, int);
extern int     sp_zgemm_dist (char *, int, doublecomplex, SuperMatrix *,
                        doublecomplex *, int, doublecomplex, doublecomplex *, int);

extern float zdistribute(fact_t, int_t, SuperMatrix *, Glu_freeable_t *,
			 zLUstruct_t *, gridinfo_t *);
extern void  pzgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *,
			      zScalePermstruct_t *, doublecomplex *,
			      int, int, gridinfo_t *, zLUstruct_t *, double *,
			      SuperLUStat_t *, int *);
extern float pzdistribute(fact_t, int_t, SuperMatrix *,
			 zScalePermstruct_t *, Glu_freeable_t *,
			 zLUstruct_t *, gridinfo_t *);
extern void  pzgssvx(superlu_dist_options_t *, SuperMatrix *,
		     zScalePermstruct_t *, doublecomplex *,
		     int, int, gridinfo_t *, zLUstruct_t *,
		     zSOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void  pzCompute_Diag_Inv(int_t, zLUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  zSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, zLUstruct_t *, gridinfo_t *, zSOLVEstruct_t *);
extern void zSolveFinalize(superlu_dist_options_t *, zSOLVEstruct_t *);
extern int_t pzgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, zSOLVEstruct_t *);
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  zldperm_dist(int_t, int_t, int_t, int_t [], int_t [],
		    doublecomplex [], int_t *, double [], double []);
extern int  zstatic_schedule(superlu_dist_options_t *, int, int,
		            zLUstruct_t *, gridinfo_t *, SuperLUStat_t *,
			    int_t *, int_t *, int *);
extern void zLUstructInit(const int_t, zLUstruct_t *);
extern void zLUstructFree(zLUstruct_t *);
extern void zDestroy_LU(int_t, gridinfo_t *, zLUstruct_t *);
extern void zDestroy_Tree(int_t, gridinfo_t *, zLUstruct_t *);
extern void zscatter_l (int ib, int ljb, int nsupc, int_t iukp, int_t* xsup,
			int klst, int nbrow, int_t lptr, int temp_nbrow,
			int_t* usub, int_t* lsub, doublecomplex *tempv,
			int* indirect_thread, int* indirect2,
			int_t ** Lrowind_bc_ptr, doublecomplex **Lnzval_bc_ptr,
			gridinfo_t * grid);
extern void zscatter_u (int ib, int jb, int nsupc, int_t iukp, int_t * xsup,
                        int klst, int nbrow, int_t lptr, int temp_nbrow,
                        int_t* lsub, int_t* usub, doublecomplex* tempv,
                        int_t ** Ufstnz_br_ptr, doublecomplex **Unzval_br_ptr,
                        gridinfo_t * grid);
extern int_t pzgstrf(superlu_dist_options_t *, int, int, double,
		    zLUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);

/* #define GPU_PROF
#define IPM_PROF */

/* Solve related */
extern void pzgstrs_Bglobal(int_t, zLUstruct_t *, gridinfo_t *,
			     doublecomplex *, int_t, int, SuperLUStat_t *, int *);
extern void pzgstrs(int_t, zLUstruct_t *, zScalePermstruct_t *, gridinfo_t *,
		    doublecomplex *, int_t, int_t, int_t, int, zSOLVEstruct_t *,
		    SuperLUStat_t *, int *);
extern void pzgstrf2_trsm(superlu_dist_options_t * options, int_t k0, int_t k,
			  double thresh, Glu_persist_t *, gridinfo_t *,
			  zLocalLU_t *, MPI_Request *, int tag_ub,
			  SuperLUStat_t *, int *info);
extern void pzgstrs2_omp(int_t k0, int_t k, Glu_persist_t *, gridinfo_t *,
			 zLocalLU_t *, Ublock_info_t *, SuperLUStat_t *);
extern int_t pzReDistribute_B_to_X(doublecomplex *B, int_t m_loc, int nrhs, int_t ldb,
				   int_t fst_row, int_t *ilsum, doublecomplex *x,
				   zScalePermstruct_t *, Glu_persist_t *,
				   gridinfo_t *, zSOLVEstruct_t *);
extern void zlsum_fmod(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int, int_t , int_t *, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);
extern void zlsum_bmod(doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void zlsum_fmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int_t , int_t *,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_fmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int, int_t , int_t *, int_t,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_bmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void zlsum_bmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int, int);

extern void pzgsrfs(int_t, SuperMatrix *, double, zLUstruct_t *,
		    zScalePermstruct_t *, gridinfo_t *,
		    doublecomplex [], int_t, doublecomplex [], int_t, int,
		    zSOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void pzgsrfs_ABXglobal(int_t, SuperMatrix *, double, zLUstruct_t *,
		  gridinfo_t *, doublecomplex *, int_t, doublecomplex *, int_t,
		  int, double *, SuperLUStat_t *, int *);
extern int   pzgsmv_AXglobal_setup(SuperMatrix *, Glu_persist_t *,
				   gridinfo_t *, int_t *, int_t *[],
				   doublecomplex *[], int_t *[], int_t []);
extern int  pzgsmv_AXglobal(int_t, int_t [], doublecomplex [], int_t [],
	                       doublecomplex [], doublecomplex []);
extern int  pzgsmv_AXglobal_abs(int_t, int_t [], doublecomplex [], int_t [],
				 doublecomplex [], double []);
extern void pzgsmv_init(SuperMatrix *, int_t *, gridinfo_t *,
			pzgsmv_comm_t *);
extern void pzgsmv(int_t, SuperMatrix *, gridinfo_t *, pzgsmv_comm_t *,
		   doublecomplex x[], doublecomplex ax[]);
extern void pzgsmv_finalize(pzgsmv_comm_t *);

/* Memory-related */
extern doublecomplex  *doublecomplexMalloc_dist(int_t);
extern doublecomplex  *doublecomplexCalloc_dist(int_t);
extern double  *doubleMalloc_dist(int_t);
extern double  *doubleCalloc_dist(int_t);
extern void  *duser_malloc_dist (int_t, int_t);
extern void  duser_free_dist (int_t, int_t);
extern int_t zQuerySpace_dist(int_t, zLUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void zClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void zScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, doublecomplex);
extern void zScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, doublecomplex);
extern void zZeroLblocks(int, int_t, gridinfo_t *, zLUstruct_t *);
extern void    zfill_dist (doublecomplex *, int_t, doublecomplex);
extern void    zinf_norm_error_dist (int_t, int_t, doublecomplex*, int_t,
                                     doublecomplex*, int_t, gridinfo_t*);
extern void    pzinf_norm_error(int, int_t, int_t, doublecomplex [], int_t,
				doublecomplex [], int_t , gridinfo_t *);
extern void  zreadhb_dist (int, FILE *, int_t *, int_t *, int_t *,
			   doublecomplex **, int_t **, int_t **);
extern void  zreadtriple_dist(FILE *, int_t *, int_t *, int_t *,
			 doublecomplex **, int_t **, int_t **);
extern void  zreadtriple_noheader(FILE *, int_t *, int_t *, int_t *,
			 doublecomplex **, int_t **, int_t **);
extern void  zreadrb_dist(int, FILE *, int_t *, int_t *, int_t *,
		     doublecomplex **, int_t **, int_t **);
extern void  zreadMM_dist(FILE *, int_t *, int_t *, int_t *,
	                  doublecomplex **, int_t **, int_t **);
extern int  zread_binary(FILE *, int_t *, int_t *, int_t *,
	                  doublecomplex **, int_t **, int_t **);

/* Distribute the data for numerical factorization */
extern float zdist_psymbtonum(fact_t, int_t, SuperMatrix *,
                                zScalePermstruct_t *, Pslu_freeable_t *,
                                zLUstruct_t *, gridinfo_t *);
extern void pzGetDiagU(int_t, zLUstruct_t *, gridinfo_t *, doublecomplex *);

extern int  z_c2cpp_GetHWPM(SuperMatrix *, gridinfo_t *, zScalePermstruct_t *);

/* Routines for debugging */
extern void  zPrintLblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
		 	   zLocalLU_t *);
extern void  zPrintUblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
			   zLocalLU_t *);
extern void  zPrint_CompCol_Matrix_dist(SuperMatrix *);
extern void  zPrint_Dense_Matrix_dist(SuperMatrix *);
extern int   zPrint_CompRowLoc_Matrix_dist(SuperMatrix *);
extern int   file_zPrint_CompRowLoc_Matrix_dist(FILE *fp, SuperMatrix *A);
extern void  PrintDoublecomplex(char *, int_t, doublecomplex *);
extern int   file_PrintDoublecomplex(FILE *fp, char *, int_t, doublecomplex *);


/* BLAS */

#ifdef USE_VENDOR_BLAS
extern void zgemm_(const char*, const char*, const int*, const int*, const int*,
                  const doublecomplex*, const doublecomplex*, const int*, const doublecomplex*,
                  const int*, const doublecomplex*, doublecomplex*, const int*, int, int);
extern void ztrsv_(char*, char*, char*, int*, doublecomplex*, int*,
                  doublecomplex*, int*, int, int, int);
extern void ztrsm_(char*, char*, char*, char*, int*, int*,
                  doublecomplex*, doublecomplex*, int*, doublecomplex*,
                  int*, int, int, int, int);
extern void zgemv_(char *, int *, int *, doublecomplex *, doublecomplex *a, int *,
                  doublecomplex *, int *, doublecomplex *, doublecomplex *, int *, int);

extern void zgeru_(int*, int*, doublecomplex*, doublecomplex*, int*,
                 doublecomplex*, int*, doublecomplex*, int*);

#else
extern int zgemm_(const char*, const char*, const int*, const int*, const int*,
                   const doublecomplex*,  const doublecomplex*,  const int*,  const doublecomplex*,
                   const int*,  const doublecomplex*, doublecomplex*, const int*);
extern int ztrsv_(char*, char*, char*, int*, doublecomplex*, int*,
                  doublecomplex*, int*);
extern int ztrsm_(char*, char*, char*, char*, int*, int*,
                  doublecomplex*, doublecomplex*, int*, doublecomplex*, int*);
extern int zgemv_(char *, int *, int *, doublecomplex *, doublecomplex *a, int *,
                  doublecomplex *, int *, doublecomplex *, doublecomplex *, int *);
extern int zgeru_(int*, int*, doublecomplex*, doublecomplex*, int*,
                 doublecomplex*, int*, doublecomplex*, int*);

#endif

extern int zscal_(int *n, doublecomplex *da, doublecomplex *dx, int *incx);
extern int zaxpy_(int *n, doublecomplex *za, doublecomplex *zx, 
	               int *incx, doublecomplex *zy, int *incy);
// LAPACK routine
extern void ztrtri_(char*, char*, int*, doublecomplex*, int*, int*);


#if 0
/*==== For 3D code ====*/

extern void pzgssvx3d (superlu_dist_options_t *, SuperMatrix *,
		       zScalePermstruct_t *, doublecomplex B[], int ldb, int nrhs,
		       gridinfo3d_t *, zLUstruct_t *, zSOLVEstruct_t *, 
		       double *berr, SuperLUStat_t *, int *info);
extern int_t pzgstrf3d(superlu_dist_options_t *, int m, int n, double anorm,
		       trf3Dpartition_t*, SCT_t *, zLUstruct_t *,
		       gridinfo3d_t *, SuperLUStat_t *, int *);
extern void zInit_HyP(HyP_t* HyP, zLocalLU_t *Llu, int_t mcb, int_t mrb );
extern void Free_HyP(HyP_t* HyP);
extern int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid);

    /* from scatter.h */
extern void
zblock_gemm_scatter( int_t lb, int_t j, Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info, doublecomplex *L_mat, int_t ldl,
                    doublecomplex *U_mat, int_t ldu,  doublecomplex *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int_t *indirect, int_t *indirect2,
                    int_t **Lrowind_bc_ptr, doublecomplex **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, doublecomplex **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *, SuperLUStat_t *
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );
/*this version uses a lock to prevent multiple thread updating the same block*/
extern void
zblock_gemm_scatter_lock( int_t lb, int_t j, omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,  Remain_info_t *Remain_info,
                         doublecomplex *L_mat, int_t ldl, doublecomplex *U_mat, int_t ldu,
                         doublecomplex *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int_t *indirect, int_t *indirect2,
                         int_t **Lrowind_bc_ptr, doublecomplex **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, doublecomplex **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );
extern int_t
zblock_gemm_scatterTopLeft( int_t lb,  int_t j, doublecomplex* bigV,
				 int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,
				 int_t* indirect, int_t* indirect2,
                                 HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                 SCT_t*SCT, SuperLUStat_t *
                               );
extern int_t 
zblock_gemm_scatterTopRight( int_t lb,  int_t j, doublecomplex* bigV,
				  int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,
				  int_t* indirect, int_t* indirect2,
                                  HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                  SCT_t*SCT, SuperLUStat_t * );
extern int_t
zblock_gemm_scatterBottomLeft( int_t lb,  int_t j, doublecomplex* bigV,
				    int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt, 
				    int_t* indirect, int_t* indirect2,
                                    HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                    SCT_t*SCT, SuperLUStat_t * );
extern int_t 
zblock_gemm_scatterBottomRight( int_t lb,  int_t j, doublecomplex* bigV,
				     int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,
				     int_t* indirect, int_t* indirect2,
                                     HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                     SCT_t*SCT, SuperLUStat_t * );

    /* from gather.h */
extern void zgather_u(int_t num_u_blks,
              Ublock_info_t *Ublock_info, int_t * usub,
              doublecomplex *uval,  doublecomplex *bigU,  int_t ldu,
              int_t *xsup, int_t klst                /* for SuperSize */
             );

extern void zgather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               doublecomplex * lval, int_t LD_lval,
               doublecomplex * L_buff );

extern void zRgather_L(int_t k, int_t *lsub, doublecomplex *lusup, gEtreeInfo_t*,
		      Glu_persist_t *, gridinfo_t *, HyP_t *,
		      int_t *myIperm, int_t *iperm_c_supno );
extern void zRgather_U(int_t k, int_t jj0, int_t *usub, doublecomplex *uval,
		      doublecomplex *bigU, gEtreeInfo_t*, Glu_persist_t *,
		      gridinfo_t *, HyP_t *, int_t *myIperm,
		      int_t *iperm_c_supno, int_t *perm_u);

    /* from xtrf3Dpartition.h */
extern trf3Dpartition_t* zinitTrf3Dpartition(int_t nsupers,
					     superlu_dist_options_t *options,
					     zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern void zDestroy_trf3Dpartition(trf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d);

extern void z3D_printMemUse(trf3Dpartition_t*  trf3Dpartition,
			    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);

extern int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
		       superlu_dist_options_t *options, zLocalLU_t *Llu,
		       int_t* xsup, int_t num_look_aheads, int_t nsupers,
		       int_t * iperm_c_supno);

extern void zinit3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t**  sForests, zLUstruct_t* LUstruct,
				  gridinfo3d_t* grid3d);

extern int_t zgatherAllFactoredLUFr(int_t* myZeroTrIdxs, sForest_t* sForests,
				   zLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
				   SCT_t* SCT );

    /* The following are from pdgstrf2.h */
#if 0 // Sherry: same routine names, but different code !!!!!!!
extern void pzgstrf2_trsm(superlu_dist_options_t *options, int_t, int_t,
                          int_t k, double thresh, Glu_persist_t *,
			  gridinfo_t *, zLocalLU_t *, MPI_Request *U_diag_blk_send_req,
			  int tag_ub, SuperLUStat_t *, int *info, SCT_t *);
#ifdef _CRAY
void pzgstrs2_omp (int_t, int_t, int_t, Glu_persist_t *, gridinfo_t *,
                      zLocalLU_t *, SuperLUStat_t *, _fcd, _fcd, _fcd);
#else
void pzgstrs2_omp (int_t, int_t, int_t, int_t *, doublecomplex*, Glu_persist_t *, gridinfo_t *,
                      zLocalLU_t *, SuperLUStat_t *, Ublock_info_t *, doublecomplex *bigV, int_t ldt, SCT_t *SCT );
#endif

#else 
extern void pzgstrf2_trsm(superlu_dist_options_t * options, int_t k0, int_t k,
			  double thresh, Glu_persist_t *, gridinfo_t *,
			  zLocalLU_t *, MPI_Request *, int tag_ub,
			  SuperLUStat_t *, int *info);
extern void pzgstrs2_omp(int_t k0, int_t k, Glu_persist_t *, gridinfo_t *,
			 zLocalLU_t *, Ublock_info_t *, SuperLUStat_t *);
#endif // same routine names   !!!!!!!!

extern int_t zLpanelUpdate(int_t off0, int_t nsupc, doublecomplex* ublk_ptr,
			  int_t ld_ujrow, doublecomplex* lusup, int_t nsupr, SCT_t*);
extern void Local_Zgstrf2(superlu_dist_options_t *options, int_t k,
			  double thresh, doublecomplex *BlockUFactor, Glu_persist_t *,
			  gridinfo_t *, zLocalLU_t *,
                          SuperLUStat_t *, int *info, SCT_t*);
extern int_t zTrs2_GatherU(int_t iukp, int_t rukp, int_t klst,
			  int_t nsupc, int_t ldu, int_t *usub,
			  doublecomplex* uval, doublecomplex *tempv);
extern int_t zTrs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
			   int_t nsupc, int_t ldu, int_t *usub,
			   doublecomplex* uval, doublecomplex *tempv);
extern int_t zTrs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
                             int_t *usub, doublecomplex* uval, doublecomplex *tempv,
                             int_t knsupc, int_t nsupr, doublecomplex* lusup,
                             Glu_persist_t *Glu_persist)  ;
extern void pzgstrs2
#ifdef _CRAY
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    zLocalLU_t *Llu, SuperLUStat_t *stat, _fcd ftcs1, _fcd ftcs2, _fcd ftcs3
);
#else
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    zLocalLU_t *Llu, SuperLUStat_t *stat
);
#endif

extern void pzgstrf2(superlu_dist_options_t *, int_t nsupers, int_t k0,
		     int_t k, double thresh, Glu_persist_t *, gridinfo_t *,
		     zLocalLU_t *, MPI_Request *, int, SuperLUStat_t *, int *);

    /* from p3dcomm.h */
extern int_t zAllocLlu_3d(int_t nsupers, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zp3dScatter(int_t n, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zscatter3dLPanels(int_t nsupers,
                       zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zscatter3dUPanels(int_t nsupers,
                       zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zcollect3dLpanels(int_t layer, int_t nsupers, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zcollect3dUpanels(int_t layer, int_t nsupers, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zp3dCollect(int_t layer, int_t n, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
/*zero out LU non zero entries*/
extern int_t zzeroSetLU(int_t nnodes, int_t* nodeList , zLUstruct_t *, gridinfo3d_t*);
extern int AllocGlu_3d(int_t n, int_t nsupers, zLUstruct_t *);
extern int DeAllocLlu_3d(int_t n, zLUstruct_t *, gridinfo3d_t*);
extern int DeAllocGlu_3d(zLUstruct_t *);

/* Reduces L and U panels of nodes in the List nodeList (size=nnnodes)
receiver[L(nodelist)] =sender[L(nodelist)] +receiver[L(nodelist)]
receiver[U(nodelist)] =sender[U(nodelist)] +receiver[U(nodelist)]
*/
int_t zreduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        doublecomplex* Lval_buf, doublecomplex* Uval_buf,
                        zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
/*reduces all nodelists required in a level*/
int_t zreduceAllAncestors3d(int_t ilvl, int_t* myNodeCount,
                           int_t** treePerm,
                           zLUValSubBuf_t* LUvsb,
                           zLUstruct_t* LUstruct,
                           gridinfo3d_t* grid3d,
                           SCT_t* SCT );
/*
	Copies factored L and U panels from sender grid to receiver grid
	receiver[L(nodelist)] <-- sender[L(nodelist)];
	receiver[U(nodelist)] <-- sender[U(nodelist)];
*/
int_t zgatherFactoredLU(int_t sender, int_t receiver,
                       int_t nnodes, int_t *nodeList, zLUValSubBuf_t*  LUvsb,
                       zLUstruct_t* LUstruct, gridinfo3d_t* grid3d,SCT_t* SCT );

/*Gathers all the L and U factors to grid 0 for solve stage 
	By  repeatidly calling above function*/
int_t zgatherAllFactoredLU(trf3Dpartition_t*  trf3Dpartition, zLUstruct_t* LUstruct,
			   gridinfo3d_t* grid3d, SCT_t* SCT );

/*Distributes data in each layer and initilizes ancestors
 as zero in required nodes*/
int_t zinit3DLUstruct( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                      int_t* nodeCount, int_t** nodeList,
                      zLUstruct_t* LUstruct, gridinfo3d_t* grid3d);

int_t zzSendLPanel(int_t k, int_t receiver,
		   zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t zzRecvLPanel(int_t k, int_t sender, doublecomplex alpha, 
                   doublecomplex beta, doublecomplex* Lval_buf,
		   zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t zzSendUPanel(int_t k, int_t receiver,
		   zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t zzRecvUPanel(int_t k, int_t sender, doublecomplex alpha,
                   doublecomplex beta, doublecomplex* Uval_buf,
		   zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);

    /* from communication_aux.h */
extern int_t zIBcast_LPanel (int_t k, int_t k0, int_t* lsub, doublecomplex* lusup,
			     gridinfo_t *, int* msgcnt, MPI_Request *,
			     int_t **ToSendR, int_t *xsup, int );
extern int_t zBcast_LPanel(int_t k, int_t k0, int_t* lsub, doublecomplex* lusup,
			   gridinfo_t *, int* msgcnt, int_t **ToSendR,
			   int_t *xsup , SCT_t*, int);
extern int_t zIBcast_UPanel(int_t k, int_t k0, int_t* usub, doublecomplex* uval,
			    gridinfo_t *, int* msgcnt, MPI_Request *,
			    int_t *ToSendD, int );
extern int_t zBcast_UPanel(int_t k, int_t k0, int_t* usub, doublecomplex* uval,
			   gridinfo_t *, int* msgcnt, int_t *ToSendD, SCT_t*, int);
extern int_t zIrecv_LPanel (int_t k, int_t k0,  int_t* Lsub_buf, 
			    doublecomplex* Lval_buf, gridinfo_t *,
			    MPI_Request *, zLocalLU_t *, int);
extern int_t zIrecv_UPanel(int_t k, int_t k0, int_t* Usub_buf, doublecomplex*,
			   zLocalLU_t *, gridinfo_t*, MPI_Request *, int);
extern int_t Wait_LSend(int_t k, gridinfo_t *grid, int_t **ToSendR,
			MPI_Request *s, SCT_t*);
extern int_t Wait_USend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t zWait_URecv(MPI_Request *, int* msgcnt, SCT_t *);
extern int_t Check_LRecv(MPI_Request*, int* msgcnt);
extern int_t zWait_LRecv(MPI_Request*, int* msgcnt, int* msgcntsU,
			 gridinfo_t *, SCT_t*);
extern int_t zISend_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t zRecv_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			      int_t src, gridinfo_t *, SCT_t*, int);
extern int_t Wait_UDiagBlockSend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t Wait_LDiagBlockSend(MPI_Request *, gridinfo_t *, SCT_t *);
extern int_t zPackLBlock(int_t k, doublecomplex* Dest, Glu_persist_t *,
			 gridinfo_t *, zLocalLU_t *);
extern int_t zISend_LDiagBlock(int_t k0, doublecomplex *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t zIRecv_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t *,
			       SCT_t*, int);
extern int_t Wait_UDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t Test_UDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t zIRecv_LDiagBlock(int_t k0, doublecomplex *L_blk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t*, SCT_t*, int);
extern int_t Wait_LDiagBlock_Recv(MPI_Request *, SCT_t *);
extern int_t Test_LDiagBlock_Recv(MPI_Request *, SCT_t *);

extern int_t zUDiagBlockRecvWait( int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
				  MPI_Request *, gridinfo_t *, zLUstruct_t *, SCT_t *);
extern int_t LDiagBlockRecvWait( int_t k, int_t* factored_U, MPI_Request *, gridinfo_t *);
#if (MPI_VERSION>2)
extern int_t zIBcast_UDiagBlock(int_t k, doublecomplex *ublk_ptr, int_t size,
				MPI_Request *, gridinfo_t *);
extern int_t zIBcast_LDiagBlock(int_t k, doublecomplex *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *);
#endif

    /* from trfCommWrapper.h */
extern int_t zDiagFactIBCast(int_t k,  int_t k0,
			     doublecomplex *BlockUFactor, doublecomplex *BlockLFactor,
			     int_t* IrecvPlcd_D, MPI_Request *, MPI_Request *,
			     MPI_Request *, MPI_Request *, gridinfo_t *,
			     superlu_dist_options_t *, double thresh,
			     zLUstruct_t *LUstruct, SuperLUStat_t *, int *info,
			     SCT_t *, int tag_ub);
extern int_t zUPanelTrSolve( int_t k, doublecomplex* BlockLFactor, doublecomplex* bigV,
			     int_t ldt, Ublock_info_t*, gridinfo_t *,
			     zLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t Wait_LUDiagSend(int_t k, MPI_Request *, MPI_Request *,
			     gridinfo_t *, SCT_t *);
extern int_t zLPanelUpdate(int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
			   MPI_Request *, doublecomplex* BlockUFactor, gridinfo_t *,
			   zLUstruct_t *, SCT_t *);
extern int_t zUPanelUpdate(int_t k, int_t* factored_U, MPI_Request *,
			   doublecomplex* BlockLFactor, doublecomplex* bigV,
			   int_t ldt, Ublock_info_t*, gridinfo_t *,
			   zLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t zIBcastRecvLPanel(int_t k, int_t k0, int* msgcnt,
			       MPI_Request *, MPI_Request *,
			       int_t* Lsub_buf, doublecomplex* Lval_buf,
			      int_t * factored, gridinfo_t *, zLUstruct_t *,
			      SCT_t *, int tag_ub);
extern int_t zIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt, MPI_Request *,
			       MPI_Request *, int_t* Usub_buf, doublecomplex* Uval_buf,
			       gridinfo_t *, zLUstruct_t *, SCT_t *, int tag_ub);
extern int_t zWaitL(int_t k, int* msgcnt, int* msgcntU, MPI_Request *,
		    MPI_Request *, gridinfo_t *, zLUstruct_t *, SCT_t *);
extern int_t zWaitU(int_t k, int* msgcnt, MPI_Request *, MPI_Request *,
		   gridinfo_t *, zLUstruct_t *, SCT_t *);
extern int_t zLPanelTrSolve(int_t k, int_t* factored_L, doublecomplex* BlockUFactor,
			    gridinfo_t *, zLUstruct_t *);

    /* from trfAux.h */
extern int_t getNsupers(int, zLUstruct_t *);
extern int_t initPackLUInfo(int_t nsupers, packLUInfo_t* packLUInfo);
extern int   freePackLUInfo(packLUInfo_t* packLUInfo);
extern int_t zSchurComplementSetup(int_t k, int *msgcnt, Ublock_info_t*,
				   Remain_info_t*, uPanelInfo_t *,
				   lPanelInfo_t *, int_t*, int_t *, int_t *,
				   doublecomplex *bigU, int_t* Lsub_buf,
				   doublecomplex* Lval_buf, int_t* Usub_buf,
				   doublecomplex* Uval_buf, gridinfo_t *, zLUstruct_t *);
extern int_t zSchurComplementSetupGPU(int_t k, msgs_t* msgs, packLUInfo_t*,
				      int_t*, int_t*, int_t*, gEtreeInfo_t*,
				      factNodelists_t*, scuBufs_t*,
				      zLUValSubBuf_t* LUvsb, gridinfo_t *,
				      zLUstruct_t *, HyP_t*);
extern doublecomplex* zgetBigV(int_t, int_t);
extern doublecomplex* zgetBigU(int_t, gridinfo_t *, zLUstruct_t *);
extern int_t getBigUSize(int_t, gridinfo_t *, zLUstruct_t *);
// permutation from superLU default
extern int_t* getPerm_c_supno(int_t nsupers, superlu_dist_options_t *,
			      zLUstruct_t *, gridinfo_t *);
extern void getSCUweight(int_t nsupers, treeList_t* treeList, zLUstruct_t *, gridinfo3d_t *);

    /* from treeFactorization.h */
extern int_t zLluBufInit(zLUValSubBuf_t*, zLUstruct_t *);
extern int_t zinitScuBufs(int_t ldt, int_t num_threads, int_t nsupers,
			  scuBufs_t*, zLUstruct_t*, gridinfo_t *);
extern int zfreeScuBufs(scuBufs_t* scuBufs);

// the generic tree factoring code 
extern int_t treeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t zsparseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo,
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t zdenseTreeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);

extern int_t zsparseTreeFactor_ASYNC(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    zLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);
extern zLUValSubBuf_t** zLluBufInitArr(int_t numLA, zLUstruct_t *LUstruct);
extern int zLluBufFreeArr(int_t numLA, zLUValSubBuf_t **LUvsbs);
extern diagFactBufs_t** zinitDiagFactBufsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid);
extern int zfreeDiagFactBufsArr(int_t mxLeafNode, diagFactBufs_t** dFBufs);
extern int_t zinitDiagFactBufs(int_t ldt, diagFactBufs_t* dFBuf);
extern int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
			    gridinfo_t *grid, SCT_t *SCT);
extern int_t checkRecvLDiag(int_t k, commRequests_t *comReqs, gridinfo_t *, SCT_t *);

    /* from ancFactorization.h (not called) */
extern int_t ancestorFactor(
    int_t ilvl,             // level of factorization 
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    zLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub, int *info
);

/*=====================*/
#endif  // end 3D prototypes

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

