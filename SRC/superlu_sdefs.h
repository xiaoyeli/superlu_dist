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

#ifndef __SUPERLU_SDEFS /* allow multiple inclusions */
#define __SUPERLU_SDEFS

/*
 * File name:	superlu_sdefs.h
 * Purpose:     Distributed SuperLU data types and function prototypes
 * History:
 */

#include "superlu_defs.h"

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
 * R      float*, dimension (A->nrow)
 *        The row scale factors for A.
 *        If DiagScale = ROW or BOTH, A is multiplied on the left by diag(R).
 *        If DiagScale = NOEQUIL or COL, R is not defined.
 *
 * C      float*, dimension (A->ncol)
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
    float *R;
    float *C; 
    int_t  *perm_r;
    int_t  *perm_c;
} sScalePermstruct_t;

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
    int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk])    */   
    long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */     
    long int Lrowind_bc_cnt;

    float **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    float *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])  */   
    long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                */    
    long int Lnzval_bc_cnt;
    
    float **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    float *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])  */   
    long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)              */   
    long int Linv_bc_cnt;
    
    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr */
    int_t *Lindval_loc_bc_dat;  /* size sum of sizes of Lindval_loc_bc_ptr[lk]) */   
    long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)                  */   
    long int Lindval_loc_bc_cnt;  
    int_t   *Unnz; /* number of nonzeros per block column in U*/
    int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */
    float **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    float *Uinv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */   
    long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */   
    long int Uinv_bc_cnt;

    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */   
    long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Ufstnz_br_cnt;
    
    float  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                  */
    float  *Unzval_br_dat;   /* size sum of sizes of Unzval_br_ptr[lk]) */   
    long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Unzval_br_cnt;
    
        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *LRtree_ptr;       /* size ceil(NSUPERS/Pr)                */
    C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *URtree_ptr;       /* size ceil(NSUPERS/Pr)			*/
#if 0
    int_t   *Lsub_buf;        /* Buffer for the remote subscripts of L */
    float  *Lval_buf;        /* Buffer for the remote nonzeros of L   */
    int_t   *Usub_buf;        /* Buffer for the remote subscripts of U */
    float  *Uval_buf;        /* Buffer for the remote nonzeros of U   */
#endif
    int_t   *Lsub_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote subscripts of L*/
    float  *Lval_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote nonzeros of L  */
    int_t   *Usub_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote subscripts of U */
    float  *Uval_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote nonzeros of U   */
    float  *ujrow;           /* used in panel factorization.          */
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
    /* 1/15/22 Sherry: changed int_t to int type */
    int   *fmod;            /* Modification count for L-solve            */
    int   **fsendx_plist;   /* Column process list to send down Xk       */
    int   *frecv;           /* Modifications to be recv'd in proc row    */
    int   nfrecvx;          /* Number of Xk I will receive in L-solve    */
    int   nfsendx;          /* Number of Xk I will send in L-solve       */
    int   *bmod;            /* Modification count for U-solve            */
    int   **bsendx_plist;   /* Column process list to send down Xk       */
    int   *brecv;           /* Modifications to be recv'd in proc row    */
    int   nbrecvx;          /* Number of Xk I will receive in U-solve    */
    int   nbsendx;          /* Number of Xk I will send in U-solve       */
    int   *mod_bit;         /* Flag contribution from each row blocks    */

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
    Ucb_indptr_t *Ucb_inddat;
    long int *Ucb_indoffset;
    long int Ucb_indcnt;  
    
    int_t  **Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t  *Ucb_valdat;      
    long int *Ucb_valoffset;
    long int Ucb_valcnt;

    /* some additional counters for L solve */
    int_t n;
    int_t nleaf;
    int_t nfrecvmod;
    int_t inv; /* whether the diagonal block is inverted*/

    /* The following variables are used in GPU trisolve*/
#ifdef GPU_ACC
    int_t *d_Lrowind_bc_dat;     
    long int *d_Lrowind_bc_offset;      
    float *d_Lnzval_bc_dat;     
    long int *d_Lnzval_bc_offset;     
    float *d_Linv_bc_dat ;     
    float *d_Uinv_bc_dat ;     
    long int *d_Linv_bc_offset ;     
    long int *d_Uinv_bc_offset ;     
    int_t *d_Lindval_loc_bc_dat ;     
    long int *d_Lindval_loc_bc_offset ;     

    int_t *d_Urbs;
    int_t   *d_Ufstnz_br_dat;  
    long int *d_Ufstnz_br_offset;  
    float *d_Unzval_br_dat;   
    long int *d_Unzval_br_offset; 

    int_t  *d_Ucb_valdat;      
    long int *d_Ucb_valoffset;    
    Ucb_indptr_t *d_Ucb_inddat;
    long int *d_Ucb_indoffset;

    int_t  *d_ilsum ;
    int_t *d_xsup ;
    C_Tree  *d_LBtree_ptr ;
    C_Tree  *d_LRtree_ptr ;
    C_Tree  *d_UBtree_ptr ;
    C_Tree  *d_URtree_ptr ;    
#endif

} sLocalLU_t;


typedef struct {
    int_t *etree;
    Glu_persist_t *Glu_persist;
    sLocalLU_t *Llu;
    char dt;
} sLUstruct_t;


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
    void *val_tosend;   /* X values to be sent to other processes */
    void *val_torecv;   /* X values to be received from other processes */
    int_t TotalIndSend;   /* Total number of indices to be sent
			     (also total number of values to be received) */
    int_t TotalValSend;   /* Total number of values to be sent.
			     (also total number of indices to be received) */
} psgsmv_comm_t;

/*-- Data structure holding the information for the solution phase --*/
typedef struct {
    int_t *row_to_proc;
    int_t *inv_perm_c;
    int_t num_diag_procs, *diag_procs, *diag_len;
    psgsmv_comm_t *gsmv_comm; /* communication metadata for SpMV,
         	       		      required by IterRefine.          */
    pxgstrs_comm_t *gstrs_comm;  /* communication metadata for SpTRSV. */
    int_t *A_colind_gsmv; /* After psgsmv_init(), the global column
                             indices of A are translated into the relative
                             positions in the gathered x-vector.
                             This is re-used in repeated calls to psgsmv() */
    int_t *xrow_to_proc; /* used by PDSLin */
    NRformat_loc3d* A3d; /* Point to 3D {A, B} gathered on 2D layer 0.
                            This needs to be peresistent between
			    3D factorization and solve.  */
} sSOLVEstruct_t;



/*==== For 3D code ====*/

// new structures for pdgstrf_4_8 

#if 0  // Sherry: moved to superlu_defs.h
typedef struct
{
    int_t nub;
    int_t klst;
    int_t ldu;
    int_t* usub;
    float* uval;
} uPanelInfo_t;

typedef struct
{
    int_t *lsub;
    float *lusup;
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
    float *lookAhead_L_buff, *Remain_L_buff;
    int_t lookAheadBlk;  /* number of blocks in look-ahead window */
    int_t RemainBlk ;    /* number of blocks outside look-ahead window */
    int_t  num_look_aheads, nsupers;
    int_t ldu, ldu_Phi;
    int_t num_u_blks, num_u_blks_Phi;

    int_t jj_cpu;
    float *bigU_Phi;
    float *bigU_host;
    int_t Lnbrow;
    int_t Rnbrow;

    int_t buffer_size;
    int_t bigu_size;
    int offloadCondition;
    int superlu_acc_offload;
    int nGPUStreams;
} HyP_t;

#endif  // Above are moved to superlu_defs.h


typedef struct 
{
    int_t * Lsub_buf ;
    float * Lval_buf ;
    int_t * Usub_buf ;
    float * Uval_buf ;
} sLUValSubBuf_t;

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
    sLUValSubBuf_t  *LUvsb;
} strf3Dpartition_t;

typedef struct
{
    float *bigU;
    float *bigV;
} sscuBufs_t;

typedef struct
{   
    float* BlockLFactor;
    float* BlockUFactor;
} sdiagFactBufs_t;

/*=====================*/

/***********************************************************************
 * Function prototypes
 ***********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


/* Supernodal LU factor related */
extern void
sCreate_CompCol_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, float *,
			    int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_CompRowLoc_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, int_t,
			       int_t, float *, int_t *, int_t *,
			       Stype_t, Dtype_t, Mtype_t);
extern void
sCompRow_to_CompCol_dist(int_t, int_t, int_t, float *, int_t *, int_t *,
                         float **, int_t **, int_t **);
extern int
psCompRow_loc_to_CompCol_global(int_t, SuperMatrix *, gridinfo_t *,
	 		        SuperMatrix *);
extern void
sCopy_CompCol_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void
sCreate_Dense_Matrix_dist(SuperMatrix *, int_t, int_t, float *, int_t,
			  Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_SuperNode_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, float *,
			      int_t *, int_t *, int_t *, int_t *, int_t *,
			      Stype_t, Dtype_t, Mtype_t);
extern void
sCopy_Dense_Matrix_dist(int_t, int_t, float *, int_t,
                        float *, int_t);

extern void    sallocateA_dist (int_t, int_t, float **, int_t **, int_t **);
extern void    sGenXtrue_dist (int_t, int_t, float *, int_t);
extern void    sFillRHS_dist (char *, int_t, float *, int_t,
                              SuperMatrix *, float *, int_t);
extern int     screate_matrix(SuperMatrix *, int, float **, int *,
			      float **, int *, FILE *, gridinfo_t *);
extern int     screate_matrix_rb(SuperMatrix *, int, float **, int *,
			      float **, int *, FILE *, gridinfo_t *);
extern int     screate_matrix_dat(SuperMatrix *, int, float **, int *,
			      float **, int *, FILE *, gridinfo_t *);
extern int screate_matrix_postfix(SuperMatrix *, int, float **, int *,
				  float **, int *, FILE *, char *, gridinfo_t *);

extern void   sScalePermstructInit(const int_t, const int_t, 
                                      sScalePermstruct_t *);
extern void   sScalePermstructFree(sScalePermstruct_t *);

/* Driver related */
extern void    sgsequ_dist (SuperMatrix *, float *, float *, float *,
			    float *, float *, int_t *);
extern float  slangs_dist (char *, SuperMatrix *);
extern void    slaqgs_dist (SuperMatrix *, float *, float *, float,
			    float, float, char *);
extern void    psgsequ (SuperMatrix *, float *, float *, float *,
			float *, float *, int_t *, gridinfo_t *);
extern float  pslangs (char *, SuperMatrix *, gridinfo_t *);
extern void    pslaqgs (SuperMatrix *, float *, float *, float,
			float, float, char *);
extern int     psPermute_Dense_Matrix(int_t, int_t, int_t [], int_t[],
				      float [], int, float [], int, int,
				      gridinfo_t *);

extern int     sp_strsv_dist (char *, char *, char *, SuperMatrix *,
			      SuperMatrix *, float *, int *);
extern int     sp_sgemv_dist (char *, float, SuperMatrix *, float *,
			      int, float, float *, int);
extern int     sp_sgemm_dist (char *, int, float, SuperMatrix *,
                        float *, int, float, float *, int);

extern float sdistribute(superlu_dist_options_t *,
                         int_t, SuperMatrix *, Glu_freeable_t *,
			 sLUstruct_t *, gridinfo_t *);
extern void  psgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *,
			      sScalePermstruct_t *, float *,
			      int, int, gridinfo_t *, sLUstruct_t *, float *,
			      SuperLUStat_t *, int *);
extern float psdistribute(superlu_dist_options_t *, int_t, SuperMatrix *,
			 sScalePermstruct_t *, Glu_freeable_t *,
			 sLUstruct_t *, gridinfo_t *);
extern void  psgssvx(superlu_dist_options_t *, SuperMatrix *,
		     sScalePermstruct_t *, float *,
		     int, int, gridinfo_t *, sLUstruct_t *,
		     sSOLVEstruct_t *, float *, SuperLUStat_t *, int *);
extern void  psCompute_Diag_Inv(int_t, sLUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  sSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, sLUstruct_t *, gridinfo_t *, sSOLVEstruct_t *);
extern void sSolveFinalize(superlu_dist_options_t *, sSOLVEstruct_t *);
extern void sDestroy_A3d_gathered_on_2d(sSOLVEstruct_t *, gridinfo3d_t *);
extern int_t psgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, sSOLVEstruct_t *);
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  sldperm_dist(int, int, int_t, int_t [], int_t [],
		    float [], int_t *, float [], float []);
extern int  sstatic_schedule(superlu_dist_options_t *, int, int,
		            sLUstruct_t *, gridinfo_t *, SuperLUStat_t *,
			    int_t *, int_t *, int *);
extern void sLUstructInit(const int_t, sLUstruct_t *);
extern void sLUstructFree(sLUstruct_t *);
extern void sDestroy_LU(int_t, gridinfo_t *, sLUstruct_t *);
extern void sDestroy_Tree(int_t, gridinfo_t *, sLUstruct_t *);
extern void sscatter_l (int ib, int ljb, int nsupc, int_t iukp, int_t* xsup,
			int klst, int nbrow, int_t lptr, int temp_nbrow,
			int_t* usub, int_t* lsub, float *tempv,
			int* indirect_thread, int* indirect2,
			int_t ** Lrowind_bc_ptr, float **Lnzval_bc_ptr,
			gridinfo_t * grid);
extern void sscatter_u (int ib, int jb, int nsupc, int_t iukp, int_t * xsup,
                        int klst, int nbrow, int_t lptr, int temp_nbrow,
                        int_t* lsub, int_t* usub, float* tempv,
                        int_t ** Ufstnz_br_ptr, float **Unzval_br_ptr,
                        gridinfo_t * grid);
extern int_t psgstrf(superlu_dist_options_t *, int, int, float anorm,
		    sLUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);

/* #define GPU_PROF
#define IPM_PROF */

/* Solve related */
extern void psgstrs_Bglobal(superlu_dist_options_t *,
                             int_t, sLUstruct_t *, gridinfo_t *,
			     float *, int_t, int, SuperLUStat_t *, int *);
extern void psgstrs(superlu_dist_options_t *, int_t,
                    sLUstruct_t *, sScalePermstruct_t *, gridinfo_t *,
		    float *, int_t, int_t, int_t, int, sSOLVEstruct_t *,
		    SuperLUStat_t *, int *);
extern void psgstrf2_trsm(superlu_dist_options_t * options, int_t k0, int_t k,
			  double thresh, Glu_persist_t *, gridinfo_t *,
			  sLocalLU_t *, MPI_Request *, int tag_ub,
			  SuperLUStat_t *, int *info);
extern void psgstrs2_omp(int_t k0, int_t k, Glu_persist_t *, gridinfo_t *,
			 sLocalLU_t *, Ublock_info_t *, SuperLUStat_t *);
extern int_t psReDistribute_B_to_X(float *B, int_t m_loc, int nrhs, int_t ldb,
				   int_t fst_row, int_t *ilsum, float *x,
				   sScalePermstruct_t *, Glu_persist_t *,
				   gridinfo_t *, sSOLVEstruct_t *);
extern void slsum_fmod(float *, float *, float *, float *,
		       int, int, int_t , int *fmod, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, sLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);
extern void slsum_bmod(float *, float *, float *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, sLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void slsum_fmod_inv(float *, float *, float *, float *,
		       int, int_t , int *fmod,
		       int_t *, gridinfo_t *, sLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void slsum_fmod_inv_master(float *, float *, float *, float *,
		       int, int, int_t , int *fmod, int_t,
		       int_t *, gridinfo_t *, sLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void slsum_bmod_inv(float *, float *, float *, float *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, sLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void slsum_bmod_inv_master(float *, float *, float *, float *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, sLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int, int);

extern void sComputeLevelsets(int , int_t , gridinfo_t *,
		  Glu_persist_t *, sLocalLU_t *, int_t *);               
			   
#ifdef GPU_ACC               
extern void slsum_fmod_inv_gpu_wrap(int_t, int_t, int_t, int_t, float *, float *, int, int, int_t , int *fmod, C_Tree  *, C_Tree  *, int_t *, int_t *, int64_t *, float *, int64_t *, float *, int64_t *, int_t *, int64_t *, int_t *, gridinfo_t *, float * , float * , int_t );
extern void slsum_bmod_inv_gpu_wrap(superlu_dist_options_t *,
int_t, int_t, int_t, int_t, float *, float *,int,int, int_t , int *bmod, C_Tree  *, C_Tree  *, int_t *, int_t *,int_t *, int64_t *, float *, int64_t *, int_t  *, int64_t *, Ucb_indptr_t *, int64_t *, float *, int64_t *,int_t *,gridinfo_t *);
#endif

extern void psgsrfs(superlu_dist_options_t *, int_t,
                    SuperMatrix *, float, sLUstruct_t *,
		    sScalePermstruct_t *, gridinfo_t *,
		    float [], int_t, float [], int_t, int,
		    sSOLVEstruct_t *, float *, SuperLUStat_t *, int *);
extern void psgsrfs_ABXglobal(superlu_dist_options_t *, int_t,
                  SuperMatrix *, float, sLUstruct_t *,
		  gridinfo_t *, float *, int_t, float *, int_t,
		  int, float *, SuperLUStat_t *, int *);
extern int   psgsmv_AXglobal_setup(SuperMatrix *, Glu_persist_t *,
				   gridinfo_t *, int_t *, int_t *[],
				   float *[], int_t *[], int_t []);
extern int  psgsmv_AXglobal(int_t, int_t [], float [], int_t [],
	                       float [], float []);
extern int  psgsmv_AXglobal_abs(int_t, int_t [], float [], int_t [],
				 float [], float []);
extern void psgsmv_init(SuperMatrix *, int_t *, gridinfo_t *,
			psgsmv_comm_t *);
extern void psgsmv(int_t, SuperMatrix *, gridinfo_t *, psgsmv_comm_t *,
		   float x[], float ax[]);
extern void psgsmv_finalize(psgsmv_comm_t *);

/* Memory-related */
extern float  *floatMalloc_dist(int_t);
extern float  *floatCalloc_dist(int_t);
extern void  *duser_malloc_dist (int_t, int_t);
extern void  duser_free_dist (int_t, int_t);
extern int_t sQuerySpace_dist(int_t, sLUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void sClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void sCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void sZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void sScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, float);
extern void sScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, float);
extern void sZeroLblocks(int, int, gridinfo_t *, sLUstruct_t *);
extern void sZeroUblocks(int iam, int n, gridinfo_t *, sLUstruct_t *);
extern void    sfill_dist (float *, int_t, float);
extern void    sinf_norm_error_dist (int_t, int_t, float*, int_t,
                                     float*, int_t, gridinfo_t*);
extern void    psinf_norm_error(int, int_t, int_t, float [], int_t,
				float [], int_t , MPI_Comm);
extern void  sreadhb_dist (int, FILE *, int_t *, int_t *, int_t *,
			   float **, int_t **, int_t **);
extern void  sreadtriple_dist(FILE *, int_t *, int_t *, int_t *,
			 float **, int_t **, int_t **);
extern void  sreadtriple_noheader(FILE *, int_t *, int_t *, int_t *,
			 float **, int_t **, int_t **);
extern void  sreadrb_dist(int, FILE *, int_t *, int_t *, int_t *,
		     float **, int_t **, int_t **);
extern void  sreadMM_dist(FILE *, int_t *, int_t *, int_t *,
	                  float **, int_t **, int_t **);
extern int  sread_binary(FILE *, int_t *, int_t *, int_t *,
	                  float **, int_t **, int_t **);

/* Distribute the data for numerical factorization */
extern float sdist_psymbtonum(superlu_dist_options_t *, int_t, SuperMatrix *,
                                sScalePermstruct_t *, Pslu_freeable_t *,
                                sLUstruct_t *, gridinfo_t *);
extern void psGetDiagU(int_t, sLUstruct_t *, gridinfo_t *, float *);

extern int  s_c2cpp_GetHWPM(SuperMatrix *, gridinfo_t *, sScalePermstruct_t *);

/* Routines for debugging */
extern void  sPrintLblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
		 	   sLocalLU_t *);
extern void  sPrintUblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
			   sLocalLU_t *);
extern void  sPrint_CompCol_Matrix_dist(SuperMatrix *);
extern void  sPrint_Dense_Matrix_dist(SuperMatrix *);
extern int   sPrint_CompRowLoc_Matrix_dist(SuperMatrix *);
extern int   file_sPrint_CompRowLoc_Matrix_dist(FILE *fp, SuperMatrix *A);
extern void  Printfloat5(char *, int_t, float *);
extern int   file_Printfloat5(FILE *, char *, int_t, float *);

extern void sGenCOOLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, sLocalLU_t *, int_t** , int_t** , float ** , int_t* , int_t* );
extern void sGenCSCLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, sLocalLU_t *, float **, int_t **, int_t **, int_t*, int_t*);
extern void sGenCSRLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, sLocalLU_t *, float **, int_t **, int_t **, int_t*, int_t*);


/* BLAS */

#ifdef USE_VENDOR_BLAS
extern void sgemm_(const char*, const char*, const int*, const int*, const int*,
                  const float*, const float*, const int*, const float*,
                  const int*, const float*, float*, const int*, int, int);
extern void strsv_(char*, char*, char*, int*, float*, int*,
                  float*, int*, int, int, int);
extern void strsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const float*, const float*, const int*,
		  float*, const int*, int, int, int, int);
extern void sgemv_(const char *, const int *, const int *, const float *,
                  const float *a, const int *, const float *, const int *,
		  const float *, float *, const int *, int);

#else
extern int sgemm_(const char*, const char*, const int*, const int*, const int*,
                   const float*,  const float*,  const int*,  const float*,
                   const int*,  const float*, float*, const int*);
extern int strsv_(char*, char*, char*, int*, float*, int*,
                  float*, int*);
extern int strsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const float*, const float*, const int*,
		  float*, const int*);
extern void sgemv_(const char *, const int *, const int *, const float *,
                  const float *a, const int *, const float *, const int *,
		  const float *, float *, const int *);
#endif

extern void sger_(const int*, const int*, const float*,
                 const float*, const int*, const float*, const int*,
		 float*, const int*);

extern int sscal_(const int *n, const float *alpha, float *dx, const int *incx);
extern int saxpy_(const int *n, const float *alpha, const float *x, 
	               const int *incx, float *y, const int *incy);

/* SuperLU BLAS interface: ssuperlu_blas.c  */
extern int superlu_sgemm(const char *transa, const char *transb,
                  int m, int n, int k, float alpha, float *a,
                  int lda, float *b, int ldb, float beta, float *c, int ldc);
extern int superlu_strsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag, const int m, const int n,
                  const float alpha, const float *a,
                  const int lda, float *b, const int ldb);
extern int superlu_sger(const int m, const int n, const float alpha,
                 const float *x, const int incx, const float *y,
                 const int incy, float *a, const int lda);
extern int superlu_sscal(const int n, const float alpha, float *x, const int incx);
extern int superlu_saxpy(const int n, const float alpha,
    const float *x, const int incx, float *y, const int incy);
extern int superlu_sgemv(const char *trans, const int m,
                  const int n, const float alpha, const float *a,
                  const int lda, const float *x, const int incx,
                  const float beta, float *y, const int incy);
extern int superlu_strsv(char *uplo, char *trans, char *diag,
                  int n, float *a, int lda, float *x, int incx);

#ifdef SLU_HAVE_LAPACK
extern void strtri_(char*, char*, int*, float*, int*, int*);
#endif

/*==== For 3D code ====*/
extern int screate_matrix3d(SuperMatrix *A, int nrhs, float **rhs,
                     int *ldb, float **x, int *ldx,
                     FILE *fp, gridinfo3d_t *grid3d);
extern int screate_matrix_postfix3d(SuperMatrix *A, int nrhs, float **rhs,
                           int *ldb, float **x, int *ldx,
                           FILE *fp, char * postfix, gridinfo3d_t *grid3d);
    
/* Matrix distributed in NRformat_loc in 3D process grid. It converts 
   it to a NRformat_loc distributed in 2D grid in grid-0 */
extern void sGatherNRformat_loc3d(fact_t Fact, NRformat_loc *A, float *B,
				   int ldb, int nrhs, gridinfo3d_t *grid3d,
				   NRformat_loc3d **);
extern int sScatter_B3d(NRformat_loc3d *A3d, gridinfo3d_t *grid3d);

extern void psgssvx3d (superlu_dist_options_t *, SuperMatrix *,
		       sScalePermstruct_t *, float B[], int ldb, int nrhs,
		       gridinfo3d_t *, sLUstruct_t *, sSOLVEstruct_t *, 
		       float *berr, SuperLUStat_t *, int *info);
extern int_t psgstrf3d(superlu_dist_options_t *, int m, int n, float anorm,
		       strf3Dpartition_t*, SCT_t *, sLUstruct_t *,
		       gridinfo3d_t *, SuperLUStat_t *, int *);
extern void sInit_HyP(HyP_t* HyP, sLocalLU_t *Llu, int_t mcb, int_t mrb );
extern void Free_HyP(HyP_t* HyP);
extern int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid);

    /* from scatter.h */
extern void
sblock_gemm_scatter( int_t lb, int_t j, Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info, float *L_mat, int ldl,
                    float *U_mat, int ldu,  float *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int *indirect, int *indirect2,
                    int_t **Lrowind_bc_ptr, float **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, float **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *, SuperLUStat_t *
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );

#ifdef _OPENMP
/*this version uses a lock to prevent multiple thread updating the same block*/
extern void
sblock_gemm_scatter_lock( int_t lb, int_t j, omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,  Remain_info_t *Remain_info,
                         float *L_mat, int_t ldl, float *U_mat, int_t ldu,
                         float *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int *indirect, int *indirect2,
                         int_t **Lrowind_bc_ptr, float **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, float **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );
#endif

extern int_t
sblock_gemm_scatterTopLeft( int_t lb,  int_t j, float* bigV,
				 int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,
				 int* indirect, int* indirect2,
                                 HyP_t* HyP, sLUstruct_t *, gridinfo_t*,
                                 SCT_t*SCT, SuperLUStat_t *
                               );
extern int_t 
sblock_gemm_scatterTopRight( int_t lb,  int_t j, float* bigV,
				  int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,
				  int* indirect, int* indirect2,
                                  HyP_t* HyP, sLUstruct_t *, gridinfo_t*,
                                  SCT_t*SCT, SuperLUStat_t * );
extern int_t
sblock_gemm_scatterBottomLeft( int_t lb,  int_t j, float* bigV,
				    int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt, 
				    int* indirect, int* indirect2,
                                    HyP_t* HyP, sLUstruct_t *, gridinfo_t*,
                                    SCT_t*SCT, SuperLUStat_t * );
extern int_t 
sblock_gemm_scatterBottomRight( int_t lb,  int_t j, float* bigV,
				     int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,
				     int* indirect, int* indirect2,
                                     HyP_t* HyP, sLUstruct_t *, gridinfo_t*,
                                     SCT_t*SCT, SuperLUStat_t * );

    /* from gather.h */
extern void sgather_u(int_t num_u_blks,
              Ublock_info_t *Ublock_info, int_t * usub,
              float *uval,  float *bigU,  int_t ldu,
              int_t *xsup, int_t klst                /* for SuperSize */
             );

extern void sgather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               float * lval, int_t LD_lval,
               float * L_buff );

extern void sRgather_L(int_t k, int_t *lsub, float *lusup, gEtreeInfo_t*,
		      Glu_persist_t *, gridinfo_t *, HyP_t *,
		      int_t *myIperm, int_t *iperm_c_supno );
extern void sRgather_U(int_t k, int_t jj0, int_t *usub, float *uval,
		      float *bigU, gEtreeInfo_t*, Glu_persist_t *,
		      gridinfo_t *, HyP_t *, int_t *myIperm,
		      int_t *iperm_c_supno, int_t *perm_u);

    /* from xtrf3Dpartition.h */
extern strf3Dpartition_t* sinitTrf3Dpartition(int_t nsupers,
					     superlu_dist_options_t *options,
					     sLUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern void sDestroy_trf3Dpartition(strf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d);

extern void s3D_printMemUse(strf3Dpartition_t*  trf3Dpartition,
			    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d);

//extern int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
//		       superlu_dist_options_t *options, sLocalLU_t *Llu,
//		       int_t* xsup, int_t num_look_aheads, int_t nsupers,
//		       int_t * iperm_c_supno);

extern void sinit3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t**  sForests, sLUstruct_t* LUstruct,
				  gridinfo3d_t* grid3d);

extern int_t sgatherAllFactoredLUFr(int_t* myZeroTrIdxs, sForest_t* sForests,
				   sLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
				   SCT_t* SCT );

    /* The following are from pdgstrf2.h */
extern int_t sLpanelUpdate(int_t off0, int_t nsupc, float* ublk_ptr,
			  int_t ld_ujrow, float* lusup, int_t nsupr, SCT_t*);
extern void Local_Sgstrf2(superlu_dist_options_t *options, int_t k,
			  double thresh, float *BlockUFactor, Glu_persist_t *,
			  gridinfo_t *, sLocalLU_t *,
                          SuperLUStat_t *, int *info, SCT_t*);
extern int_t sTrs2_GatherU(int_t iukp, int_t rukp, int_t klst,
			  int_t nsupc, int_t ldu, int_t *usub,
			  float* uval, float *tempv);
extern int_t sTrs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
			   int_t nsupc, int_t ldu, int_t *usub,
			   float* uval, float *tempv);
extern int_t sTrs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
                             int_t *usub, float* uval, float *tempv,
                             int_t knsupc, int nsupr, float* lusup,
                             Glu_persist_t *Glu_persist)  ;
extern void psgstrs2
#ifdef _CRAY
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    sLocalLU_t *Llu, SuperLUStat_t *stat, _fcd ftcs1, _fcd ftcs2, _fcd ftcs3
);
#else
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    sLocalLU_t *Llu, SuperLUStat_t *stat
);
#endif

extern void psgstrf2(superlu_dist_options_t *, int_t nsupers, int_t k0,
		     int_t k, double thresh, Glu_persist_t *, gridinfo_t *,
		     sLocalLU_t *, MPI_Request *, int, SuperLUStat_t *, int *);

    /* from p3dcomm.h */
extern int_t sAllocLlu_3d(int_t nsupers, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t sp3dScatter(int_t n, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t sscatter3dLPanels(int_t nsupers,
                       sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t sscatter3dUPanels(int_t nsupers,
                       sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t scollect3dLpanels(int_t layer, int_t nsupers, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t scollect3dUpanels(int_t layer, int_t nsupers, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t sp3dCollect(int_t layer, int_t n, sLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
/*zero out LU non zero entries*/
extern int_t szeroSetLU(int_t nnodes, int_t* nodeList , sLUstruct_t *, gridinfo3d_t*);
extern int sAllocGlu_3d(int_t n, int_t nsupers, sLUstruct_t *);
extern int sDeAllocLlu_3d(int_t n, sLUstruct_t *, gridinfo3d_t*);
extern int sDeAllocGlu_3d(sLUstruct_t *);

/* Reduces L and U panels of nodes in the List nodeList (size=nnnodes)
receiver[L(nodelist)] =sender[L(nodelist)] +receiver[L(nodelist)]
receiver[U(nodelist)] =sender[U(nodelist)] +receiver[U(nodelist)]
*/
int_t sreduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        float* Lval_buf, float* Uval_buf,
                        sLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
/*reduces all nodelists required in a level*/
extern int sreduceAllAncestors3d(int_t ilvl, int_t* myNodeCount,
                           int_t** treePerm,
                           sLUValSubBuf_t* LUvsb,
                           sLUstruct_t* LUstruct,
                           gridinfo3d_t* grid3d,
                           SCT_t* SCT );
/*
	Copies factored L and U panels from sender grid to receiver grid
	receiver[L(nodelist)] <-- sender[L(nodelist)];
	receiver[U(nodelist)] <-- sender[U(nodelist)];
*/
int_t sgatherFactoredLU(int_t sender, int_t receiver,
                       int_t nnodes, int_t *nodeList, sLUValSubBuf_t*  LUvsb,
                       sLUstruct_t* LUstruct, gridinfo3d_t* grid3d,SCT_t* SCT );

/*Gathers all the L and U factors to grid 0 for solve stage 
	By  repeatidly calling above function*/
int_t sgatherAllFactoredLU(strf3Dpartition_t*  trf3Dpartition, sLUstruct_t* LUstruct,
			   gridinfo3d_t* grid3d, SCT_t* SCT );

/*Distributes data in each layer and initilizes ancestors
 as zero in required nodes*/
int_t sinit3DLUstruct( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                      int_t* nodeCount, int_t** nodeList,
                      sLUstruct_t* LUstruct, gridinfo3d_t* grid3d);

int_t szSendLPanel(int_t k, int_t receiver,
		   sLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t szRecvLPanel(int_t k, int_t sender, float alpha, 
                   float beta, float* Lval_buf,
		   sLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t szSendUPanel(int_t k, int_t receiver,
		   sLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t szRecvUPanel(int_t k, int_t sender, float alpha,
                   float beta, float* Uval_buf,
		   sLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);

    /* from communication_aux.h */
extern int_t sIBcast_LPanel (int_t k, int_t k0, int_t* lsub, float* lusup,
			     gridinfo_t *, int* msgcnt, MPI_Request *,
			     int **ToSendR, int_t *xsup, int );
extern int_t sBcast_LPanel(int_t k, int_t k0, int_t* lsub, float* lusup,
			   gridinfo_t *, int* msgcnt, int **ToSendR,
			   int_t *xsup , SCT_t*, int);
extern int_t sIBcast_UPanel(int_t k, int_t k0, int_t* usub, float* uval,
			    gridinfo_t *, int* msgcnt, MPI_Request *,
			    int *ToSendD, int );
extern int_t sBcast_UPanel(int_t k, int_t k0, int_t* usub, float* uval,
			   gridinfo_t *, int* msgcnt, int *ToSendD, SCT_t*, int);
extern int_t sIrecv_LPanel (int_t k, int_t k0,  int_t* Lsub_buf, 
			    float* Lval_buf, gridinfo_t *,
			    MPI_Request *, sLocalLU_t *, int);
extern int_t sIrecv_UPanel(int_t k, int_t k0, int_t* Usub_buf, float*,
			   sLocalLU_t *, gridinfo_t*, MPI_Request *, int);
extern int_t sWait_URecv(MPI_Request *, int* msgcnt, SCT_t *);
extern int_t sWait_LRecv(MPI_Request*, int* msgcnt, int* msgcntsU,
			 gridinfo_t *, SCT_t*);
extern int_t sISend_UDiagBlock(int_t k0, float *ublk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t sRecv_UDiagBlock(int_t k0, float *ublk_ptr, int_t size,
			      int_t src, gridinfo_t *, SCT_t*, int);
extern int_t sPackLBlock(int_t k, float* Dest, Glu_persist_t *,
			 gridinfo_t *, sLocalLU_t *);
extern int_t sISend_LDiagBlock(int_t k0, float *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t sIRecv_UDiagBlock(int_t k0, float *ublk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t *,
			       SCT_t*, int);
extern int_t sIRecv_LDiagBlock(int_t k0, float *L_blk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t*, SCT_t*, int);
extern int_t sUDiagBlockRecvWait( int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
				  MPI_Request *, gridinfo_t *, sLUstruct_t *, SCT_t *);
extern int_t LDiagBlockRecvWait( int_t k, int_t* factored_U, MPI_Request *, gridinfo_t *);

#if (MPI_VERSION>2)
extern int_t sIBcast_UDiagBlock(int_t k, float *ublk_ptr, int_t size,
				MPI_Request *, gridinfo_t *);
extern int_t sIBcast_LDiagBlock(int_t k, float *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *);
#endif

    /* from trfCommWrapper.h */
extern int_t sDiagFactIBCast(int_t k,  int_t k0,
			     float *BlockUFactor, float *BlockLFactor,
			     int_t* IrecvPlcd_D, MPI_Request *, MPI_Request *,
			     MPI_Request *, MPI_Request *, gridinfo_t *,
			     superlu_dist_options_t *, double thresh,
			     sLUstruct_t *LUstruct, SuperLUStat_t *, int *info,
			     SCT_t *, int tag_ub);
extern int_t sUPanelTrSolve( int_t k, float* BlockLFactor, float* bigV,
			     int_t ldt, Ublock_info_t*, gridinfo_t *,
			     sLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t sLPanelUpdate(int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
			   MPI_Request *, float* BlockUFactor, gridinfo_t *,
			   sLUstruct_t *, SCT_t *);
extern int_t sUPanelUpdate(int_t k, int_t* factored_U, MPI_Request *,
			   float* BlockLFactor, float* bigV,
			   int_t ldt, Ublock_info_t*, gridinfo_t *,
			   sLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t sIBcastRecvLPanel(int_t k, int_t k0, int* msgcnt,
			       MPI_Request *, MPI_Request *,
			       int_t* Lsub_buf, float* Lval_buf,
			      int_t * factored, gridinfo_t *, sLUstruct_t *,
			      SCT_t *, int tag_ub);
extern int_t sIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt, MPI_Request *,
			       MPI_Request *, int_t* Usub_buf, float* Uval_buf,
			       gridinfo_t *, sLUstruct_t *, SCT_t *, int tag_ub);
extern int_t sWaitL(int_t k, int* msgcnt, int* msgcntU, MPI_Request *,
		    MPI_Request *, gridinfo_t *, sLUstruct_t *, SCT_t *);
extern int_t sWaitU(int_t k, int* msgcnt, MPI_Request *, MPI_Request *,
		   gridinfo_t *, sLUstruct_t *, SCT_t *);
extern int_t sLPanelTrSolve(int_t k, int_t* factored_L, float* BlockUFactor,
			    gridinfo_t *, sLUstruct_t *);

    /* from trfAux.h */
extern int getNsupers(int, Glu_persist_t *);
extern int_t initPackLUInfo(int_t nsupers, packLUInfo_t* packLUInfo);
extern int   freePackLUInfo(packLUInfo_t* packLUInfo);
extern int_t sSchurComplementSetup(int_t k, int *msgcnt, Ublock_info_t*,
				   Remain_info_t*, uPanelInfo_t *,
				   lPanelInfo_t *, int_t*, int_t *, int_t *,
				   float *bigU, int_t* Lsub_buf,
				   float* Lval_buf, int_t* Usub_buf,
				   float* Uval_buf, gridinfo_t *, sLUstruct_t *);
extern int_t sSchurComplementSetupGPU(int_t k, msgs_t* msgs, packLUInfo_t*,
				      int_t*, int_t*, int_t*, gEtreeInfo_t*,
				      factNodelists_t*, sscuBufs_t*,
				      sLUValSubBuf_t* LUvsb, gridinfo_t *,
				      sLUstruct_t *, HyP_t*);
extern float* sgetBigV(int_t, int_t);
extern float* sgetBigU(superlu_dist_options_t *,
                           int_t, gridinfo_t *, sLUstruct_t *);
// permutation from superLU default

    /* from treeFactorization.h */
extern int_t sLluBufInit(sLUValSubBuf_t*, sLUstruct_t *);
extern int_t sinitScuBufs(superlu_dist_options_t *,
                          int_t ldt, int_t num_threads, int_t nsupers,
			  sscuBufs_t*, sLUstruct_t*, gridinfo_t *);
extern int sfreeScuBufs(sscuBufs_t* scuBufs);

#if 0 // NOT CALLED
// the generic tree factoring code 
extern int_t treeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    sscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    sLUValSubBuf_t* LUvsb,
    sdiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);
#endif

extern int_t ssparseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo,
    commRequests_t *comReqs,    // lists of communication requests
    sscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    sLUValSubBuf_t* LUvsb,
    sdiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t sdenseTreeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    sscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    sLUValSubBuf_t* LUvsb,
    sdiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);

extern int_t ssparseTreeFactor_ASYNC(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    sscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    sLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    sdiagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);
extern sLUValSubBuf_t** sLluBufInitArr(int_t numLA, sLUstruct_t *LUstruct);
extern int sLluBufFreeArr(int_t numLA, sLUValSubBuf_t **LUvsbs);
extern sdiagFactBufs_t** sinitDiagFactBufsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid);
extern int sfreeDiagFactBufsArr(int_t mxLeafNode, sdiagFactBufs_t** dFBufs);
extern int_t sinitDiagFactBufs(int_t ldt, sdiagFactBufs_t* dFBuf);
extern int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
			    gridinfo_t *grid, SCT_t *SCT);
extern int_t checkRecvLDiag(int_t k, commRequests_t *comReqs, gridinfo_t *, SCT_t *);

#if 0 // NOT CALLED
/* from ancFactorization.h (not called) */
extern int_t ancestorFactor(
    int_t ilvl,             // level of factorization 
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    sscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    sLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    sdiagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    sLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub, int *info
);
#endif

/*== end 3D prototypes ===================*/


#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

