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
 * -- Distributed SuperLU routine (version 9.0) --
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
    int_t   **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc);
    	                         free'd in ztrs_compute_communication_structure routinies */
    int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk]) */
    long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)              */
    long int Lrowind_bc_cnt;

    doublecomplex **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc);
    	                         free'd in ztrs_compute_communication_structure routinies */
    doublecomplex *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])  */
    long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                */
    long int Lnzval_bc_cnt;

    doublecomplex **Linv_bc_ptr;    /* size ceil(NSUPERS/Pc);
    	                         free'd in ztrs_compute_communication_structure routinies */
    doublecomplex *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])  */
    long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)              */
    long int Linv_bc_cnt;

    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc);
                                     pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr;
    	                             free'd in ztrs_compute_communication_structure routinies */

    int_t *Lindval_loc_bc_dat;  /* size: sum of sizes of Lindval_loc_bc_ptr[lk]) */
    long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)  */
    long int Lindval_loc_bc_cnt;

    /* for new U format -> */
    int_t   **Ucolind_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
    int_t *Ucolind_bc_dat;  /* size: sum of sizes of Ucolind_bc_ptr[lk])    */
    int64_t *Ucolind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
    int64_t Ucolind_bc_cnt;

    doublecomplex **Unzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    doublecomplex *Unzval_bc_dat;  /* size: sum of sizes of Unzval_bc_ptr[lk])  */
    int64_t *Unzval_bc_offset;  /* size ceil(NSUPERS/Pc)                */
    int64_t Unzval_bc_cnt;

    int_t   **Uindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Ucolind_bc_ptr and Unzval_bc_ptr */
    int_t *Uindval_loc_bc_dat; /* size: sum of sizes of Uindval_loc_bc_ptr[lk]) */
    int64_t *Uindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)   */
    int64_t Uindval_loc_bc_cnt;

    int_t   **Uind_br_ptr; /* size ceil(NSUPERS/Pr) pointers to locations in Ucolind_bc_ptr for each block row */
    int_t *Uind_br_dat;  /* size: sum of sizes of Uind_br_ptr[lk])    */
    int64_t *Uind_br_offset;  /* size ceil(NSUPERS/Pr)                 */
    int64_t Uind_br_cnt;

    int_t   **Ucolind_br_ptr; /* size ceil(NSUPERS/Pr)                 */
    int_t *Ucolind_br_dat;  /* size: sum of sizes of Ucolind_br_ptr[lk])    */
    int64_t *Ucolind_br_offset;  /* size ceil(NSUPERS/Pr)                 */
    int64_t Ucolind_br_cnt;

    doublecomplex **Unzval_br_new_ptr;  /* size ceil(NSUPERS/Pr)                 */
    doublecomplex *Unzval_br_new_dat;  /* size: sum of sizes of Unzval_br_ptr[lk])  */
    int64_t *Unzval_br_new_offset;  /* size ceil(NSUPERS/Pr)                */
    int64_t Unzval_br_new_cnt;

    /* end for new U format <- */

    int_t   *Unnz; /* number of nonzeros per block column in U*/
    int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */
    doublecomplex **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    doublecomplex *Uinv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */
    long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
    long int Uinv_bc_cnt;

    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */
    long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Ufstnz_br_cnt;

    doublecomplex  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                  */
    doublecomplex  *Unzval_br_dat;   /* size sum of sizes of Unzval_br_ptr[lk]) */
    long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Unzval_br_cnt;

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
    int   nleaf;
    int   nroot;
    /*-- Auxiliary arrays used for forward/back solves. --*/
    int_t   *ilsum;           /* Starting position of each supernode in lsum
				 (local)  */
    int_t   ldalsum;          /* LDA of lsum (local) */
    int_t   SolveMsgSent;     /* Number of actual messages sent in LU-solve */
    int_t   SolveMsgVol;      /* Volume of messages sent in the solve phase */
    int   *bcols_masked;      /* Local block column IDs in my 2D grid */

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
    int_t nfrecvmod;
    int_t inv; /* whether the diagonal block is inverted*/
    int nbcol_masked; /*number of local block columns in my 2D grid*/

#ifdef GPU_ACC
    /* The following variables are used in GPU trisolve */

    int_t *d_Lrowind_bc_dat;
    long int *d_Lrowind_bc_offset;
    doublecomplex *d_Lnzval_bc_dat;
    long int *d_Lnzval_bc_offset;
    int_t *d_Ucolind_bc_dat;
    int64_t *d_Ucolind_bc_offset;
    int_t *d_Uind_br_dat;
    int64_t *d_Uind_br_offset;
    doublecomplex *d_Unzval_bc_dat;
    long int *d_Unzval_bc_offset;
    int_t *d_Ucolind_br_dat;  /* size: sum of sizes of Ucolind_br_ptr[lk])    */
    int64_t *d_Ucolind_br_offset;  /* size ceil(NSUPERS/Pr)                 */
    doublecomplex *d_Unzval_br_new_dat;  /* size: sum of sizes of Unzval_br_ptr[lk])  */
    int64_t *d_Unzval_br_new_offset;  /* size ceil(NSUPERS/Pr)                */

    doublecomplex *d_Linv_bc_dat ;
    doublecomplex *d_Uinv_bc_dat ;
    long int *d_Linv_bc_offset ;
    long int *d_Uinv_bc_offset ;
    int_t *d_Lindval_loc_bc_dat ;
    int64_t *d_Lindval_loc_bc_offset ;
    int_t *d_Uindval_loc_bc_dat ;
    int64_t *d_Uindval_loc_bc_offset ;
    int   *d_bcols_masked;      /* Local block column IDs in my 2D grid */

    //    long int *d_Lindval_loc_bc_offset ;
    //    int_t *d_Urbs;
    //    int_t   *d_Ufstnz_br_dat;
    //    long int *d_Ufstnz_br_offset;
    //    doublecomplex *d_Unzval_br_dat;
    //    long int *d_Unzval_br_offset;
    //    int_t  *d_Ucb_valdat;
    //    long int *d_Ucb_valoffset;
    //    Ucb_indptr_t *d_Ucb_inddat;
    //    long int *d_Ucb_indoffset;

    int_t  *d_ilsum ;
    int_t *d_xsup ;
    C_Tree  *d_LBtree_ptr ;
    C_Tree  *d_LRtree_ptr ;
    C_Tree  *d_UBtree_ptr ;
    C_Tree  *d_URtree_ptr ;
    gridinfo_t *d_grid;
#endif

} zLocalLU_t;

typedef struct
{
    int_t * Lsub_buf ;
    doublecomplex * Lval_buf ;
    int_t * Usub_buf ;
    doublecomplex * Uval_buf ;
} zLUValSubBuf_t;

typedef struct
{
    int_t nsupers;
    gEtreeInfo_t gEtreeInfo;
    int_t* iperm_c_supno;
    int_t* myNodeCount;
    int_t* myTreeIdxs;
    int_t* myZeroTrIdxs;
    int_t** treePerm;
    sForest_t** sForests;
    int_t* supernode2treeMap;
    int* supernodeMask;
    zLUValSubBuf_t  *LUvsb;
    SupernodeToGridMap_t* superGridMap;
    int maxLvl; // YL: store this to avoid the use of grid3d

    /* Sherry added the following 3 for variable size batch. 2/17/23 */
    int mxLeafNode; /* number of leaf nodes. */
    int *diagDims;  /* dimensions of the diagonal blocks at any level of the tree */
    int *gemmCsizes; /* sizes of the C matrices at any level of the tree. */
} ztrf3Dpartition_t;


typedef struct {
    int_t *etree;
    Glu_persist_t *Glu_persist;
    zLocalLU_t *Llu;
    ztrf3Dpartition_t *trf3Dpart;
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
    void  *val_tosend;   /* X values to be sent to other processes */
    void  *val_torecv;   /* X values to be received from other processes */
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
    NRformat_loc3d* A3d; /* Point to 3D {A, B} gathered on 2D layer 0.
                            This needs to be peresistent between
			    3D factorization and solve.  */
    #ifdef GPU_ACC
    doublecomplex *d_lsum, *d_lsum_save;      /* used for device lsum*/
    doublecomplex *d_x;         /* used for device solution vector*/
    int  *d_fmod_save, *d_fmod;         /* used for device fmod vector*/
    int  *d_bmod_save, *d_bmod;         /* used for device bmod vector*/
    #endif
} zSOLVEstruct_t;



/*==== For 3D code ====*/

// new structures for pdgstrf_4_8

#if 0  // Sherry: moved to superlu_defs.h
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
    int offloadCondition;
    int superlu_acc_offload;
    int nGPUStreams;
} HyP_t;

#endif  // Above are moved to superlu_defs.h


int_t scuStatUpdate(
    int_t knsupc,
    HyP_t* HyP,
    SCT_t* SCT,
    SuperLUStat_t *stat
    );



typedef struct
{
    doublecomplex *bigU;
    doublecomplex *bigV;
} zscuBufs_t;

typedef struct
{
    doublecomplex* BlockLFactor;
    doublecomplex* BlockUFactor;
} zdiagFactBufs_t;


typedef struct zxT_struct
{
	doublecomplex* xT;
	int_t ldaspaT;
	int_t* ilsumT;
} zxT_struct;

typedef struct zlsumBmod_buff_t
{
    doublecomplex * tX;    // buffer for reordered X
    doublecomplex * tU;    // buffer for packedU
    int_t *indCols; //
}zlsumBmod_buff_t;

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
extern void
zCompCol_to_CompRow_dist(int_t m, int_t n, int_t nnz,
                         doublecomplex *a, int_t *colptr, int_t *rowind,
                         doublecomplex **at, int_t **rowptr, int_t **colind);
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
extern int zcreate_matrix_postfix(SuperMatrix *, int, doublecomplex **, int *,
				  doublecomplex **, int *, FILE *, char *, gridinfo_t *);

extern void   zScalePermstructInit(const int_t, const int_t,
                                      zScalePermstruct_t *);
extern void   zScalePermstructFree(zScalePermstruct_t *);

/* Driver related */
extern void    zgsequ_dist (SuperMatrix *, double *, double *, double *,
			    double *, double *, int *);
extern double  zlangs_dist (char *, SuperMatrix *);
extern void    zlaqgs_dist (SuperMatrix *, double *, double *, double,
			    double, double, char *);
extern void    pzgsequ (SuperMatrix *, double *, double *, double *,
			double *, double *, int *, gridinfo_t *);
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

extern float zdistribute(superlu_dist_options_t *,
                         int_t, SuperMatrix *, Glu_freeable_t *,
			 zLUstruct_t *, gridinfo_t *);
extern void  pzgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *,
			      zScalePermstruct_t *, doublecomplex *,
			      int, int, gridinfo_t *, zLUstruct_t *, double *,
			      SuperLUStat_t *, int *);
extern float pzdistribute(superlu_dist_options_t *, int_t, SuperMatrix *,
			 zScalePermstruct_t *, Glu_freeable_t *,
			 zLUstruct_t *, gridinfo_t *);
extern float pzdistribute_allgrid(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     zScalePermstruct_t *ScalePermstruct,
	     Glu_freeable_t *Glu_freeable, zLUstruct_t *LUstruct,
	     gridinfo_t *grid, int* supernodeMask);

extern float pzdistribute_allgrid_index_only(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     zScalePermstruct_t *ScalePermstruct,
	     Glu_freeable_t *Glu_freeable, zLUstruct_t *LUstruct,
	     gridinfo_t *grid, int* supernodeMask);
extern void  pzgssvx(superlu_dist_options_t *, SuperMatrix *,
		     zScalePermstruct_t *, doublecomplex *,
		     int, int, gridinfo_t *, zLUstruct_t *,
		     zSOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void  pzCompute_Diag_Inv(int_t, zLUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  zSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, zLUstruct_t *, gridinfo_t *, zSOLVEstruct_t *);
extern void zSolveFinalize(superlu_dist_options_t *, zSOLVEstruct_t *);
extern void zDestroy_A3d_gathered_on_2d(zSOLVEstruct_t *, gridinfo3d_t *);
extern int_t pzgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, zSOLVEstruct_t *);
extern int_t pzgstrs_init_device_lsum_x(superlu_dist_options_t *, int_t , int_t , int_t , gridinfo_t *,
	     zLUstruct_t *, zSOLVEstruct_t *, int*);
extern int_t pzgstrs_delete_device_lsum_x(zSOLVEstruct_t *);
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  zldperm_dist(int, int, int_t, int_t [], int_t [],
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
extern int_t pzgstrf(superlu_dist_options_t *, int, int, double anorm,
		    zLUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);

/* #define GPU_PROF
#define IPM_PROF */

/* Solve related */
extern void pzgstrs_Bglobal(superlu_dist_options_t *,
                             int_t, zLUstruct_t *, gridinfo_t *,
			     doublecomplex *, int_t, int, SuperLUStat_t *, int *);
extern void pzgstrs(superlu_dist_options_t *, int_t,
                    zLUstruct_t *, zScalePermstruct_t *, gridinfo_t *,
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
		       int, int, int_t , int *fmod, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);
extern void zlsum_bmod(doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void zlsum_fmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int_t , int *fmod,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_fmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int, int_t , int *fmod, int_t,
		       int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_bmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void zlsum_bmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, zLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int, int);

extern void zComputeLevelsets(int , int_t , gridinfo_t *,
		  Glu_persist_t *, zLocalLU_t *, int_t *);

#ifdef GPU_ACC
extern void pzconvertU(superlu_dist_options_t *, gridinfo_t *, zLUstruct_t *, SuperLUStat_t *, int);

extern void zlsum_fmod_inv_gpu_wrap(int, int, int, int, doublecomplex *, doublecomplex *, int, int, int_t , int *fmod, C_Tree  *, C_Tree  *, int_t *, int_t *, int64_t *, doublecomplex *, int64_t *, doublecomplex *, int64_t *, int_t *, int64_t *, int_t *, int *, gridinfo_t *,
int_t , uint64_t* ,uint64_t* ,doublecomplex* ,doublecomplex* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int* ,int);

extern void zlsum_bmod_inv_gpu_wrap(superlu_dist_options_t *, int, int, int, int, doublecomplex *, doublecomplex *,int,int, int_t , int *, C_Tree  *, C_Tree  *, int_t *, int_t *, int64_t *,int_t *, int64_t *, int_t *, int64_t *, doublecomplex *, int64_t *, doublecomplex *, int64_t *, doublecomplex  *, int64_t *, int_t *, int64_t *, int_t *,gridinfo_t *,
                                    int_t, uint64_t*, uint64_t*, doublecomplex*, doublecomplex*,
                                    int*, int*, int*, int*,
                                    int*, int*, int*, int*, int*,
                                    int*, int*, int*, int*, int*, int*,
                                    int*, int*, int*, int); //int*); //int*, doublecomplex*);

#endif

extern void pzgsrfs(superlu_dist_options_t *, int_t,
                    SuperMatrix *, double, zLUstruct_t *,
		    zScalePermstruct_t *, gridinfo_t *,
		    doublecomplex [], int_t, doublecomplex [], int_t, int,
		    zSOLVEstruct_t *, double *, SuperLUStat_t *, int *);

extern void pzgsrfs3d(superlu_dist_options_t *, int_t,
            SuperMatrix *, double, zLUstruct_t *,
	        zScalePermstruct_t *, gridinfo3d_t *,
	        ztrf3Dpartition_t*  , doublecomplex *, int_t, doublecomplex *, int_t, int,
	        zSOLVEstruct_t *, double *, SuperLUStat_t *, int *);


extern void pzgsrfs_ABXglobal(superlu_dist_options_t *, int_t,
                  SuperMatrix *, double, zLUstruct_t *,
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

extern int_t zinitLsumBmod_buff(int_t ns, int nrhs, zlsumBmod_buff_t* lbmod_buf);
extern int_t zleafForestBackSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  zLUstruct_t * LUstruct,
                            zScalePermstruct_t * ScalePermstruct,
                            ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            doublecomplex * x, doublecomplex * lsum, doublecomplex * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, zlsumBmod_buff_t* lbmod_buf,
                            zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t znonLeafForestBackSolve3d( int_t treeId,  zLUstruct_t * LUstruct,
                                zScalePermstruct_t * ScalePermstruct,
                                ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                 doublecomplex * x, doublecomplex * lsum, zxT_struct *xT_s,doublecomplex * recvbuf,
                                MPI_Request * send_req,
                                int nrhs, zlsumBmod_buff_t* lbmod_buf,
                                zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t zlasum_bmod_Tree(int_t  pTree, int_t cTree, doublecomplex *lsum, doublecomplex *x,
                       zxT_struct *xT_s,
                       int    nrhs, zlsumBmod_buff_t* lbmod_buf,
                       zLUstruct_t * LUstruct,
                       ztrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat);
extern int_t zlsumForestBsolve(int_t k, int_t treeId,
                       doublecomplex *lsum, doublecomplex *x,  zxT_struct *xT_s,int    nrhs, zlsumBmod_buff_t* lbmod_buf,
                       zLUstruct_t * LUstruct,
                       ztrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat);

extern int_t  zbCastXk2Pck  (int_t k, zxT_struct *xT_s, int nrhs,
                     zLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer);

extern int_t  zlsumReducePrK (int_t k, doublecomplex*x, doublecomplex* lsum, doublecomplex* recvbuf, int nrhs,
                      zLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer);

extern int_t znonLeafForestForwardSolve3d( int_t treeId,  zLUstruct_t * LUstruct,
                                   zScalePermstruct_t * ScalePermstruct,
                                   ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                   doublecomplex * x, doublecomplex * lsum,
                                   zxT_struct *xT_s,
                                   doublecomplex * recvbuf, doublecomplex* rtemp,
                                   MPI_Request * send_req,
                                   int nrhs,
                                   zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);
extern int_t zleafForestForwardSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  zLUstruct_t * LUstruct,
                               zScalePermstruct_t * ScalePermstruct,
                               ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               doublecomplex * x, doublecomplex * lsum, doublecomplex * recvbuf, doublecomplex* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);


extern int ztrs_compute_communication_structure(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                           zScalePermstruct_t * ScalePermstruct,
                           int* supernodeMask, gridinfo_t *grid, SuperLUStat_t * stat);
extern int_t zreduceSolvedX_newsolve(int_t treeId, int_t sender, int_t receiver, doublecomplex* x, int nrhs,
                      ztrf3Dpartition_t*  trf3Dpartition, zLUstruct_t* LUstruct, gridinfo3d_t* grid3d, doublecomplex* recvbuf, xtrsTimer_t *xtrsTimer);

extern void zlsum_fmod_leaf (
  int_t treeId,
  ztrf3Dpartition_t*  trf3Dpartition,
    doublecomplex *lsum,    /* Sum of local modifications.                        */
    doublecomplex *x,       /* X array (local)                                    */
    doublecomplex *xk,      /* X[k].                                              */
    doublecomplex *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    zLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat, xtrsTimer_t *xtrsTimer);

extern void zlsum_fmod_leaf_newsolve (
    ztrf3Dpartition_t*  trf3Dpartition,
    doublecomplex *lsum,    /* Sum of local modifications.                        */
    doublecomplex *x,       /* X array (local)                                    */
    doublecomplex *xk,      /* X[k].                                              */
    doublecomplex *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    zLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer);


extern void zlsum_bmod_GG
(
    doublecomplex *lsum,        /* Sum of local modifications.                    */
    doublecomplex *x,           /* X array (local).                               */
    doublecomplex *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    zlsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    zLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat, xtrsTimer_t *xtrsTimer);

extern void zlsum_bmod_GG_newsolve (
    ztrf3Dpartition_t*  trf3Dpartition,
    doublecomplex *lsum,        /* Sum of local modifications.                    */
    doublecomplex *x,           /* X array (local).                               */
    doublecomplex *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    zlsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    zLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat
    , xtrsTimer_t *xtrsTimer);

extern int_t
pzReDistribute3d_B_to_X (doublecomplex *B, int_t m_loc, int nrhs, int_t ldb,
                       int_t fst_row, int_t * ilsum, doublecomplex *x,
                       zScalePermstruct_t * ScalePermstruct,
                       Glu_persist_t * Glu_persist,
                       gridinfo3d_t * grid3d, zSOLVEstruct_t * SOLVEstruct);


extern int_t
pzReDistribute3d_X_to_B (int_t n, doublecomplex *B, int_t m_loc, int_t ldb,
                       int_t fst_row, int nrhs, doublecomplex *x, int_t * ilsum,
                       zScalePermstruct_t * ScalePermstruct,
                       Glu_persist_t * Glu_persist, gridinfo3d_t * grid3d,
                       zSOLVEstruct_t * SOLVEstruct);

extern void
pzgstrs3d (superlu_dist_options_t *, int_t n, zLUstruct_t * LUstruct,
           zScalePermstruct_t * ScalePermstruct,
           ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, doublecomplex *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info);

extern void
pzgstrs3d_newsolve (superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
           zScalePermstruct_t * ScalePermstruct,
           ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, doublecomplex *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info);

extern int_t pzgsTrBackSolve3d(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                        zScalePermstruct_t * ScalePermstruct,
                        ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        doublecomplex *x3d, doublecomplex *lsum3d,
                        zxT_struct *xT_s,
                        doublecomplex * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pzgsTrForwardSolve3d(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                           zScalePermstruct_t * ScalePermstruct,
                           ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           doublecomplex *x3d, doublecomplex *lsum3d,
                           zxT_struct *xT_s,
                           doublecomplex * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pzgsTrForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                           zScalePermstruct_t * ScalePermstruct,
                           ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           doublecomplex *x3d, doublecomplex *lsum3d,
                           doublecomplex * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pzgsTrBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                        ztrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        doublecomplex *x3d, doublecomplex *lsum3d,
                        doublecomplex * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        zSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

int_t zbroadcastAncestor3d( ztrf3Dpartition_t*  trf3Dpartition,
			   zLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT );

extern int_t zlocalSolveXkYk( trtype_t trtype, int_t k, doublecomplex* x, int nrhs,
                      zLUstruct_t * LUstruct, gridinfo_t * grid,
                      SuperLUStat_t * stat);

extern int_t ziBcastXk2Pck(int_t k, doublecomplex* x, int nrhs,
                   int** sendList, MPI_Request *send_req,
                   zLUstruct_t * LUstruct, gridinfo_t * grid,xtrsTimer_t *xtrsTimer);

extern int_t ztrs_B_init3d(int_t nsupers, doublecomplex* x, int nrhs, zLUstruct_t * LUstruct, gridinfo3d_t *grid3d);
extern int_t ztrs_X_gather3d(doublecomplex* x, int nrhs, ztrf3Dpartition_t*  trf3Dpartition,
                     zLUstruct_t* LUstruct,
                     gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer);
extern int_t zfsolveReduceLsum3d(int_t treeId, int_t sender, int_t receiver, doublecomplex* lsum, doublecomplex* recvbuf, int nrhs,
                         ztrf3Dpartition_t*  trf3Dpartition, zLUstruct_t* LUstruct,
                          gridinfo3d_t* grid3d,xtrsTimer_t *xtrsTimer);

extern int_t zbsolve_Xt_bcast(int_t ilvl, zxT_struct *xT_s, int nrhs, ztrf3Dpartition_t*  trf3Dpartition,
                     zLUstruct_t * LUstruct,gridinfo3d_t* grid3d , xtrsTimer_t *xtrsTimer);

extern int_t zp2pSolvedX3d(int_t treeId, int_t sender, int_t receiver, doublecomplex* x, int nrhs,
                      ztrf3Dpartition_t*  trf3Dpartition, zLUstruct_t* LUstruct, gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer);

/* Memory-related */
extern doublecomplex  *doublecomplexMalloc_dist(int_t);
extern doublecomplex  *doublecomplexCalloc_dist(int_t);
extern double  *doubleMalloc_dist(int_t);
extern double  *doubleCalloc_dist(int_t);
extern void  *zuser_malloc_dist (int_t, int_t);
extern void  zuser_free_dist (int_t, int_t);
extern int_t zQuerySpace_dist(int_t, zLUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void zClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void zScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, doublecomplex);
extern void zScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, doublecomplex);
extern void zZeroLblocks(int, int, gridinfo_t *, zLUstruct_t *);
extern void zZeroUblocks(int iam, int n, gridinfo_t *, zLUstruct_t *);
extern double zMaxAbsLij(int iam, int n, Glu_persist_t *,
		 zLUstruct_t *, gridinfo_t *);
extern double zMaxAbsUij(int iam, int n, Glu_persist_t *,
		 zLUstruct_t *, gridinfo_t *);
extern void    zfill_dist (doublecomplex *, int_t, doublecomplex);
extern void    zinf_norm_error_dist (int_t, int_t, doublecomplex*, int_t,
                                     doublecomplex*, int_t, gridinfo_t*);
extern void    pzinf_norm_error(int, int_t, int_t, doublecomplex [], int_t,
				doublecomplex [], int_t , MPI_Comm);
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

extern void validateInput_pzgssvx3d(superlu_dist_options_t *, SuperMatrix *A,
       int ldb, int nrhs, gridinfo3d_t *, int *info);
extern void zallocScalePermstruct_RC(zScalePermstruct_t *, int_t m, int_t n);
extern void zscaleMatrixDiagonally(fact_t Fact, zScalePermstruct_t *, SuperMatrix *,
       	    		SuperLUStat_t *, gridinfo_t *, int *rowequ, int *colequ, int *iinfo);
extern void zperform_row_permutation(superlu_dist_options_t *, fact_t Fact,
           zScalePermstruct_t *, zLUstruct_t *LUstruct, int_t m, int_t n,
	       gridinfo_t *, SuperMatrix *A, SuperMatrix *GA, SuperLUStat_t *,
	       int job, int Equil, int *rowequ, int *colequ, int *iinfo);
extern double zcomputeA_Norm(int notran, SuperMatrix *, gridinfo_t *);
extern int ztrs_compute_communication_structure(superlu_dist_options_t *options,
       int_t n, zLUstruct_t *, zScalePermstruct_t * ScalePermstruct,
       int* supernodeMask, gridinfo_t *, SuperLUStat_t *);

/* Distribute the data for numerical factorization */
extern float zdist_psymbtonum(superlu_dist_options_t *, int_t, SuperMatrix *,
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

extern void zGenCOOLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, zLocalLU_t *, int_t** , int_t** , doublecomplex ** , int_t* , int_t* );
extern void zGenCSCLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, zLocalLU_t *, doublecomplex **, int_t **, int_t **, int_t*, int_t*);
extern void zGenCSRLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, zLocalLU_t *, doublecomplex **, int_t **, int_t **, int_t*, int_t*);

/* multi-GPU */
#ifdef GPU_ACC
// extern void create_nv_buffer(int* , int*, int* , int* );
extern void nv_init_wrapper(MPI_Comm);
// extern void nv_init_wrapper(int* c, char *v[], int* omp_mpi_level);
extern void zprepare_multiGPU_buffers(int,int,int,int,int,int);
extern void zdelete_multiGPU_buffers();
#endif

/* BLAS */

#ifdef USE_VENDOR_BLAS
extern void zgemm_(const char*, const char*, const int*, const int*, const int*,
                  const doublecomplex*, const doublecomplex*, const int*, const doublecomplex*,
                  const int*, const doublecomplex*, doublecomplex*, const int*, int, int);
extern void ztrsv_(char*, char*, char*, int*, doublecomplex*, int*,
                  doublecomplex*, int*, int, int, int);
extern void ztrsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const doublecomplex*, const doublecomplex*, const int*,
		  doublecomplex*, const int*, int, int, int, int);
extern void zgemv_(const char *, const int *, const int *, const doublecomplex *,
                  const doublecomplex *a, const int *, const doublecomplex *, const int *,
		  const doublecomplex *, doublecomplex *, const int *, int);

#else
extern int zgemm_(const char*, const char*, const int*, const int*, const int*,
                   const doublecomplex*,  const doublecomplex*,  const int*,  const doublecomplex*,
                   const int*,  const doublecomplex*, doublecomplex*, const int*);
extern int ztrsv_(char*, char*, char*, int*, doublecomplex*, int*,
                  doublecomplex*, int*);
extern int ztrsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const doublecomplex*, const doublecomplex*, const int*,
		  doublecomplex*, const int*);
extern void zgemv_(const char *, const int *, const int *, const doublecomplex *,
                  const doublecomplex *a, const int *, const doublecomplex *, const int *,
		  const doublecomplex *, doublecomplex *, const int *);
#endif

extern void zgeru_(const int*, const int*, const doublecomplex*,
                 const doublecomplex*, const int*, const doublecomplex*, const int*,
		 doublecomplex*, const int*);

extern int zscal_(const int *n, const doublecomplex *alpha, doublecomplex *dx, const int *incx);
extern int zaxpy_(const int *n, const doublecomplex *alpha, const doublecomplex *x,
	               const int *incx, doublecomplex *y, const int *incy);

/* SuperLU BLAS interface: zsuperlu_blas.c  */
extern int superlu_zgemm(const char *transa, const char *transb,
                  int m, int n, int k, doublecomplex alpha, doublecomplex *a,
                  int lda, doublecomplex *b, int ldb, doublecomplex beta, doublecomplex *c, int ldc);
extern int superlu_ztrsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag, const int m, const int n,
                  const doublecomplex alpha, const doublecomplex *a,
                  const int lda, doublecomplex *b, const int ldb);
extern int superlu_zger(const int m, const int n, const doublecomplex alpha,
                 const doublecomplex *x, const int incx, const doublecomplex *y,
                 const int incy, doublecomplex *a, const int lda);
extern int superlu_zscal(const int n, const doublecomplex alpha, doublecomplex *x, const int incx);
extern int superlu_zaxpy(const int n, const doublecomplex alpha,
    const doublecomplex *x, const int incx, doublecomplex *y, const int incy);
extern int superlu_zgemv(const char *trans, const int m,
                  const int n, const doublecomplex alpha, const doublecomplex *a,
                  const int lda, const doublecomplex *x, const int incx,
                  const doublecomplex beta, doublecomplex *y, const int incy);
extern int superlu_ztrsv(char *uplo, char *trans, char *diag,
                  int n, doublecomplex *a, int lda, doublecomplex *x, int incx);

#ifdef SLU_HAVE_LAPACK
extern void ztrtri_(char*, char*, int*, doublecomplex*, int*, int*);
#endif

/*==== For 3D code ====*/
extern int zcreate_matrix3d(SuperMatrix *A, int nrhs, doublecomplex **rhs,
                     int *ldb, doublecomplex **x, int *ldx,
                     FILE *fp, gridinfo3d_t *grid3d);
extern int zcreate_matrix_postfix3d(SuperMatrix *A, int nrhs, doublecomplex **rhs,
                           int *ldb, doublecomplex **x, int *ldx,
                           FILE *fp, char * postfix, gridinfo3d_t *grid3d);
extern int zcreate_block_diag_3d(SuperMatrix *A, int batchCount, int nrhs, doublecomplex **rhs,
				 int *ldb, doublecomplex **x, int *ldx,
				 FILE *fp, char * postfix, gridinfo3d_t *grid3d);
extern int zcreate_batch_systems(handle_t *SparseMatrix_handles, int batchCount,
				 int nrhs, doublecomplex **rhs, int *ldb, doublecomplex **x, int *ldx,
				 FILE *fp, char * postfix, gridinfo3d_t *grid3d);

/* Matrix distributed in NRformat_loc in 3D process grid. It converts
   it to a NRformat_loc distributed in 2D grid in grid-0 */
extern void zGatherNRformat_loc3d(fact_t Fact, NRformat_loc *A, doublecomplex *B,
				   int ldb, int nrhs, gridinfo3d_t *grid3d,
				   NRformat_loc3d **);
extern void zGatherNRformat_loc3d_allgrid(fact_t Fact, NRformat_loc *A, doublecomplex *B,
				   int ldb, int nrhs, gridinfo3d_t *grid3d,
				   NRformat_loc3d **);
extern int zScatter_B3d(NRformat_loc3d *A3d, gridinfo3d_t *grid3d);

extern void pzgssvx3d (superlu_dist_options_t *, SuperMatrix *,
		       zScalePermstruct_t *, doublecomplex B[], int ldb, int nrhs,
		       gridinfo3d_t *, zLUstruct_t *, zSOLVEstruct_t *,
		       double *berr, SuperLUStat_t *, int *info);
extern int_t pzgstrf3d(superlu_dist_options_t *, int m, int n, double anorm,
		       ztrf3Dpartition_t*, SCT_t *, zLUstruct_t *,
		       gridinfo3d_t *, SuperLUStat_t *, int *);
extern void zInit_HyP(superlu_dist_options_t *, HyP_t* HyP, zLocalLU_t *Llu, int_t mcb, int_t mrb);
extern void Free_HyP(HyP_t* HyP);
extern int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid);

    /* from scatter.h */
extern void
zblock_gemm_scatter( int_t lb, int_t j, Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info, doublecomplex *L_mat, int ldl,
                    doublecomplex *U_mat, int ldu,  doublecomplex *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int *indirect, int *indirect2,
                    int_t **Lrowind_bc_ptr, doublecomplex **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, doublecomplex **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *, SuperLUStat_t *
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );

#ifdef _OPENMP
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
                         int *indirect, int *indirect2,
                         int_t **Lrowind_bc_ptr, doublecomplex **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, doublecomplex **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );
#endif

extern int_t
zblock_gemm_scatterTopLeft( int_t lb,  int_t j, doublecomplex* bigV,
				 int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,
				 int* indirect, int* indirect2,
                                 HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                 SCT_t*SCT, SuperLUStat_t *
                               );
extern int_t
zblock_gemm_scatterTopRight( int_t lb,  int_t j, doublecomplex* bigV,
				  int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,
				  int* indirect, int* indirect2,
                                  HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                  SCT_t*SCT, SuperLUStat_t * );
extern int_t
zblock_gemm_scatterBottomLeft( int_t lb,  int_t j, doublecomplex* bigV,
				    int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt,
				    int* indirect, int* indirect2,
                                    HyP_t* HyP, zLUstruct_t *, gridinfo_t*,
                                    SCT_t*SCT, SuperLUStat_t * );
extern int_t
zblock_gemm_scatterBottomRight( int_t lb,  int_t j, doublecomplex* bigV,
				     int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,
				     int* indirect, int* indirect2,
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

    /* from pxdistribute3d.h */
extern void zbcastPermutedSparseA(SuperMatrix *A,
                          zScalePermstruct_t *ScalePermstruct,
                          Glu_freeable_t *Glu_freeable,
                          zLUstruct_t *LUstruct, gridinfo3d_t *grid3d);

extern void znewTrfPartitionInit(int_t nsupers,  zLUstruct_t *LUstruct, gridinfo3d_t *grid3d);


    /* from xtrf3Dpartition.h */
extern ztrf3Dpartition_t* zinitTrf3Dpartition(int_t nsupers,
					     superlu_dist_options_t *options,
					     zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern ztrf3Dpartition_t* zinitTrf3Dpartition_allgrid(int_t n,
					     superlu_dist_options_t *options,
					     zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern ztrf3Dpartition_t* zinitTrf3DpartitionLUstructgrid0(int_t n,
					     superlu_dist_options_t *options,
					     zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern void zDestroy_trf3Dpartition(ztrf3Dpartition_t *trf3Dpartition);

extern void z3D_printMemUse(ztrf3Dpartition_t*  trf3Dpartition,
			    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d);

//extern int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
//		       superlu_dist_options_t *options, zLocalLU_t *Llu,
//		       int_t* xsup, int_t num_look_aheads, int_t nsupers,
//		       int_t * iperm_c_supno);

extern void zinit3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t**  sForests, zLUstruct_t* LUstruct,
				  gridinfo3d_t* grid3d);

extern int_t zgatherAllFactoredLUFr(int_t* myZeroTrIdxs, sForest_t* sForests,
				   zLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
				   SCT_t* SCT );

    /* The following are from pdgstrf2.h */
extern int_t zLpanelUpdate(int_t off0, int_t nsupc, doublecomplex* ublk_ptr,
			  int_t ld_ujrow, doublecomplex* lusup, int_t nsupr, SCT_t*);
extern void zgstrf2(int_t k, doublecomplex* diagBlk, int_t LDA, doublecomplex* BlockUfactor, int_t LDU,
            double thresh, int_t* xsup, superlu_dist_options_t *options,
            SuperLUStat_t *stat, int *info);
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
                             int_t knsupc, int nsupr, doublecomplex* lusup,
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
extern int_t zp3dScatter(int_t n, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t zscatter3dLPanels(int_t nsupers,
                       zLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t zscatter3dUPanels(int_t nsupers,
                       zLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t zcollect3dLpanels(int_t layer, int_t nsupers, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zcollect3dUpanels(int_t layer, int_t nsupers, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t zp3dCollect(int_t layer, int_t n, zLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
/*zero out LU non zero entries*/
extern int_t zzeroSetLU(int_t nnodes, int_t* nodeList , zLUstruct_t *, gridinfo3d_t*);
extern int zAllocGlu_3d(int_t n, int_t nsupers, zLUstruct_t *);
extern int zDeAllocLlu_3d(int_t n, zLUstruct_t *, gridinfo3d_t*);
extern int zDeAllocGlu_3d(zLUstruct_t *);

/* Reduces L and U panels of nodes in the List nodeList (size=nnnodes)
receiver[L(nodelist)] =sender[L(nodelist)] +receiver[L(nodelist)]
receiver[U(nodelist)] =sender[U(nodelist)] +receiver[U(nodelist)]
*/
int_t zreduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        doublecomplex* Lval_buf, doublecomplex* Uval_buf,
                        zLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
/*reduces all nodelists required in a level*/
extern int zreduceAllAncestors3d(int_t ilvl, int_t* myNodeCount,
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
int_t zgatherAllFactoredLU(ztrf3Dpartition_t*  trf3Dpartition, zLUstruct_t* LUstruct,
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
			     int **ToSendR, int_t *xsup, int );
extern int_t zBcast_LPanel(int_t k, int_t k0, int_t* lsub, doublecomplex* lusup,
			   gridinfo_t *, int* msgcnt, int **ToSendR,
			   int_t *xsup , SCT_t*, int);
extern int_t zIBcast_UPanel(int_t k, int_t k0, int_t* usub, doublecomplex* uval,
			    gridinfo_t *, int* msgcnt, MPI_Request *,
			    int *ToSendD, int );
extern int_t zBcast_UPanel(int_t k, int_t k0, int_t* usub, doublecomplex* uval,
			   gridinfo_t *, int* msgcnt, int *ToSendD, SCT_t*, int);
extern int_t zIrecv_LPanel (int_t k, int_t k0,  int_t* Lsub_buf,
			    doublecomplex* Lval_buf, gridinfo_t *,
			    MPI_Request *, zLocalLU_t *, int);
extern int_t zIrecv_UPanel(int_t k, int_t k0, int_t* Usub_buf, doublecomplex*,
			   zLocalLU_t *, gridinfo_t*, MPI_Request *, int);
extern int_t zWait_URecv(MPI_Request *, int* msgcnt, SCT_t *);
extern int_t zWait_LRecv(MPI_Request*, int* msgcnt, int* msgcntsU,
			 gridinfo_t *, SCT_t*);
extern int_t zISend_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t zRecv_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			      int_t src, gridinfo_t *, SCT_t*, int);
extern int_t zPackLBlock(int_t k, doublecomplex* Dest, Glu_persist_t *,
			 gridinfo_t *, zLocalLU_t *);
extern int_t zISend_LDiagBlock(int_t k0, doublecomplex *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t zIRecv_UDiagBlock(int_t k0, doublecomplex *ublk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t *,
			       SCT_t*, int);
extern int_t zIRecv_LDiagBlock(int_t k0, doublecomplex *L_blk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t*, SCT_t*, int);
extern int_t zUDiagBlockRecvWait( int_t k,  int* IrecvPlcd_D, int* factored_L,
				  MPI_Request *, gridinfo_t *, zLUstruct_t *, SCT_t *);

#if (MPI_VERSION>2)
extern int_t zIBcast_UDiagBlock(int_t k, doublecomplex *ublk_ptr, int_t size,
				MPI_Request *, gridinfo_t *);
extern int_t zIBcast_LDiagBlock(int_t k, doublecomplex *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *);
#endif

    /* from trfCommWrapper.h */
extern int_t zDiagFactIBCast(int_t k,  int_t k0,
			     doublecomplex *BlockUFactor, doublecomplex *BlockLFactor,
			     int* IrecvPlcd_D, MPI_Request *, MPI_Request *,
			     MPI_Request *, MPI_Request *, gridinfo_t *,
			     superlu_dist_options_t *, double thresh,
			     zLUstruct_t *LUstruct, SuperLUStat_t *, int *info,
			     SCT_t *, int tag_ub);
extern int_t zUPanelTrSolve( int_t k, doublecomplex* BlockLFactor, doublecomplex* bigV,
			     int_t ldt, Ublock_info_t*, gridinfo_t *,
			     zLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t zLPanelUpdate(int_t k,  int* IrecvPlcd_D, int* factored_L,
			   MPI_Request *, doublecomplex* BlockUFactor, gridinfo_t *,
			   zLUstruct_t *, SCT_t *);
extern int_t zUPanelUpdate(int_t k, int* factored_U, MPI_Request *,
			   doublecomplex* BlockLFactor, doublecomplex* bigV,
			   int_t ldt, Ublock_info_t*, gridinfo_t *,
			   zLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t zIBcastRecvLPanel(int_t k, int_t k0, int* msgcnt,
			       MPI_Request *, MPI_Request *,
			       int_t* Lsub_buf, doublecomplex* Lval_buf,
			      int * factored, gridinfo_t *, zLUstruct_t *,
			      SCT_t *, int tag_ub);
extern int_t zIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt, MPI_Request *,
			       MPI_Request *, int_t* Usub_buf, doublecomplex* Uval_buf,
			       gridinfo_t *, zLUstruct_t *, SCT_t *, int tag_ub);
extern int_t zWaitL(int_t k, int* msgcnt, int* msgcntU, MPI_Request *,
		    MPI_Request *, gridinfo_t *, zLUstruct_t *, SCT_t *);
extern int_t zWaitU(int_t k, int* msgcnt, MPI_Request *, MPI_Request *,
		   gridinfo_t *, zLUstruct_t *, SCT_t *);
extern int_t zLPanelTrSolve(int_t k, int* factored_L, doublecomplex* BlockUFactor,
			    gridinfo_t *, zLUstruct_t *);

    /* from trfAux.h */
extern int getNsupers(int, Glu_persist_t *);
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
				      factNodelists_t*, zscuBufs_t*,
				      zLUValSubBuf_t* LUvsb, gridinfo_t *,
				      zLUstruct_t *, HyP_t*);
extern doublecomplex* zgetBigV(int_t, int_t);
extern doublecomplex* zgetBigU(superlu_dist_options_t *,
                           int_t, gridinfo_t *, zLUstruct_t *);
// permutation from superLU default

    /* from treeFactorization.h */
extern int_t zLluBufInit(zLUValSubBuf_t*, zLUstruct_t *);
extern int_t zinitScuBufs(superlu_dist_options_t *,
                          int_t ldt, int_t num_threads, int_t nsupers,
			  zscuBufs_t*, zLUstruct_t*, gridinfo_t *);
extern int zfreeScuBufs(zscuBufs_t* scuBufs);

#if 0 // NOT CALLED
// the generic tree factoring code
extern int_t treeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    zscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    zdiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);
#endif

extern int_t zsparseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo,
    commRequests_t *comReqs,    // lists of communication requests
    zscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    zdiagFactBufs_t *dFBuf,
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
    zscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    zdiagFactBufs_t *dFBuf,
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
    zscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    zLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    zdiagFactBufs_t **dFBufs,         // size maxEtree level
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
extern zdiagFactBufs_t** zinitDiagFactBufsArr(int mxLeafNode, int ldt, gridinfo_t* grid);
extern zdiagFactBufs_t** zinitDiagFactBufsArrMod(int mxLeafNode, int* ldts, gridinfo_t* grid);
extern int zfreeDiagFactBufsArr(int mxLeafNode, zdiagFactBufs_t** dFBufs);
extern int zinitDiagFactBufs(int ldt, zdiagFactBufs_t* dFBuf);
extern int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
			    gridinfo_t *grid, SCT_t *SCT);
extern int_t checkRecvLDiag(int_t k, commRequests_t *comReqs, gridinfo_t *, SCT_t *);


extern int pzflatten_LDATA(superlu_dist_options_t *options, int_t n, zLUstruct_t * LUstruct,
                           gridinfo_t *grid, SuperLUStat_t * stat);
extern void pzconvert_flatten_skyline2UROWDATA(superlu_dist_options_t *, gridinfo_t *,
	                 zLUstruct_t *, SuperLUStat_t *, int n);
extern void pzconvertUROWDATA2skyline(superlu_dist_options_t *, gridinfo_t *,
       	    		zLUstruct_t *, SuperLUStat_t *, int n);

extern int_t
zReDistribute_A(SuperMatrix *A, zScalePermstruct_t *ScalePermstruct,
                Glu_freeable_t *Glu_freeable, int_t *xsup, int_t *supno,
                gridinfo_t *grid, int_t *colptr[], int_t *rowind[],
                doublecomplex *a[]);
extern float
pzdistribute3d_Yang(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     zScalePermstruct_t *ScalePermstruct,
	     Glu_freeable_t *Glu_freeable, zLUstruct_t *LUstruct,
	     gridinfo3d_t *grid3d);


#if 0 // NOT CALLED
/* from ancFactorization.h (not called) */
extern int_t ancestorFactor(
    int_t ilvl,             // level of factorization
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    zscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    zLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    zdiagFactBufs_t **dFBufs,         // size maxEtree level
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
#endif

    /* Batch interface */
extern int pzgssvx3d_csc_batch(
		superlu_dist_options_t *, int batchCount, int m, int n,	int nnz,
		int nrhs, handle_t *, doublecomplex **RHSptr, int *ldRHS,
		double **ReqPtr, double **CeqPtr,
		int **RpivPtr, int **CpivPtr, DiagScale_t *DiagScale,
		handle_t *F, doublecomplex **Xptr, int *ldX, double **Berrs,
		gridinfo3d_t *grid3d, SuperLUStat_t *stat, int *info
		//DeviceContext context /* device context including queues, events, dependencies */
		);
extern int zequil_batch(
    superlu_dist_options_t *, int batchCount, int m, int n, handle_t *,
    double **ReqPtr, double **CeqPtr, DiagScale_t *
    //    DeviceContext context /* device context including queues, events, dependencies */
    );
extern int zpivot_batch(
    superlu_dist_options_t *, int batchCount, int m, int n, handle_t *,
    double **ReqPtr, double **CeqPtr, DiagScale_t *, int **RpivPtr
    //    DeviceContext context /* device context including queues, events, dependencies */
    );

/*== end 3D prototypes ===================*/

extern doublecomplex *zready_x;
extern doublecomplex *zready_lsum;

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

