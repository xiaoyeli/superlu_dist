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

#ifndef __SUPERLU_DDEFS /* allow multiple inclusions */
#define __SUPERLU_DDEFS

/*
 * File name:	superlu_ddefs.h
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
} dScalePermstruct_t;

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
    	                         free'd in trs_compute_communication_structure routinies */
    int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk]) */
    long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)              */
    long int Lrowind_bc_cnt;

    double **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc);
    	                         free'd in trs_compute_communication_structure routinies */
    double *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])  */   
    long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                */    
    long int Lnzval_bc_cnt;
    
    double **Linv_bc_ptr;    /* size ceil(NSUPERS/Pc);
    	                         free'd in trs_compute_communication_structure routinies */
    double *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])  */   
    long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)              */   
    long int Linv_bc_cnt;
    
    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc);
                                     pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr;
    	                             free'd in trs_compute_communication_structure routinies */
				     
    int_t *Lindval_loc_bc_dat;  /* size: sum of sizes of Lindval_loc_bc_ptr[lk]) */   
    long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)  */   
    long int Lindval_loc_bc_cnt;
    
    /* for new U format -> */
    int_t   **Ucolind_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
    int_t *Ucolind_bc_dat;  /* size: sum of sizes of Ucolind_bc_ptr[lk])    */   
    int64_t *Ucolind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */     
    int64_t Ucolind_bc_cnt;

    double **Unzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    double *Unzval_bc_dat;  /* size: sum of sizes of Unzval_bc_ptr[lk])  */   
    int64_t *Unzval_bc_offset;  /* size ceil(NSUPERS/Pc)                */    
    int64_t Unzval_bc_cnt;

    int_t   **Uindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Ucolind_bc_ptr and Unzval_bc_ptr */
    int_t *Uindval_loc_bc_dat; /* size: sum of sizes of Uindval_loc_bc_ptr[lk]) */   
    int64_t *Uindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)   */
    int64_t Uindval_loc_bc_cnt;  
    /* end for new U format <- */
    
    int_t   *Unnz; /* number of nonzeros per block column in U*/
    int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */
    double **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    double *Uinv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */
    long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
    long int Uinv_bc_cnt;

    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */
    long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Ufstnz_br_cnt;

    double  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                  */
    double  *Unzval_br_dat;   /* size sum of sizes of Unzval_br_ptr[lk]) */
    long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
    long int Unzval_br_cnt;

        /*-- Data structures used for broadcast and reduction trees. --*/
    C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *LRtree_ptr;       /* size ceil(NSUPERS/Pr)                */
    C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    C_Tree  *URtree_ptr;       /* size ceil(NSUPERS/Pr)			*/
#if 0
    int_t   *Lsub_buf;        /* Buffer for the remote subscripts of L */
    double  *Lval_buf;        /* Buffer for the remote nonzeros of L   */
    int_t   *Usub_buf;        /* Buffer for the remote subscripts of U */
    double  *Uval_buf;        /* Buffer for the remote nonzeros of U   */
#endif
    int_t   *Lsub_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote subscripts of L*/
    double  *Lval_buf_2[MAX_LOOKAHEADS];   /* Buffers for the remote nonzeros of L  */
    int_t   *Usub_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote subscripts of U */
    double  *Uval_buf_2[MAX_LOOKAHEADS];   /* Buffer for the remote nonzeros of U   */
    double  *ujrow;           /* used in panel factorization.          */
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
    double *d_Lnzval_bc_dat;     
    long int *d_Lnzval_bc_offset;
    int_t *d_Ucolind_bc_dat;     
    int64_t *d_Ucolind_bc_offset;      
    double *d_Unzval_bc_dat;     
    long int *d_Unzval_bc_offset;        
    
    double *d_Linv_bc_dat ;     
    double *d_Uinv_bc_dat ;     
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
    //    double *d_Unzval_br_dat;   
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

} dLocalLU_t;


typedef struct
{
    int_t * Lsub_buf ;
    double * Lval_buf ;
    int_t * Usub_buf ;
    double * Uval_buf ;
} dLUValSubBuf_t;


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
    int* supernodeMask;
    dLUValSubBuf_t  *LUvsb;
    int maxLvl; // YL: store this to avoid the use of grid3d
    
    /* Sherry added the following 3 for variable size batch. 2/17/23 */
    int mxLeafNode; /* number of leaf nodes. */
    int *diagDims;  /* dimensions of the diagonal blocks at any level of the tree */
    int *gemmCsizes; /* sizes of the C matrices at any level of the tree. */
} dtrf3Dpartition_t;


typedef struct {
    int_t *etree;
    Glu_persist_t *Glu_persist;
    dLocalLU_t *Llu;
    dtrf3Dpartition_t *trf3Dpartition;
    char dt;
} dLUstruct_t;


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
} pdgsmv_comm_t;

/*-- Data structure holding the information for the solution phase --*/
typedef struct {
    int_t *row_to_proc;
    int_t *inv_perm_c;
    int_t num_diag_procs, *diag_procs, *diag_len;
    pdgsmv_comm_t *gsmv_comm; /* communication metadata for SpMV,
         	       		      required by IterRefine.          */
    pxgstrs_comm_t *gstrs_comm;  /* communication metadata for SpTRSV. */
    int_t *A_colind_gsmv; /* After pdgsmv_init(), the global column
                             indices of A are translated into the relative
                             positions in the gathered x-vector.
                             This is re-used in repeated calls to pdgsmv() */
    int_t *xrow_to_proc; /* used by PDSLin */
    NRformat_loc3d* A3d; /* Point to 3D {A, B} gathered on 2D layer 0.
                            This needs to be peresistent between
			    3D factorization and solve.  */
    #ifdef GPU_ACC
    double *d_lsum, *d_lsum_save;      /* used for device lsum*/
    double *d_x;         /* used for device solution vector*/
    int  *d_fmod_save, *d_fmod;         /* used for device fmod vector*/
    int  *d_bmod_save, *d_bmod;         /* used for device bmod vector*/
    #endif         
} dSOLVEstruct_t;



/*==== For 3D code ====*/

// new structures for pdgstrf_4_8

#if 0  // Sherry: moved to superlu_defs.h
typedef struct
{
    int_t nub;
    int_t klst;
    int_t ldu;
    int_t* usub;
    double* uval;
} uPanelInfo_t;

typedef struct
{
    int_t *lsub;
    double *lusup;
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
    double *lookAhead_L_buff, *Remain_L_buff;
    int_t lookAheadBlk;  /* number of blocks in look-ahead window */
    int_t RemainBlk ;    /* number of blocks outside look-ahead window */
    int_t  num_look_aheads, nsupers;
    int_t ldu, ldu_Phi;
    int_t num_u_blks, num_u_blks_Phi;

    int_t jj_cpu;
    double *bigU_Phi;
    double *bigU_host;
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
    double *bigU;
    double *bigV;
} dscuBufs_t;

typedef struct
{
    double* BlockLFactor;
    double* BlockUFactor;
} ddiagFactBufs_t;


typedef struct xT_struct
{
	double* xT;
	int_t ldaspaT;
	int_t* ilsumT;
} xT_struct;

typedef struct lsumBmod_buff_t
{
    double * tX;    // buffer for reordered X
    double * tU;    // buffer for packedU
    int_t *indCols; //
}lsumBmod_buff_t;

typedef enum trtype_t {UPPER_TRI, LOWER_TRI} trtype_t;

/*=====================*/

/***********************************************************************
 * Function prototypes
 ***********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


/* Supernodal LU factor related */
extern void
dCreate_CompCol_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, double *,
			    int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_CompRowLoc_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, int_t,
			       int_t, double *, int_t *, int_t *,
			       Stype_t, Dtype_t, Mtype_t);
extern void
dCompRow_to_CompCol_dist(int_t, int_t, int_t, double *, int_t *, int_t *,
                         double **, int_t **, int_t **);
extern int
pdCompRow_loc_to_CompCol_global(int_t, SuperMatrix *, gridinfo_t *,
	 		        SuperMatrix *);
extern void
dCopy_CompCol_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void
dCreate_Dense_Matrix_dist(SuperMatrix *, int_t, int_t, double *, int_t,
			  Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_SuperNode_Matrix_dist(SuperMatrix *, int_t, int_t, int_t, double *,
			      int_t *, int_t *, int_t *, int_t *, int_t *,
			      Stype_t, Dtype_t, Mtype_t);
extern void
dCopy_Dense_Matrix_dist(int_t, int_t, double *, int_t,
                        double *, int_t);

extern void    dallocateA_dist (int_t, int_t, double **, int_t **, int_t **);
extern void    dGenXtrue_dist (int_t, int_t, double *, int_t);
extern void    dFillRHS_dist (char *, int_t, double *, int_t,
                              SuperMatrix *, double *, int_t);
extern int     dcreate_matrix(SuperMatrix *, int, double **, int *,
			      double **, int *, FILE *, gridinfo_t *);
extern int     dcreate_matrix_rb(SuperMatrix *, int, double **, int *,
			      double **, int *, FILE *, gridinfo_t *);
extern int     dcreate_matrix_dat(SuperMatrix *, int, double **, int *,
			      double **, int *, FILE *, gridinfo_t *);
extern int dcreate_matrix_postfix(SuperMatrix *, int, double **, int *,
				  double **, int *, FILE *, char *, gridinfo_t *);

extern void   dScalePermstructInit(const int_t, const int_t,
                                      dScalePermstruct_t *);
extern void   dScalePermstructFree(dScalePermstruct_t *);

/* Driver related */
extern void    dgsequ_dist (SuperMatrix *, double *, double *, double *,
			    double *, double *, int_t *);
extern double  dlangs_dist (char *, SuperMatrix *);
extern void    dlaqgs_dist (SuperMatrix *, double *, double *, double,
			    double, double, char *);
extern void    pdgsequ (SuperMatrix *, double *, double *, double *,
			double *, double *, int_t *, gridinfo_t *);
extern double  pdlangs (char *, SuperMatrix *, gridinfo_t *);
extern void    pdlaqgs (SuperMatrix *, double *, double *, double,
			double, double, char *);
extern int     pdPermute_Dense_Matrix(int_t, int_t, int_t [], int_t[],
				      double [], int, double [], int, int,
				      gridinfo_t *);

extern int     sp_dtrsv_dist (char *, char *, char *, SuperMatrix *,
			      SuperMatrix *, double *, int *);
extern int     sp_dgemv_dist (char *, double, SuperMatrix *, double *,
			      int, double, double *, int);
extern int     sp_dgemm_dist (char *, int, double, SuperMatrix *,
                        double *, int, double, double *, int);

extern float ddistribute(superlu_dist_options_t *,
                         int_t, SuperMatrix *, Glu_freeable_t *,
			 dLUstruct_t *, gridinfo_t *);
extern void  pdgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *,
			      dScalePermstruct_t *, double *,
			      int, int, gridinfo_t *, dLUstruct_t *, double *,
			      SuperLUStat_t *, int *);
extern float pddistribute(superlu_dist_options_t *, int_t, SuperMatrix *,
			 dScalePermstruct_t *, Glu_freeable_t *,
			 dLUstruct_t *, gridinfo_t *);
extern float pddistribute_allgrid(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     dScalePermstruct_t *ScalePermstruct,
	     Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
	     gridinfo_t *grid, int* supernodeMask);

extern float pddistribute_allgrid_index_only(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     dScalePermstruct_t *ScalePermstruct,
	     Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
	     gridinfo_t *grid, int* supernodeMask);

extern void  pdgssvx(superlu_dist_options_t *, SuperMatrix *,
		     dScalePermstruct_t *, double *,
		     int, int, gridinfo_t *, dLUstruct_t *,
		     dSOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void  pdCompute_Diag_Inv(int_t, dLUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  dSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, dLUstruct_t *, gridinfo_t *, dSOLVEstruct_t *);
extern void dSolveFinalize(superlu_dist_options_t *, dSOLVEstruct_t *);
extern void dDestroy_A3d_gathered_on_2d(dSOLVEstruct_t *, gridinfo3d_t *);
extern int_t pdgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, dSOLVEstruct_t *);
extern int_t pdgstrs_init_device_lsum_x(superlu_dist_options_t *, int_t , int_t , int_t , gridinfo_t *,
	     dLUstruct_t *, dSOLVEstruct_t *, int*);    
extern int_t pdgstrs_delete_device_lsum_x(dSOLVEstruct_t *);                           
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  dldperm_dist(int, int, int_t, int_t [], int_t [],
		    double [], int_t *, double [], double []);
extern int  dstatic_schedule(superlu_dist_options_t *, int, int,
		            dLUstruct_t *, gridinfo_t *, SuperLUStat_t *,
			    int_t *, int_t *, int *);
extern void dLUstructInit(const int_t, dLUstruct_t *);
extern void dLUstructFree(dLUstruct_t *);
extern void dDestroy_LU(int_t, gridinfo_t *, dLUstruct_t *);
extern void dDestroy_Tree(int_t, gridinfo_t *, dLUstruct_t *);
extern void dscatter_l (int ib, int ljb, int nsupc, int_t iukp, int_t* xsup,
			int klst, int nbrow, int_t lptr, int temp_nbrow,
			int_t* usub, int_t* lsub, double *tempv,
			int* indirect_thread, int* indirect2,
			int_t ** Lrowind_bc_ptr, double **Lnzval_bc_ptr,
			gridinfo_t * grid);
extern void dscatter_u (int ib, int jb, int nsupc, int_t iukp, int_t * xsup,
                        int klst, int nbrow, int_t lptr, int temp_nbrow,
                        int_t* lsub, int_t* usub, double* tempv,
                        int_t ** Ufstnz_br_ptr, double **Unzval_br_ptr,
                        gridinfo_t * grid);
extern int_t pdgstrf(superlu_dist_options_t *, int, int, double anorm,
		    dLUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);

/* #define GPU_PROF
#define IPM_PROF */

/* Solve related */
extern void pdgstrs_Bglobal(superlu_dist_options_t *,
                             int_t, dLUstruct_t *, gridinfo_t *,
			     double *, int_t, int, SuperLUStat_t *, int *);
extern void pdgstrs(superlu_dist_options_t *, int_t,
                    dLUstruct_t *, dScalePermstruct_t *, gridinfo_t *,
		    double *, int_t, int_t, int_t, int, dSOLVEstruct_t *,
		    SuperLUStat_t *, int *);
extern void pdgstrf2_trsm(superlu_dist_options_t * options, int_t k0, int_t k,
			  double thresh, Glu_persist_t *, gridinfo_t *,
			  dLocalLU_t *, MPI_Request *, int tag_ub,
			  SuperLUStat_t *, int *info);
extern void pdgstrs2_omp(int_t k0, int_t k, Glu_persist_t *, gridinfo_t *,
			 dLocalLU_t *, Ublock_info_t *, SuperLUStat_t *);
extern int_t pdReDistribute_B_to_X(double *B, int_t m_loc, int nrhs, int_t ldb,
				   int_t fst_row, int_t *ilsum, double *x,
				   dScalePermstruct_t *, Glu_persist_t *,
				   gridinfo_t *, dSOLVEstruct_t *);
extern void dlsum_fmod(double *, double *, double *, double *,
		       int, int, int_t , int *fmod, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, dLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);
extern void dlsum_bmod(double *, double *, double *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, dLocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void dlsum_fmod_inv(double *, double *, double *, double *,
		       int, int_t , int *fmod,
		       int_t *, gridinfo_t *, dLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void dlsum_fmod_inv_master(double *, double *, double *, double *,
		       int, int, int_t , int *fmod, int_t,
		       int_t *, gridinfo_t *, dLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void dlsum_bmod_inv(double *, double *, double *, double *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, dLocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void dlsum_bmod_inv_master(double *, double *, double *, double *,
                       int, int_t, int *bmod, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, dLocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int, int);

extern void dComputeLevelsets(int , int_t , gridinfo_t *,
		  Glu_persist_t *, dLocalLU_t *, int_t *);               
			   
#ifdef GPU_ACC               
extern void pdconvertU(superlu_dist_options_t *, gridinfo_t *, dLUstruct_t *, SuperLUStat_t *, int);
extern void dlsum_fmod_inv_gpu_wrap(int_t, int_t, int_t, int_t, double *,double *,int,int, int_t , int_t *, C_Tree  *, C_Tree  *, int_t *, int_t *,long int *, double *, long int *, double *, long int *, int_t *, long int *, int_t *, int *, gridinfo_t *,
                                    int_t, uint64_t*, uint64_t*, double*, double*, int*, int*, int*,
                                    int*, int*, int*, int*, int*, int*,int*,
                                    int*, int*, int*, int*, int*, int*,
                                    int*, int*, int);
extern void dlsum_bmod_inv_gpu_wrap(superlu_dist_options_t *, int_t, int_t, int_t, int_t, double *, double *,int,int, int_t , int *, C_Tree  *, C_Tree  *, int_t *, int_t *, int64_t *, double *, int64_t *, double  *, int64_t *, int_t *, int64_t *, int_t *,gridinfo_t *,
                                    int_t, uint64_t*, uint64_t*, double*, double*,
                                    int*, int*, int*, int*,
                                    int*, int*, int*, int*, int*,
                                    int*, int*, int*, int*, int*, int*,
                                    int*, int*, int*, int); //int*); //int*, double*);
extern void pdReDistribute_B_to_X_gpu_wrap(double *, int_t, int_t, int, int_t, int_t, int_t *, int_t, int_t, int_t, double *, dScalePermstruct_t *, dSOLVEstruct_t *, Glu_persist_t *, gridinfo_t *);
extern void pdReDistribute_X_to_B_gpu_wrap(double *, int_t, int_t, int, int_t, int_t, int_t *, int_t, int_t, int_t, double *, dScalePermstruct_t *, dSOLVEstruct_t *, Glu_persist_t *, gridinfo_t *);

#endif

extern void pdgsrfs(superlu_dist_options_t *, int_t,
                    SuperMatrix *, double, dLUstruct_t *,
		    dScalePermstruct_t *, gridinfo_t *,
		    double [], int_t, double [], int_t, int,
		    dSOLVEstruct_t *, double *, SuperLUStat_t *, int *);


extern void pdgsrfs3d(superlu_dist_options_t *options, int_t n,
            SuperMatrix *A, double anorm, dLUstruct_t *LUstruct,
	        dScalePermstruct_t *ScalePermstruct, gridinfo3d_t *grid3d,
	        dtrf3Dpartition_t*  trf3Dpartition, double *B, int_t ldb, double *X, int_t ldx, int nrhs,
	        dSOLVEstruct_t *SOLVEstruct, double *berr, SuperLUStat_t *stat, int *info);

extern void pdgsrfs_ABXglobal(superlu_dist_options_t *, int_t,
                  SuperMatrix *, double, dLUstruct_t *,
		  gridinfo_t *, double *, int_t, double *, int_t,
		  int, double *, SuperLUStat_t *, int *);
extern int   pdgsmv_AXglobal_setup(SuperMatrix *, Glu_persist_t *,
				   gridinfo_t *, int_t *, int_t *[],
				   double *[], int_t *[], int_t []);
extern int  pdgsmv_AXglobal(int_t, int_t [], double [], int_t [],
	                       double [], double []);
extern int  pdgsmv_AXglobal_abs(int_t, int_t [], double [], int_t [],
				 double [], double []);
extern void pdgsmv_init(SuperMatrix *, int_t *, gridinfo_t *,
			pdgsmv_comm_t *);
extern void pdgsmv(int_t, SuperMatrix *, gridinfo_t *, pdgsmv_comm_t *,
		   double x[], double ax[]);
extern void pdgsmv_finalize(pdgsmv_comm_t *);


extern int_t initLsumBmod_buff(int_t ns, int nrhs, lsumBmod_buff_t* lbmod_buf);
extern int_t leafForestBackSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  dLUstruct_t * LUstruct,
                            dScalePermstruct_t * ScalePermstruct,
                            dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            double * x, double * lsum, double * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, lsumBmod_buff_t* lbmod_buf,
                            dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t nonLeafForestBackSolve3d( int_t treeId,  dLUstruct_t * LUstruct,
                                dScalePermstruct_t * ScalePermstruct,
                                dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                 double * x, double * lsum, xT_struct *xT_s,double * recvbuf,
                                MPI_Request * send_req,
                                int nrhs, lsumBmod_buff_t* lbmod_buf,
                                dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t dlasum_bmod_Tree(int_t  pTree, int_t cTree, double *lsum, double *x,
                       xT_struct *xT_s,
                       int    nrhs, lsumBmod_buff_t* lbmod_buf,
                       dLUstruct_t * LUstruct,
                       dtrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat);
extern int_t lsumForestBsolve(int_t k, int_t treeId,
                       double *lsum, double *x,  xT_struct *xT_s,int    nrhs, lsumBmod_buff_t* lbmod_buf,
                       dLUstruct_t * LUstruct,
                       dtrf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat);

extern int_t  bCastXk2Pck  (int_t k, xT_struct *xT_s, int nrhs,
                     dLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer);

extern int_t  lsumReducePrK (int_t k, double*x, double* lsum, double* recvbuf, int nrhs,
                      dLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer);

extern int* getBmod3d(int_t treeId, int_t nlb, sForest_t* sforest, dLUstruct_t * LUstruct, dtrf3Dpartition_t*  trf3Dpartition, gridinfo_t * grid);

extern int* getBmod3d_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, dLUstruct_t * LUstruct, gridinfo_t * grid);

extern int* getBrecvTree(int_t nlb, sForest_t* sforest,  int* bmod, gridinfo_t * grid);

extern int* getBrecvTree_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, int* bmod, gridinfo_t * grid);


extern int getNrootUsolveTree(int_t* nbrecvmod, sForest_t* sforest, int* brecv,
	int* bmod, gridinfo_t * grid);

extern int getNbrecvX(sForest_t* sforest, int_t* Urbs, gridinfo_t * grid);
extern int getNbrecvX_newsolve(int_t nsupers, int* supernodeMask, int_t* Urbs, Ucb_indptr_t **Ucb_indptr, gridinfo_t * grid);
extern int getNrootUsolveTree_newsolve(int_t* nbrecvmod, int_t nsupers, int* supernodeMask, int* brecv, int* bmod, gridinfo_t * grid);


extern int_t nonLeafForestForwardSolve3d( int_t treeId,  dLUstruct_t * LUstruct,
                                   dScalePermstruct_t * ScalePermstruct,
                                   dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                   double * x, double * lsum,
                                   xT_struct *xT_s,
                                   double * recvbuf, double* rtemp,
                                   MPI_Request * send_req,
                                   int nrhs,
                                   dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);
extern int_t leafForestForwardSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  dLUstruct_t * LUstruct,
                               dScalePermstruct_t * ScalePermstruct,
                               dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               double * x, double * lsum, double * recvbuf, double* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);
extern int* getfmodLeaf(int_t nlb, dLUstruct_t * LUstruct);
extern int* getfmod_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, dLUstruct_t * LUstruct, gridinfo_t * grid);
extern int getNfrecvxLeaf(sForest_t* sforest, dLUstruct_t * LUstruct, gridinfo_t * grid);
extern int getNfrecvx_newsolve(int_t nsupers, int* supernodeMask, dLUstruct_t * LUstruct, gridinfo_t * grid);
extern int_t getNfrecvmodLeaf(int* nleaf, sForest_t* sforest, int* frecv, int* fmod, gridinfo_t * grid);
extern int_t getNfrecvmod_newsolve(int* nleaf, int_t nsupers, int* supernodeMask, int* frecv, int* fmod, gridinfo_t * grid);
extern int* getfrecvLeaf( sForest_t* sforest, int_t nlb, int* fmod,
  dLUstruct_t * LUstruct, gridinfo_t * grid);
extern int* getfrecv_newsolve(int_t nsupers, int* supernodeMask, int_t nlb, int* fmod,
                     dLUstruct_t * LUstruct, gridinfo_t * grid);
extern int_t trs_compute_communication_structure(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           int* supernodeMask, gridinfo_t *grid, SuperLUStat_t * stat);
extern int_t reduceSolvedX_newsolve(int_t treeId, int_t sender, int_t receiver, double* x, int nrhs,
                      dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, double* recvbuf, xtrsTimer_t *xtrsTimer);

extern void dlsum_fmod_leaf (
  int_t treeId,
  dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,    /* Sum of local modifications.                        */
    double *x,       /* X array (local)                                    */
    double *xk,      /* X[k].                                              */
    double *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat, xtrsTimer_t *xtrsTimer);

extern void dlsum_fmod_leaf_newsolve (
    dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,    /* Sum of local modifications.                        */
    double *x,       /* X array (local)                                    */
    double *xk,      /* X[k].                                              */
    double *rtemp,   /* Result of full matrix-vector multiply.             */
    int   nrhs,      /* Number of right-hand sides.                        */
    int   knsupc,    /* Size of supernode k.                               */
    int_t k,         /* The k-th component of X.                           */
    int *fmod,     /* Modification count for L-solve.                    */
    int_t nlb,       /* Number of L blocks.                                */
    int_t lptr,      /* Starting position in lsub[*].                      */
    int_t luptr,     /* Starting position in lusup[*].                     */
    int_t *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer);


extern void dlsum_bmod_GG
(
    double *lsum,        /* Sum of local modifications.                    */
    double *x,           /* X array (local).                               */
    double *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    lsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat, xtrsTimer_t *xtrsTimer);

extern void dlsum_bmod_GG_newsolve (
    dtrf3Dpartition_t*  trf3Dpartition,
    double *lsum,        /* Sum of local modifications.                    */
    double *x,           /* X array (local).                               */
    double *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    lsumBmod_buff_t* lbmod_buf,
    int_t  k,            /* The k-th component of X.                       */
    int  *bmod,        /* Modification count for L-solve.                */
    int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
    Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
    int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
    int_t  *xsup,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    MPI_Request send_req[], /* input/output */
    SuperLUStat_t *stat
    , xtrsTimer_t *xtrsTimer);

extern int_t
pdReDistribute3d_B_to_X (double *B, int_t m_loc, int nrhs, int_t ldb,
                       int_t fst_row, int_t * ilsum, double *x,
                       dScalePermstruct_t * ScalePermstruct,
                       Glu_persist_t * Glu_persist,
                       gridinfo3d_t * grid3d, dSOLVEstruct_t * SOLVEstruct);


extern int_t
pdReDistribute3d_X_to_B (int_t n, double *B, int_t m_loc, int_t ldb,
                       int_t fst_row, int nrhs, double *x, int_t * ilsum,
                       dScalePermstruct_t * ScalePermstruct,
                       Glu_persist_t * Glu_persist, gridinfo3d_t * grid3d,
                       dSOLVEstruct_t * SOLVEstruct);

extern void
pdgstrs3d (superlu_dist_options_t *, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info);

extern void
pdgstrs3d_newsolve (superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
           dScalePermstruct_t * ScalePermstruct,
           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, double *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info);

extern int_t pdgsTrBackSolve3d(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                        dScalePermstruct_t * ScalePermstruct,
                        dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        double *x3d, double *lsum3d,
                        xT_struct *xT_s,
                        double * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pdgsTrForwardSolve3d(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           double *x3d, double *lsum3d,
                           xT_struct *xT_s,
                           double * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pdgsTrForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                           dScalePermstruct_t * ScalePermstruct,
                           dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           double *x3d, double *lsum3d,
                           double * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

extern int_t pdgsTrBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n, dLUstruct_t * LUstruct,
                        dtrf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        double *x3d, double *lsum3d,
                        double * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        dSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer);

int_t dbroadcastAncestor3d( dtrf3Dpartition_t*  trf3Dpartition,
			   dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT );

extern int_t localSolveXkYk( trtype_t trtype, int_t k, double* x, int nrhs,
                      dLUstruct_t * LUstruct, gridinfo_t * grid,
                      SuperLUStat_t * stat);

extern int_t iBcastXk2Pck(int_t k, double* x, int nrhs,
                   int** sendList, MPI_Request *send_req,
                   dLUstruct_t * LUstruct, gridinfo_t * grid,xtrsTimer_t *xtrsTimer);

extern int_t trs_B_init3d(int_t nsupers, double* x, int nrhs, dLUstruct_t * LUstruct, gridinfo3d_t *grid3d);
extern int_t trs_X_gather3d(double* x, int nrhs, dtrf3Dpartition_t*  trf3Dpartition,
                     dLUstruct_t* LUstruct,
                     gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer);
extern int_t fsolveReduceLsum3d(int_t treeId, int_t sender, int_t receiver, double* lsum, double* recvbuf, int nrhs,
                         dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct,
                          gridinfo3d_t* grid3d,xtrsTimer_t *xtrsTimer);

extern int_t bsolve_Xt_bcast(int_t ilvl, xT_struct *xT_s, int nrhs, dtrf3Dpartition_t*  trf3Dpartition,
                     dLUstruct_t * LUstruct,gridinfo3d_t* grid3d , xtrsTimer_t *xtrsTimer);

extern int_t zAllocBcast(int_t size, void** ptr, gridinfo3d_t* grid3d);
extern int_t zAllocBcast_gridID(int_t size, void** ptr, int_t gridID, gridinfo3d_t* grid3d);

extern int_t p2pSolvedX3d(int_t treeId, int_t sender, int_t receiver, double* x, int nrhs,
                      dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer);





/* Memory-related */
extern double  *doubleMalloc_dist(int_t);
extern double  *doubleCalloc_dist(int_t);
extern void  *duser_malloc_dist (int_t, int_t);
extern void  duser_free_dist (int_t, int_t);
extern int_t dQuerySpace_dist(int_t, dLUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void dClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void dCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void dZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void dScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, double);
extern void dScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, double);
extern void dZeroLblocks(int, int, gridinfo_t *, dLUstruct_t *);
extern void dZeroUblocks(int iam, int n, gridinfo_t *, dLUstruct_t *);
extern void    dfill_dist (double *, int_t, double);
extern void    dinf_norm_error_dist (int_t, int_t, double*, int_t,
                                     double*, int_t, gridinfo_t*);
extern void    pdinf_norm_error(int, int_t, int_t, double [], int_t,
				double [], int_t , MPI_Comm);
extern void  dreadhb_dist (int, FILE *, int_t *, int_t *, int_t *,
			   double **, int_t **, int_t **);
extern void  dreadtriple_dist(FILE *, int_t *, int_t *, int_t *,
			 double **, int_t **, int_t **);
extern void  dreadtriple_noheader(FILE *, int_t *, int_t *, int_t *,
			 double **, int_t **, int_t **);
extern void  dreadrb_dist(int, FILE *, int_t *, int_t *, int_t *,
		     double **, int_t **, int_t **);
extern void  dreadMM_dist(FILE *, int_t *, int_t *, int_t *,
	                  double **, int_t **, int_t **);
extern int  dread_binary(FILE *, int_t *, int_t *, int_t *,
	                  double **, int_t **, int_t **);

/* Distribute the data for numerical factorization */
extern float ddist_psymbtonum(superlu_dist_options_t *, int_t, SuperMatrix *,
                                dScalePermstruct_t *, Pslu_freeable_t *,
                                dLUstruct_t *, gridinfo_t *);
extern void pdGetDiagU(int_t, dLUstruct_t *, gridinfo_t *, double *);

extern int  d_c2cpp_GetHWPM(SuperMatrix *, gridinfo_t *, dScalePermstruct_t *);

/* Routines for debugging */
extern void  dPrintLblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
		 	   dLocalLU_t *);
extern void  dPrintUblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
			   dLocalLU_t *);
extern void  dPrint_CompCol_Matrix_dist(SuperMatrix *);
extern void  dPrint_Dense_Matrix_dist(SuperMatrix *);
extern int   dPrint_CompRowLoc_Matrix_dist(SuperMatrix *);
extern int   file_dPrint_CompRowLoc_Matrix_dist(FILE *fp, SuperMatrix *A);
extern void  Printdouble5(char *, int_t, double *);
extern int   file_Printdouble5(FILE *, char *, int_t, double *);

extern void dGenCOOLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, dLocalLU_t *, int_t** , int_t** , double ** , int_t* , int_t* );
extern void dGenCSCLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, dLocalLU_t *, double **, int_t **, int_t **, int_t*, int_t*);
extern void dGenCSRLblocks(int, int_t, gridinfo_t*,
		  Glu_persist_t*, dLocalLU_t *, double **, int_t **, int_t **, int_t*, int_t*);

/* multi-GPU */
#ifdef GPU_SOLVE
// extern void create_nv_buffer(int* , int*, int* , int* );
extern void nv_init_wrapper(MPI_Comm);
// extern void nv_init_wrapper(int* c, char *v[], int* omp_mpi_level);
extern void prepare_multiGPU_buffers(int,int,int,int,int,int);
extern void delete_multiGPU_buffers();
#endif
/* BLAS */

#ifdef USE_VENDOR_BLAS
extern void dgemm_(const char*, const char*, const int*, const int*, const int*,
                  const double*, const double*, const int*, const double*,
                  const int*, const double*, double*, const int*, int, int);
extern void dtrsv_(char*, char*, char*, int*, double*, int*,
                  double*, int*, int, int, int);
extern void dtrsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const double*, const double*, const int*,
		  double*, const int*, int, int, int, int);
extern void dgemv_(const char *, const int *, const int *, const double *,
                  const double *a, const int *, const double *, const int *,
		  const double *, double *, const int *, int);

#else
extern int dgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double*,  const double*,  const int*,  const double*,
                   const int*,  const double*, double*, const int*);
extern int dtrsv_(char*, char*, char*, int*, double*, int*,
                  double*, int*);
extern int dtrsm_(const char*, const char*, const char*, const char*,
                  const int*, const int*, const double*, const double*, const int*,
		  double*, const int*);
extern void dgemv_(const char *, const int *, const int *, const double *,
                  const double *a, const int *, const double *, const int *,
		  const double *, double *, const int *);
#endif

extern void dger_(const int*, const int*, const double*,
                 const double*, const int*, const double*, const int*,
		 double*, const int*);

extern int dscal_(const int *n, const double *alpha, double *dx, const int *incx);
extern int daxpy_(const int *n, const double *alpha, const double *x,
	               const int *incx, double *y, const int *incy);

/* SuperLU BLAS interface: dsuperlu_blas.c  */
extern int superlu_dgemm(const char *transa, const char *transb,
                  int m, int n, int k, double alpha, double *a,
                  int lda, double *b, int ldb, double beta, double *c, int ldc);
extern int superlu_dtrsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag, const int m, const int n,
                  const double alpha, const double *a,
                  const int lda, double *b, const int ldb);
extern int superlu_dger(const int m, const int n, const double alpha,
                 const double *x, const int incx, const double *y,
                 const int incy, double *a, const int lda);
extern int superlu_dscal(const int n, const double alpha, double *x, const int incx);
extern int superlu_daxpy(const int n, const double alpha,
    const double *x, const int incx, double *y, const int incy);
extern int superlu_dgemv(const char *trans, const int m,
                  const int n, const double alpha, const double *a,
                  const int lda, const double *x, const int incx,
                  const double beta, double *y, const int incy);
extern int superlu_dtrsv(char *uplo, char *trans, char *diag,
                  int n, double *a, int lda, double *x, int incx);

#ifdef SLU_HAVE_LAPACK
extern void dtrtri_(char*, char*, int*, double*, int*, int*);
#endif

/*==== For 3D code ====*/
extern int dcreate_matrix3d(SuperMatrix *A, int nrhs, double **rhs,
                     int *ldb, double **x, int *ldx,
                     FILE *fp, gridinfo3d_t *grid3d);
extern int dcreate_matrix_postfix3d(SuperMatrix *A, int nrhs, double **rhs,
                           int *ldb, double **x, int *ldx,
                           FILE *fp, char * postfix, gridinfo3d_t *grid3d);
extern int dcreate_block_diag_3d(SuperMatrix *A, int batchCount, int nrhs, double **rhs,
				 int *ldb, double **x, int *ldx,
				 FILE *fp, char * postfix, gridinfo3d_t *grid3d);
    
/* Matrix distributed in NRformat_loc in 3D process grid. It converts 
   it to a NRformat_loc distributed in 2D grid in grid-0 */
extern void dGatherNRformat_loc3d(fact_t Fact, NRformat_loc *A, double *B,
				   int ldb, int nrhs, gridinfo3d_t *grid3d,
				   NRformat_loc3d **);

extern void dGatherNRformat_loc3d_allgrid(fact_t Fact, NRformat_loc *A, double *B,
				   int ldb, int nrhs, gridinfo3d_t *grid3d,
				   NRformat_loc3d **);
extern int dScatter_B3d(NRformat_loc3d *A3d, gridinfo3d_t *grid3d);

extern void pdgssvx3d (superlu_dist_options_t *, SuperMatrix *,
		       dScalePermstruct_t *, double B[], int ldb, int nrhs,
		       gridinfo3d_t *, dLUstruct_t *, dSOLVEstruct_t *,
		       double *berr, SuperLUStat_t *, int *info);
extern int_t pdgstrf3d(superlu_dist_options_t *, int m, int n, double anorm,
		       dtrf3Dpartition_t*, SCT_t *, dLUstruct_t *,
		       gridinfo3d_t *, SuperLUStat_t *, int *);
extern void dInit_HyP(HyP_t* HyP, dLocalLU_t *Llu, int_t mcb, int_t mrb );
extern void Free_HyP(HyP_t* HyP);
extern int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid);



    /* from scatter.h */
extern void
dblock_gemm_scatter( int_t lb, int_t j, Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info, double *L_mat, int ldl,
                    double *U_mat, int ldu,  double *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int *indirect, int *indirect2,
                    int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *, SuperLUStat_t *
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );

#ifdef _OPENMP
/*this version uses a lock to prevent multiple thread updating the same block*/
extern void
dblock_gemm_scatter_lock( int_t lb, int_t j, omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,  Remain_info_t *Remain_info,
                         double *L_mat, int_t ldl, double *U_mat, int_t ldu,
                         double *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int *indirect, int *indirect2,
                         int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );
#endif

extern int_t
dblock_gemm_scatterTopLeft( int_t lb,  int_t j, double* bigV,
				 int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,
				 int* indirect, int* indirect2,
                                 HyP_t* HyP, dLUstruct_t *, gridinfo_t*,
                                 SCT_t*SCT, SuperLUStat_t *
                               );
extern int_t
dblock_gemm_scatterTopRight( int_t lb,  int_t j, double* bigV,
				  int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,
				  int* indirect, int* indirect2,
                                  HyP_t* HyP, dLUstruct_t *, gridinfo_t*,
                                  SCT_t*SCT, SuperLUStat_t * );
extern int_t
dblock_gemm_scatterBottomLeft( int_t lb,  int_t j, double* bigV,
				    int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt,
				    int* indirect, int* indirect2,
                                    HyP_t* HyP, dLUstruct_t *, gridinfo_t*,
                                    SCT_t*SCT, SuperLUStat_t * );
extern int_t
dblock_gemm_scatterBottomRight( int_t lb,  int_t j, double* bigV,
				     int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,
				     int* indirect, int* indirect2,
                                     HyP_t* HyP, dLUstruct_t *, gridinfo_t*,
                                     SCT_t*SCT, SuperLUStat_t * );

    /* from gather.h */
extern void dgather_u(int_t num_u_blks,
              Ublock_info_t *Ublock_info, int_t * usub,
              double *uval,  double *bigU,  int_t ldu,
              int_t *xsup, int_t klst                /* for SuperSize */
             );

extern void dgather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               double * lval, int_t LD_lval,
               double * L_buff );

extern void dRgather_L(int_t k, int_t *lsub, double *lusup, gEtreeInfo_t*,
		      Glu_persist_t *, gridinfo_t *, HyP_t *,
		      int_t *myIperm, int_t *iperm_c_supno );
extern void dRgather_U(int_t k, int_t jj0, int_t *usub, double *uval,
		      double *bigU, gEtreeInfo_t*, Glu_persist_t *,
		      gridinfo_t *, HyP_t *, int_t *myIperm,
		      int_t *iperm_c_supno, int_t *perm_u);

    /* from xtrf3Dpartition.h */
extern dtrf3Dpartition_t* dinitTrf3Dpartition(int_t nsupers,
					     superlu_dist_options_t *options,
					     dLUstruct_t *LUstruct, gridinfo3d_t * grid3d);

extern dtrf3Dpartition_t* dinitTrf3Dpartition_allgrid(int_t n, superlu_dist_options_t *options,
				      dLUstruct_t *LUstruct, gridinfo3d_t * grid3d
				      );

extern dtrf3Dpartition_t* dinitTrf3DpartitionLUstructgrid0(int_t n,
					     superlu_dist_options_t *options,
					     dLUstruct_t *LUstruct, gridinfo3d_t * grid3d);                         
extern void dDestroy_trf3Dpartition(dtrf3Dpartition_t *trf3Dpartition);

extern void d3D_printMemUse(dtrf3Dpartition_t*  trf3Dpartition,
			    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d);

//extern int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
//		       superlu_dist_options_t *options, dLocalLU_t *Llu,
//		       int_t* xsup, int_t num_look_aheads, int_t nsupers,
//		       int_t * iperm_c_supno);

extern void dinit3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t**  sForests, dLUstruct_t* LUstruct,
				  gridinfo3d_t* grid3d);

extern int_t dgatherAllFactoredLUFr(int_t* myZeroTrIdxs, sForest_t* sForests,
				   dLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
				   SCT_t* SCT );

    /* The following are from pdgstrf2.h */
extern int_t dLpanelUpdate(int_t off0, int_t nsupc, double* ublk_ptr,
			  int_t ld_ujrow, double* lusup, int_t nsupr, SCT_t*);
extern void dgstrf2(int_t k, double* diagBlk, int_t LDA, double* BlockUfactor, int_t LDU,
            double thresh, int_t* xsup, superlu_dist_options_t *options,
            SuperLUStat_t *stat, int *info);
extern void Local_Dgstrf2(superlu_dist_options_t *options, int_t k,
			  double thresh, double *BlockUFactor, Glu_persist_t *,
			  gridinfo_t *, dLocalLU_t *,
                          SuperLUStat_t *, int *info, SCT_t*);
extern int_t dTrs2_GatherU(int_t iukp, int_t rukp, int_t klst,
			  int_t nsupc, int_t ldu, int_t *usub,
			  double* uval, double *tempv);
extern int_t dTrs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
			   int_t nsupc, int_t ldu, int_t *usub,
			   double* uval, double *tempv);
extern int_t dTrs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
                             int_t *usub, double* uval, double *tempv,
                             int_t knsupc, int nsupr, double* lusup,
                             Glu_persist_t *Glu_persist)  ;
extern void pdgstrs2
#ifdef _CRAY
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    dLocalLU_t *Llu, SuperLUStat_t *stat, _fcd ftcs1, _fcd ftcs2, _fcd ftcs3
);
#else
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    dLocalLU_t *Llu, SuperLUStat_t *stat
);
#endif

extern void pdgstrf2(superlu_dist_options_t *, int_t nsupers, int_t k0,
		     int_t k, double thresh, Glu_persist_t *, gridinfo_t *,
		     dLocalLU_t *, MPI_Request *, int, SuperLUStat_t *, int *);

    /* from p3dcomm.h */
extern int_t dAllocLlu_3d(int_t nsupers, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t dp3dScatter(int_t n, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t dscatter3dLPanels(int_t nsupers,
                       dLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t dscatter3dUPanels(int_t nsupers,
                       dLUstruct_t * LUstruct, gridinfo3d_t* grid3d, int *supernodeMask);
extern int_t dcollect3dLpanels(int_t layer, int_t nsupers, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t dcollect3dUpanels(int_t layer, int_t nsupers, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
extern int_t dp3dCollect(int_t layer, int_t n, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d);
/*zero out LU non zero entries*/
extern int_t dzeroSetLU(int_t nnodes, int_t* nodeList , dLUstruct_t *, gridinfo3d_t*);
extern int dAllocGlu_3d(int_t n, int_t nsupers, dLUstruct_t *);
extern int dDeAllocLlu_3d(int_t n, dLUstruct_t *, gridinfo3d_t*);
extern int dDeAllocGlu_3d(dLUstruct_t *);

/* Reduces L and U panels of nodes in the List nodeList (size=nnnodes)
receiver[L(nodelist)] =sender[L(nodelist)] +receiver[L(nodelist)]
receiver[U(nodelist)] =sender[U(nodelist)] +receiver[U(nodelist)]
*/
int_t dreduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        double* Lval_buf, double* Uval_buf,
                        dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
/*reduces all nodelists required in a level*/
extern int dreduceAllAncestors3d(int_t ilvl, int_t* myNodeCount,
                           int_t** treePerm,
                           dLUValSubBuf_t* LUvsb,
                           dLUstruct_t* LUstruct,
                           gridinfo3d_t* grid3d,
                           SCT_t* SCT );
/*
	Copies factored L and U panels from sender grid to receiver grid
	receiver[L(nodelist)] <-- sender[L(nodelist)];
	receiver[U(nodelist)] <-- sender[U(nodelist)];
*/
int_t dgatherFactoredLU(int_t sender, int_t receiver,
                       int_t nnodes, int_t *nodeList, dLUValSubBuf_t*  LUvsb,
                       dLUstruct_t* LUstruct, gridinfo3d_t* grid3d,SCT_t* SCT );

/*Gathers all the L and U factors to grid 0 for solve stage
	By  repeatidly calling above function*/
int_t dgatherAllFactoredLU(dtrf3Dpartition_t*  trf3Dpartition, dLUstruct_t* LUstruct,
			   gridinfo3d_t* grid3d, SCT_t* SCT );

/*Distributes data in each layer and initilizes ancestors
 as zero in required nodes*/
int_t dinit3DLUstruct( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                      int_t* nodeCount, int_t** nodeList,
                      dLUstruct_t* LUstruct, gridinfo3d_t* grid3d);

int_t dzSendLPanel(int_t k, int_t receiver,
		   dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t dzRecvLPanel(int_t k, int_t sender, double alpha,
                   double beta, double* Lval_buf,
		   dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t dzSendUPanel(int_t k, int_t receiver,
		   dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t dzRecvUPanel(int_t k, int_t sender, double alpha,
                   double beta, double* Uval_buf,
		   dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);

    /* from communication_aux.h */
extern int_t dIBcast_LPanel (int_t k, int_t k0, int_t* lsub, double* lusup,
			     gridinfo_t *, int* msgcnt, MPI_Request *,
			     int **ToSendR, int_t *xsup, int );
extern int_t dBcast_LPanel(int_t k, int_t k0, int_t* lsub, double* lusup,
			   gridinfo_t *, int* msgcnt, int **ToSendR,
			   int_t *xsup , SCT_t*, int);
extern int_t dIBcast_UPanel(int_t k, int_t k0, int_t* usub, double* uval,
			    gridinfo_t *, int* msgcnt, MPI_Request *,
			    int *ToSendD, int );
extern int_t dBcast_UPanel(int_t k, int_t k0, int_t* usub, double* uval,
			   gridinfo_t *, int* msgcnt, int *ToSendD, SCT_t*, int);
extern int_t dIrecv_LPanel (int_t k, int_t k0,  int_t* Lsub_buf,
			    double* Lval_buf, gridinfo_t *,
			    MPI_Request *, dLocalLU_t *, int);
extern int_t dIrecv_UPanel(int_t k, int_t k0, int_t* Usub_buf, double*,
			   dLocalLU_t *, gridinfo_t*, MPI_Request *, int);
extern int_t dWait_URecv(MPI_Request *, int* msgcnt, SCT_t *);
extern int_t dWait_LRecv(MPI_Request*, int* msgcnt, int* msgcntsU,
			 gridinfo_t *, SCT_t*);
extern int_t dISend_UDiagBlock(int_t k0, double *ublk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t dRecv_UDiagBlock(int_t k0, double *ublk_ptr, int_t size,
			      int_t src, gridinfo_t *, SCT_t*, int);
extern int_t dPackLBlock(int_t k, double* Dest, Glu_persist_t *,
			 gridinfo_t *, dLocalLU_t *);
extern int_t dISend_LDiagBlock(int_t k0, double *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *, int);
extern int_t dIRecv_UDiagBlock(int_t k0, double *ublk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t *,
			       SCT_t*, int);
extern int_t dIRecv_LDiagBlock(int_t k0, double *L_blk_ptr, int_t size,
			       int_t src, MPI_Request *, gridinfo_t*, SCT_t*, int);
extern int_t dUDiagBlockRecvWait( int_t k,  int* IrecvPlcd_D, int* factored_L,
				  MPI_Request *, gridinfo_t *, dLUstruct_t *, SCT_t *);

#if (MPI_VERSION>2)
extern int_t dIBcast_UDiagBlock(int_t k, double *ublk_ptr, int_t size,
				MPI_Request *, gridinfo_t *);
extern int_t dIBcast_LDiagBlock(int_t k, double *lblk_ptr, int_t size,
			       MPI_Request *, gridinfo_t *);
#endif

    /* from trfCommWrapper.h */
extern int_t dDiagFactIBCast(int_t k,  int_t k0,
			     double *BlockUFactor, double *BlockLFactor,
			     int* IrecvPlcd_D, MPI_Request *, MPI_Request *,
			     MPI_Request *, MPI_Request *, gridinfo_t *,
			     superlu_dist_options_t *, double thresh,
			     dLUstruct_t *LUstruct, SuperLUStat_t *, int *info,
			     SCT_t *, int tag_ub);
extern int_t dUPanelTrSolve( int_t k, double* BlockLFactor, double* bigV,
			     int_t ldt, Ublock_info_t*, gridinfo_t *,
			     dLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t dLPanelUpdate(int_t k,  int* IrecvPlcd_D, int* factored_L,
			   MPI_Request *, double* BlockUFactor, gridinfo_t *,
			   dLUstruct_t *, SCT_t *);
extern int_t dUPanelUpdate(int_t k, int* factored_U, MPI_Request *,
			   double* BlockLFactor, double* bigV,
			   int_t ldt, Ublock_info_t*, gridinfo_t *,
			   dLUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t dIBcastRecvLPanel(int_t k, int_t k0, int* msgcnt,
			       MPI_Request *, MPI_Request *,
			       int_t* Lsub_buf, double* Lval_buf,
			      int * factored, gridinfo_t *, dLUstruct_t *,
			      SCT_t *, int tag_ub);
extern int_t dIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt, MPI_Request *,
			       MPI_Request *, int_t* Usub_buf, double* Uval_buf,
			       gridinfo_t *, dLUstruct_t *, SCT_t *, int tag_ub);
extern int_t dWaitL(int_t k, int* msgcnt, int* msgcntU, MPI_Request *,
		    MPI_Request *, gridinfo_t *, dLUstruct_t *, SCT_t *);
extern int_t dWaitU(int_t k, int* msgcnt, MPI_Request *, MPI_Request *,
		   gridinfo_t *, dLUstruct_t *, SCT_t *);
extern int_t dLPanelTrSolve(int_t k, int* factored_L, double* BlockUFactor,
			    gridinfo_t *, dLUstruct_t *);

    /* from trfAux.h */
extern int getNsupers(int, Glu_persist_t *);
extern int_t initPackLUInfo(int_t nsupers, packLUInfo_t* packLUInfo);
extern int   freePackLUInfo(packLUInfo_t* packLUInfo);
extern int_t dSchurComplementSetup(int_t k, int *msgcnt, Ublock_info_t*,
				   Remain_info_t*, uPanelInfo_t *,
				   lPanelInfo_t *, int_t*, int_t *, int_t *,
				   double *bigU, int_t* Lsub_buf,
				   double* Lval_buf, int_t* Usub_buf,
				   double* Uval_buf, gridinfo_t *, dLUstruct_t *);
extern int_t dSchurComplementSetupGPU(int_t k, msgs_t* msgs, packLUInfo_t*,
				      int_t*, int_t*, int_t*, gEtreeInfo_t*,
				      factNodelists_t*, dscuBufs_t*,
				      dLUValSubBuf_t* LUvsb, gridinfo_t *,
				      dLUstruct_t *, HyP_t*);
extern double* dgetBigV(int_t, int_t);
extern double* dgetBigU(superlu_dist_options_t *,
                           int_t, gridinfo_t *, dLUstruct_t *);
// permutation from superLU default

    /* from treeFactorization.h */
extern int_t dLluBufInit(dLUValSubBuf_t*, dLUstruct_t *);
extern int_t dinitScuBufs(superlu_dist_options_t *,
                          int_t ldt, int_t num_threads, int_t nsupers,
			  dscuBufs_t*, dLUstruct_t*, gridinfo_t *);
extern int dfreeScuBufs(dscuBufs_t* scuBufs);

#if 0 // NOT CALLED
// the generic tree factoring code
extern int_t treeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    dscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    dLUValSubBuf_t* LUvsb,
    ddiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);
#endif

extern int_t dsparseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo,
    commRequests_t *comReqs,    // lists of communication requests
    dscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    dLUValSubBuf_t* LUvsb,
    ddiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t ddenseTreeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    dscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    dLUValSubBuf_t* LUvsb,
    ddiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);

extern int_t dsparseTreeFactor_ASYNC(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    dscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    dLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    ddiagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);
extern dLUValSubBuf_t** dLluBufInitArr(int_t numLA, dLUstruct_t *LUstruct);
extern int dLluBufFreeArr(int_t numLA, dLUValSubBuf_t **LUvsbs);
extern ddiagFactBufs_t** dinitDiagFactBufsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid);
extern ddiagFactBufs_t** dinitDiagFactBufsArrMod(int_t mxLeafNode, int_t* ldts, gridinfo_t* grid);
extern int dfreeDiagFactBufsArr(int_t mxLeafNode, ddiagFactBufs_t** dFBufs);
extern int_t dinitDiagFactBufs(int_t ldt, ddiagFactBufs_t* dFBuf);
extern int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
			    gridinfo_t *grid, SCT_t *SCT);
extern int_t checkRecvLDiag(int_t k, commRequests_t *comReqs, gridinfo_t *, SCT_t *);

#if 0 // NOT CALLED
/* from ancFactorization.h (not called) */
extern int_t ancestorFactor(
    int_t ilvl,             // level of factorization
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    dscuBufs_t *scuBufs,     // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    dLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    ddiagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    dLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub, int *info
);
#endif

/*== end 3D prototypes ===================*/


#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

