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
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * April 5, 2015
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
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

/*-- Auxiliary data type used in PxGSTRS/PxGSTRS1. */
typedef struct {
    int_t lbnum;  /* Row block number (local).      */
    int_t indpos; /* Starting position in Uindex[]. */
} Ucb_indptr_t;

/*
 * On each processor, the blocks in L are stored in compressed block
 * column format, the blocks in U are stored in compressed block row format.
 */
#define MAX_LOOKAHEADS 50
typedef struct {
    int_t   **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
    double **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    double **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr */
    int_t   *Unnz; /* number of nonzeros per block column in U*/
	int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */
    double  **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    double  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
        /*-- Data structures used for broadcast and reduction trees. --*/
    BcTree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    RdTree  *LRtree_ptr;       /* size ceil(NSUPERS/Pr)                */
    BcTree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    RdTree  *URtree_ptr;       /* size ceil(NSUPERS/Pr)			*/
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
} LocalLU_t;


typedef struct {
    int_t *etree;
    Glu_persist_t *Glu_persist;
    LocalLU_t *Llu;
    char dt;
} LUstruct_t;


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
    double *val_tosend;   /* X values to be sent to other processes */
    double *val_torecv;   /* X values to be received from other processes */
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
} SOLVEstruct_t;


/*==== For 3D code ====*/

// new structures for pdgstrf_4_8 

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

typedef struct
{
    Remain_info_t *lookAhead_info, *Remain_info;
    Ublock_info_t *Ublock_info, *Ublock_info_Phi;
    
    int_t first_l_block_acc , first_u_block_acc;
    int_t last_offload ;
    int_t *Lblock_dirty_bit, * Ublock_dirty_bit;
    double *lookAhead_L_buff, *Remain_L_buff;
    int_t lookAheadBlk , RemainBlk ;
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
    int_t offloadCondition;
    int_t superlu_acc_offload;
    int_t nCudaStreams;

} HyP_t;

typedef struct 
{
    int_t * Lsub_buf ;
    double * Lval_buf ;
    int_t * Usub_buf ;
    double * Uval_buf ;
} LUValSubBuf_t;

int_t scuStatUpdate(
    int_t knsupc,
    HyP_t* HyP, 
    SCT_t* SCT,
    SuperLUStat_t *stat
    );

typedef struct trf3Dpartition_t
{
    gEtreeInfo_t gEtreeInfo;
    int_t* iperm_c_supno;
    int_t* myNodeCount;
    int_t* myTreeIdxs;
    int_t* myZeroTrIdxs;
    int_t** treePerm;
    sForest_t** sForests;
    int_t* supernode2treeMap;
    LUValSubBuf_t *LUvsb;
} trf3Dpartition_t;

typedef struct
{
    double *bigU;
    double *bigV;
} scuBufs_t;

typedef struct
{   
    double* BlockLFactor;
    double* BlockUFactor;
} diagFactBufs_t;

typedef struct
{
    Ublock_info_t* Ublock_info;
    Remain_info_t*  Remain_info;
    uPanelInfo_t* uPanelInfo;
    lPanelInfo_t* lPanelInfo;
} packLUInfo_t;

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
extern int 	   dcreate_matrix_postfix(SuperMatrix *, int, double **, int *,
				  double **, int *, FILE *, char *, gridinfo_t *);


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

extern float ddistribute(fact_t, int_t, SuperMatrix *, Glu_freeable_t *,
			 LUstruct_t *, gridinfo_t *);
extern void  pdgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *,
			      ScalePermstruct_t *, double *,
			      int, int, gridinfo_t *, LUstruct_t *, double *,
			      SuperLUStat_t *, int *);
extern float pddistribute(fact_t, int_t, SuperMatrix *,
			 ScalePermstruct_t *, Glu_freeable_t *,
			 LUstruct_t *, gridinfo_t *);
extern void  pdgssvx(superlu_dist_options_t *, SuperMatrix *,
		     ScalePermstruct_t *, double *,
		     int, int, gridinfo_t *, LUstruct_t *,
		     SOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void  pdCompute_Diag_Inv(int_t, LUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  dSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, LUstruct_t *, gridinfo_t *, SOLVEstruct_t *);
extern void dSolveFinalize(superlu_dist_options_t *, SOLVEstruct_t *);
extern int_t pxgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, SOLVEstruct_t *);
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  dldperm_dist(int_t, int_t, int_t, int_t [], int_t [],
		    double [], int_t *, double [], double []);
extern int  static_schedule(superlu_dist_options_t *, int, int,
		            LUstruct_t *, gridinfo_t *, SuperLUStat_t *,
			    int_t *, int_t *, int *);
extern void LUstructInit(const int_t, LUstruct_t *);
extern void LUstructFree(LUstruct_t *);
extern void Destroy_LU(int_t, gridinfo_t *, LUstruct_t *);
extern void Destroy_Tree(int_t, gridinfo_t *, LUstruct_t *);

/* #define GPU_PROF
#define IPM_PROF */

extern int_t pdgstrf(superlu_dist_options_t *, int, int, double,
		    LUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);
extern void pdgstrs_Bglobal(int_t, LUstruct_t *, gridinfo_t *,
			     double *, int_t, int, SuperLUStat_t *, int *);
extern void pdgstrs(int_t, LUstruct_t *, ScalePermstruct_t *, gridinfo_t *,
		    double *, int_t, int_t, int_t, int, SOLVEstruct_t *,
		    SuperLUStat_t *, int *);
extern void dlsum_fmod(double *, double *, double *, double *,
		       int, int, int_t , int_t *, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, LocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);
extern void dlsum_bmod(double *, double *, double *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void dlsum_fmod_inv(double *, double *, double *, double *,
		       int, int_t , int_t *,
		       int_t *, gridinfo_t *, LocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void dlsum_fmod_inv_master(double *, double *, double *, double *,
		       int, int, int_t , int_t *, int_t,
		       int_t *, gridinfo_t *, LocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void dlsum_bmod_inv(double *, double *, double *, double *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void dlsum_bmod_inv_master(double *, double *, double *, double *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       SuperLUStat_t **, int_t, int_t, int, int);

extern void pdgsrfs(int_t, SuperMatrix *, double, LUstruct_t *,
		    ScalePermstruct_t *, gridinfo_t *,
		    double [], int_t, double [], int_t, int,
		    SOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void pdgsrfs_ABXglobal(int_t, SuperMatrix *, double, LUstruct_t *,
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

/* Memory-related */
extern double  *doubleMalloc_dist(int_t);
extern double  *doubleCalloc_dist(int_t);
extern void  *duser_malloc_dist (int_t, int_t);
extern void  duser_free_dist (int_t, int_t);
extern int_t dQuerySpace_dist(int_t, LUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void dClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void dCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void dZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void dScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, double);
extern void dScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, double);
extern void dZeroLblocks(int, int_t, gridinfo_t *, LUstruct_t *);
extern void    dfill_dist (double *, int_t, double);
extern void    dinf_norm_error_dist (int_t, int_t, double*, int_t,
                                     double*, int_t, gridinfo_t*);
extern void    pdinf_norm_error(int, int_t, int_t, double [], int_t,
				double [], int_t , gridinfo_t *);
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
extern float ddist_psymbtonum(fact_t, int_t, SuperMatrix *,
                                ScalePermstruct_t *, Pslu_freeable_t *,
                                LUstruct_t *, gridinfo_t *);
extern void pdGetDiagU(int_t, LUstruct_t *, gridinfo_t *, double *);


/* Routines for debugging */
extern void  dPrintLblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
		 	   LocalLU_t *);
extern void  dPrintUblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
			   LocalLU_t *);
extern void  dPrint_CompCol_Matrix_dist(SuperMatrix *);
extern void  dPrint_Dense_Matrix_dist(SuperMatrix *);
extern int   dPrint_CompRowLoc_Matrix_dist(SuperMatrix *);
extern int   file_dPrint_CompRowLoc_Matrix_dist(FILE *fp, SuperMatrix *A);
extern int   file_PrintDouble5(FILE *, char *, int_t, double *);


/* BLAS */

#ifdef USE_VENDOR_BLAS
extern void dgemm_(const char*, const char*, const int*, const int*, const int*,
                  const double*, const double*, const int*, const double*,
                  const int*, const double*, double*, const int*, int, int);
extern void dtrsv_(char*, char*, char*, int*, double*, int*,
                  double*, int*, int, int, int);
extern void dtrsm_(char*, char*, char*, char*, int*, int*,
                  double*, double*, int*, double*,
                  int*, int, int, int, int);
extern void dgemv_(char *, int *, int *, double *, double *a, int *,
                  double *, int *, double *, double *, int *, int);
extern void dtrtri_(char*, char*, int*, double*, int*,int*);

extern void dger_(int*, int*, double*, double*, int*,
                 double*, int*, double*, int*);

#else
extern int dgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double*,  const double*,  const int*,  const double*,
                   const int*,  const double*, double*, const int*);
extern int dtrsv_(char*, char*, char*, int*, double*, int*,
                  double*, int*);
extern int dtrsm_(char*, char*, char*, char*, int*, int*,
                  double*, double*, int*, double*, int*);
extern int dgemv_(char *, int *, int *, double *, double *a, int *,
                  double *, int *, double *, double *, int *);
extern void dger_(int*, int*, double*, double*, int*,
                 double*, int*, double*, int*);

#endif

/*==== For 3D code ====*/

extern int_t pdgstrf3d(superlu_dist_options_t *, int m, int n, double anorm,
		       trf3Dpartition_t*, SCT_t *, LUstruct_t *, gridinfo3d_t *,
		       SuperLUStat_t *, int *);
extern int_t zSendLPanel(int_t, int_t, LUstruct_t*,  gridinfo3d_t*, SCT_t*);
extern int_t zRecvLPanel(int_t, int_t, double, double, double*,
			 LUstruct_t*,  gridinfo3d_t*, SCT_t* SCT);
extern int_t zSendUPanel(int_t, int_t, LUstruct_t*,  gridinfo3d_t*, SCT_t*);
extern int_t zRecvUPanel(int_t, int_t, double, double, double*,
			 LUstruct_t*,  gridinfo3d_t*, SCT_t*);
extern void Init_HyP(HyP_t* HyP, LocalLU_t *Llu, int_t mcb, int_t mrb );
extern void Free_HyP(HyP_t* HyP);
extern void DistPrint(char* function_name,  double value, char* Units, gridinfo_t* grid);
extern void DistPrint3D(char* function_name,  double value, char* Units, gridinfo3d_t* grid3d);
extern void treeImbalance3D(gridinfo3d_t *grid3d, SCT_t* SCT);
extern void SCT_printComm3D(gridinfo3d_t *grid3d, SCT_t* SCT);
extern int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid);

    /* from scatter.h */
extern void
block_gemm_scatter( int_t lb, int_t j,
                    Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info,
                    double *L_mat, int_t ldl,
                    double *U_mat, int_t ldu,
                    double *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int_t *indirect,
                    int_t *indirect2,
                    int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *grid,
                    SuperLUStat_t *stat
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );
/*this version uses a lock to prevent multiple thread updating the same block*/
void
block_gemm_scatter_lock( int_t lb, int_t j,
                         omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,
                         Remain_info_t *Remain_info,
                         double *L_mat, int_t ldl,
                         double *U_mat, int_t ldu,
                         double *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int_t *indirect,
                         int_t *indirect2,
                         int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *grid
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );

int_t block_gemm_scatterTopLeft( int_t lb,  int_t j,
                                 double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                 HyP_t* HyP,
                                 LUstruct_t *LUstruct,
                                 gridinfo_t* grid,
                                 SCT_t*SCT, SuperLUStat_t *stat
                               );
int_t block_gemm_scatterTopRight( int_t lb,  int_t j,
                                  double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                  HyP_t* HyP,
                                  LUstruct_t *LUstruct,
                                  gridinfo_t* grid,
                                  SCT_t*SCT, SuperLUStat_t *stat
                                );
int_t block_gemm_scatterBottomLeft( int_t lb,  int_t j,
                                    double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                    HyP_t* HyP,
                                    LUstruct_t *LUstruct,
                                    gridinfo_t* grid,
                                    SCT_t*SCT, SuperLUStat_t *stat
                                  );
int_t block_gemm_scatterBottomRight( int_t lb,  int_t j,
                                     double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                     HyP_t* HyP,
                                     LUstruct_t *LUstruct,
                                     gridinfo_t* grid,
                                     SCT_t*SCT, SuperLUStat_t *stat
                                   );

extern void gather_u(int_t num_u_blks,
              Ublock_info_t *Ublock_info, int_t * usub,
              double *uval,  double *bigU,  int_t ldu,
              int_t *xsup, int_t klst                /* for SuperSize */
             );

extern void gather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               double * lval, int_t LD_lval,
               double * L_buff );

    /* from gather.h */
extern void Rgather_L(int_t k, int_t *lsub, double *lusup, gEtreeInfo_t*,
		      Glu_persist_t *, gridinfo_t *, HyP_t *,
		      int_t *myIperm, int_t *iperm_c_supno );
extern void Rgather_U(int_t k, int_t jj0, int_t *usub, double *uval,
		      double *bigU, gEtreeInfo_t*, Glu_persist_t *,
		      gridinfo_t *, HyP_t *, int_t *myIperm,
		      int_t *iperm_c_supno, int_t *perm_u);

    /* from xtrf3Dpartition.h */
extern trf3Dpartition_t* initTrf3Dpartition(int_t nsupers,
					    superlu_dist_options_t *options,
					    LUstruct_t *LUstruct, gridinfo3d_t * grid3d);
extern void printMemUse(trf3Dpartition_t*  trf3Dpartition,
			 LUstruct_t *LUstruct, gridinfo3d_t * grid3d);

extern int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
		       superlu_dist_options_t *options, LocalLU_t *Llu,
		       int_t* xsup, int_t num_look_aheads, int_t nsupers,
		       int_t * iperm_c_supno);

extern void init3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
				  sForest_t**  sForests, LUstruct_t* LUstruct,
				  gridinfo3d_t* grid3d);

extern int_t gatherAllFactoredLUFr(int_t* myZeroTrIdxs, sForest_t* sForests,
				   LUstruct_t* LUstruct, gridinfo3d_t* grid3d,
				   SCT_t* SCT );

    /* The following are from pdgstrf2.h */
#if 0 // Sherry: same routine names, but different code !!!!!!!
extern void pdgstrf2_trsm(superlu_dist_options_t *options, int_t, int_t,
                          int_t k, double thresh, Glu_persist_t *,
			  gridinfo_t *, LocalLU_t *, MPI_Request *U_diag_blk_send_req,
			  SuperLUStat_t *, int *info, SCT_t *);
#ifdef _CRAY
void pdgstrs2_omp (int_t, int_t, int_t, Glu_persist_t *, gridinfo_t *,
                      LocalLU_t *, SuperLUStat_t *, _fcd, _fcd, _fcd);
#else
void pdgstrs2_omp (int_t, int_t, int_t, int_t *, double*, Glu_persist_t *, gridinfo_t *,
                      LocalLU_t *, SuperLUStat_t *, Ublock_info_t *, double *bigV, int_t ldt, SCT_t *SCT );
#endif

#endif // same routine names   !!!!!!!!

extern int_t LpanelUpdate(int_t off0, int_t nsupc, double* ublk_ptr,
			  int_t ld_ujrow, double* lusup, int_t nsupr, SCT_t*);
extern void Local_Dgstrf2(superlu_dist_options_t *options, int_t k,
			  double thresh, double *BlockUFactor, Glu_persist_t *,
			  gridinfo_t *, LocalLU_t *,
                          SuperLUStat_t *, int *info, SCT_t*);
extern int_t Trs2_GatherU(int_t iukp, int_t rukp, int_t klst,
			  int_t nsupc, int_t ldu, int_t *usub,
			  double* uval, double *tempv);
extern int_t Trs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
			   int_t nsupc, int_t ldu, int_t *usub,
			   double* uval, double *tempv);
extern int_t Trs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
                             int_t *usub,
                             double* uval, double *tempv,
                             int_t knsupc, int_t nsupr, double*lusup,
                             Glu_persist_t *Glu_persist)  ;
extern int_t Trs2_InitUblock_info(int_t klst, int_t nb, Ublock_info_t *,
                                  int_t *usub, Glu_persist_t *, SuperLUStat_t*);

extern void pdgstrs2_mpf(int_t m, int_t k0, int_t k, double *Lval_buf, 
			 int_t nsupr, Glu_persist_t *,
			 gridinfo_t *, LocalLU_t *, SuperLUStat_t *,
			 Ublock_info_t *, double *bigV, int_t ldt, SCT_t *);
extern void pdgstrs2
#ifdef _CRAY
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    LocalLU_t *Llu, SuperLUStat_t *stat, _fcd ftcs1, _fcd ftcs2, _fcd ftcs3
);
#else
(
    int_t m, int_t k0, int_t k, Glu_persist_t *Glu_persist, gridinfo_t *grid,
    LocalLU_t *Llu, SuperLUStat_t *stat
);
#endif

extern void pdgstrf2(superlu_dist_options_t *, int_t nsupers, int_t k0,
		     int_t k, double thresh, Glu_persist_t *, gridinfo_t *,
		     LocalLU_t *, MPI_Request *, SuperLUStat_t *, int *);

    /* from p3dcomm.h */
int_t AllocLlu(int_t nsupers, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);
int_t AllocGlu(int_t n, int_t nsupers, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

int_t p3dScatter(int_t n, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);


int_t scatter3dLPanels(int_t nsupers,
                       LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

int_t scatter3dUPanels(int_t nsupers,
                       LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

int_t collect3dLpanels(int_t layer, int_t nsupers, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

int_t collect3dUpanels(int_t layer, int_t nsupers, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

int_t p3dCollect(int_t layer, int_t n, LUstruct_t * LUstruct, gridinfo3d_t* grid3d);

/*zero out LU non zero entries*/
int_t zeroSetLU(int_t nnodes, int_t* nodeList , LUstruct_t *LUstruct, gridinfo3d_t* grid3d);


/* Reduces L and U panels of nodes in the List nodeList (size=nnnodes)
receiver[L(nodelist)] =sender[L(nodelist)] +receiver[L(nodelist)]
receiver[U(nodelist)] =sender[U(nodelist)] +receiver[U(nodelist)]
*/

int_t reduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        double* Lval_buf, double* Uval_buf,
                        LUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);


/*reduces all nodelists required in a level*/
int_t reduceAllAncestors3d(int_t ilvl, int_t* myNodeCount,
                           int_t** treePerm,
                           LUValSubBuf_t*LUvsb,
                           LUstruct_t* LUstruct,
                           gridinfo3d_t* grid3d,
                           SCT_t* SCT );

/*
	Copies factored L and U panels from sender grid to receiver grid
	receiver[L(nodelist)] <-- sender[L(nodelist)];
	receiver[U(nodelist)] <-- sender[U(nodelist)];
*/
int_t gatherFactoredLU(int_t sender, int_t receiver,
                       int_t nnodes, int_t *nodeList, LUValSubBuf_t*LUvsb,
                       LUstruct_t* LUstruct, gridinfo3d_t* grid3d,SCT_t* SCT );

/*Gathers all the L and U factors to grid 0 for solve stage 
	By  repeatidly calling above function

*/
int_t gatherAllFactoredLU(
    trf3Dpartition_t*  trf3Dpartition,
    LUstruct_t* LUstruct,
    gridinfo3d_t* grid3d,
    SCT_t* SCT );


/*Distributes data in each layer and initilizes ancestors
 as zero in required nodes*/
int_t init3DLUstruct( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                      int_t* nodeCount, int_t** nodeList,
                      LUstruct_t* LUstruct, gridinfo3d_t* grid3d);

/*
Returns list of permutation for each
tree that I update
*/
int_t** getTreePerm( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                     int_t* nodeCount, int_t** nodeList,
                     int_t* perm_c_supno, int_t* iperm_c_supno,
                     gridinfo3d_t* grid3d);

/*number of nodes in each level of the trees which I update*/
int_t* getMyNodeCounts(int_t maxLvl, int_t* myTreeIdxs, int_t* gNodeCount);


int_t checkIntVector3d(int_t* vec, int_t len,  gridinfo3d_t* grid3d);

int_t reduceStat(PhaseType PHASE, 
  SuperLUStat_t *stat, gridinfo3d_t * grid3d);

int_t zSendLPanel(int_t k, int_t receiver,
                     LUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t zRecvLPanel(int_t k, int_t sender, double alpha, double beta,
                  double* Lval_buf,
                  LUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);

int_t zSendUPanel(int_t k, int_t receiver,
                  LUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);
int_t zRecvUPanel(int_t k, int_t sender, double alpha, double beta,
                  double* Uval_buf,
                  LUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT);

    /* from trfCommWrapper.h */
extern int_t DiagFactIBCast(int_t k,  int_t k0,
			    double *BlockUFactor, double *BlockLFactor,
			    int_t* IrecvPlcd_D, MPI_Request *, MPI_Request *,
			    MPI_Request *, MPI_Request *, gridinfo_t *,
			    superlu_dist_options_t *, double thresh,
			    LUstruct_t *LUstruct, SuperLUStat_t *, int *info,
			    SCT_t *);
extern int_t UPanelTrSolve( int_t k, double* BlockLFactor, double* bigV,
			    int_t ldt, Ublock_info_t*, gridinfo_t *,
			    LUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t Wait_LUDiagSend(int_t k, MPI_Request *, MPI_Request *,
			     gridinfo_t *, SCT_t *);
extern int_t LPanelUpdate(int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
			  MPI_Request *, double* BlockUFactor, gridinfo_t *,
			  LUstruct_t *, SCT_t *);
extern int_t UPanelUpdate(int_t k, int_t* factored_U, MPI_Request *,
			  double* BlockLFactor, double* bigV,
			  int_t ldt, Ublock_info_t*, gridinfo_t *,
			  LUstruct_t *, SuperLUStat_t *, SCT_t *);
extern int_t IBcastRecvLPanel(int_t k, int_t k0, int* msgcnt,
			      MPI_Request *, MPI_Request *,
			      int_t* Lsub_buf, double* Lval_buf,
			      int_t * factored, gridinfo_t *, LUstruct_t *,
			      SCT_t *);
extern int_t IBcastRecvUPanel(int_t k, int_t k0, int* msgcnt, MPI_Request *,
			      MPI_Request *, int_t* Usub_buf, double* Uval_buf,
			      gridinfo_t *, LUstruct_t *, SCT_t *);
extern int_t WaitL(int_t k, int* msgcnt, int* msgcntU, MPI_Request *,
		   MPI_Request *, gridinfo_t *, LUstruct_t *, SCT_t *);
extern int_t WaitU(int_t k, int* msgcnt, MPI_Request *, MPI_Request *,
		   gridinfo_t *, LUstruct_t *, SCT_t *);
extern int_t LPanelTrSolve(int_t k, int_t* factored_L, double* BlockUFactor,
			   gridinfo_t *, LUstruct_t *);

    /* from trfAux.h */
extern int_t getNsupers(int, LUstruct_t *);
extern int_t SchurComplementSetup(int_t k, int *msgcnt, Ublock_info_t*,
				  Remain_info_t*, uPanelInfo_t *,
				  lPanelInfo_t *, int_t*, int_t *, int_t *,
				  double *bigU, int_t* Lsub_buf,
				  double* Lval_buf, int_t* Usub_buf,
				  double* Uval_buf, gridinfo_t *, LUstruct_t *);
extern int_t SchurComplementSetupGPU(int_t k, msgs_t* msgs, packLUInfo_t*,
				     int_t*, int_t*, int_t*, gEtreeInfo_t*,
				     factNodelists_t*, scuBufs_t*,
				     LUValSubBuf_t* LUvsb, gridinfo_t *,
				     LUstruct_t *, HyP_t*);
extern double* getBigV(int_t, int_t);
extern double* getBigU(int_t, gridinfo_t *, LUstruct_t *);
extern int_t getBigUSize(int_t, gridinfo_t *, LUstruct_t *);
// permutation from superLU default
extern int_t* getPerm_c_supno(int_t nsupers, superlu_dist_options_t *,
			      LUstruct_t *, gridinfo_t *);

    /* from treeFactorization.h */
extern int_t LluBufInit(LUValSubBuf_t*, LUstruct_t *);
extern int_t initScuBufs(int_t ldt, int_t num_threads, int_t nsupers,
                  scuBufs_t* scuBufs,
                  LUstruct_t* LUstruct,
                  gridinfo_t * grid);
extern int_t initPackLUInfo(int_t nsupers, packLUInfo_t* packLUInfo);

extern int_t ancestorFactor(
    int_t ilvl,             // level of factorization 
    sForest_t* sforest,
    commRequests_t **comReqss, // lists of communication requests, size maxEtree level
    scuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,             // size=num Look ahead
    LUValSubBuf_t**LUvsbs,     // size=num Look ahead
    diagFactBufs_t **dFBufs,   // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t* gEtreeInfo,  // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

// the generic tree factoring code 
extern int_t treeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    LUValSubBuf_t*LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t sparseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo,
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    LUValSubBuf_t*LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t denseTreeFactor(
    int_t nnnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    LUValSubBuf_t*LUvsb,
    diagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

extern int_t sparseTreeFactor_ASYNC(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    LUValSubBuf_t**LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);
extern LUValSubBuf_t** LluBufInitArr(int_t numLA, LUstruct_t *LUstruct);
extern diagFactBufs_t** initDiagFactBufsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid);
extern int_t initDiagFactBufs(int_t ldt, diagFactBufs_t* dFBuf);
extern int_t sDiagFactIBCast(int_t k, diagFactBufs_t *dFBuf,
                      factStat_t *factStat,
                      commRequests_t *comReqs,
                      gridinfo_t *grid,
                      superlu_dist_options_t *options,
                      double thresh,
                      LUstruct_t *LUstruct,
                      SuperLUStat_t *stat, int *info,
                      SCT_t *SCT
                     );
extern int_t sLPanelUpdate( int_t k, diagFactBufs_t *dFBuf,
                     factStat_t *factStat,
                     commRequests_t *comReqs,
                     gridinfo_t *grid,
                     LUstruct_t *LUstruct, SCT_t *SCT);
extern int_t sUPanelUpdate( int_t k,
                     int_t ldt,
                     diagFactBufs_t *dFBuf,
                     factStat_t *factStat,
                     commRequests_t *comReqs,
                     scuBufs_t* scuBufs,
                     packLUInfo_t* packLUInfo,
                     gridinfo_t *grid,
                     LUstruct_t *LUstruct,
                     SuperLUStat_t *stat, SCT_t *SCT);
extern int_t sIBcastRecvLPanel(
    int_t k,
    commRequests_t *comReqs,
    LUValSubBuf_t* LUvsb,
    msgs_t* msgs,
    factStat_t *factStat,
    gridinfo_t *grid,
    LUstruct_t *LUstruct, SCT_t *SCT);

extern int_t sIBcastRecvUPanel(
    int_t k,
    commRequests_t *comReqs,
    LUValSubBuf_t* LUvsb,
    msgs_t* msgs,
    factStat_t *factStat,
    gridinfo_t *grid,
    LUstruct_t *LUstruct, SCT_t *SCT);
extern int_t sWaitL(int_t k,
             commRequests_t *comReqs,
             msgs_t* msgs,
             gridinfo_t *grid,
             LUstruct_t *LUstruct, SCT_t *SCT);
extern int_t sWaitU(int_t k,
             commRequests_t *comReqs,
             msgs_t* msgs,
             gridinfo_t *grid,
             LUstruct_t *LUstruct, SCT_t *SCT);
extern int_t sWait_LUDiagSend(int_t k,  commRequests_t *comReqs,
                       gridinfo_t *grid, SCT_t *SCT);
extern int_t sSchurComplementSetup(int_t k, msgs_t* msgs,
                            packLUInfo_t* packLUInfo,
                            int_t* gIperm_c_supno, int_t*perm_c_supno,
                            factNodelists_t* fNlists,
                            scuBufs_t* scuBufs, LUValSubBuf_t* LUvsb,
                            gridinfo_t *grid, LUstruct_t *LUstruct);
extern int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
                     gridinfo_t *grid, SCT_t *SCT);
extern int_t sLPanelTrSolve( int_t k,  diagFactBufs_t *dFBuf,
                      factStat_t *factStat,
                      commRequests_t *comReqs,
                      gridinfo_t *grid,
                      LUstruct_t *LUstruct, SCT_t *SCT);
extern int_t checkRecvLDiag(int_t k,
                     commRequests_t *comReqs,
                     gridinfo_t *grid,
                     SCT_t *SCT);
extern int_t sUPanelTrSolve( int_t k,
                      int_t ldt,
                      diagFactBufs_t *dFBuf,
                      scuBufs_t* scuBufs,
                      packLUInfo_t* packLUInfo,
                      gridinfo_t *grid,
                      LUstruct_t *LUstruct,
                      SuperLUStat_t *stat, SCT_t *SCT);
    /* from ancFactorization.h */
int_t ancestorFactor(
    int_t ilvl,             // level of factorization 
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    LUValSubBuf_t**LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT,
    int *info
);

/*=====================*/

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

