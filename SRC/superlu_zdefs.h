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
 * -- Distributed SuperLU routine (version 6.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * April 5, 2015
 * September 18, 2018  version 6.0
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
    doublecomplex **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    doublecomplex **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc)                 */
    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)  pointers to locations in Lrowind_bc_ptr and Lnzval_bc_ptr */
    int_t   *Unnz; /* number of nonzeros per block column in U*/
	int_t   **Lrowind_bc_2_lsum; /* size ceil(NSUPERS/Pc)  map indices of Lrowind_bc_ptr to indices of lsum  */  
    doublecomplex  **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc)     	*/
    int_t   **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
    doublecomplex  **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr)                 */
        /*-- Data structures used for broadcast and reduction trees. --*/
    BcTree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    RdTree  *LRtree_ptr;       /* size ceil(NSUPERS/Pr)                */
    BcTree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
    RdTree  *URtree_ptr;       /* size ceil(NSUPERS/Pr)			*/
#if 0
    int_t   *Lsub_buf;        /* Buffer for the remote subscripts of L */
    double  *Lval_buf;        /* Buffer for the remote nonzeros of L   */
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
} SOLVEstruct_t;


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
			 LUstruct_t *, gridinfo_t *);
extern void  pzgssvx_ABglobal(superlu_dist_options_t *, SuperMatrix *, 
			      ScalePermstruct_t *, doublecomplex *,
			      int, int, gridinfo_t *, LUstruct_t *, double *,
			      SuperLUStat_t *, int *);
extern float pzdistribute(fact_t, int_t, SuperMatrix *, 
			 ScalePermstruct_t *, Glu_freeable_t *, 
			 LUstruct_t *, gridinfo_t *);
extern void  pzgssvx(superlu_dist_options_t *, SuperMatrix *, 
		     ScalePermstruct_t *, doublecomplex *,
		     int, int, gridinfo_t *, LUstruct_t *,
		     SOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void  pzCompute_Diag_Inv(int_t, LUstruct_t *,gridinfo_t *, SuperLUStat_t *, int *);
extern int  zSolveInit(superlu_dist_options_t *, SuperMatrix *, int_t [], int_t [],
		       int_t, LUstruct_t *, gridinfo_t *, SOLVEstruct_t *);
extern void zSolveFinalize(superlu_dist_options_t *, SOLVEstruct_t *);
extern int_t pxgstrs_init(int_t, int_t, int_t, int_t,
                          int_t [], int_t [], gridinfo_t *grid,
	                  Glu_persist_t *, SOLVEstruct_t *);
extern void pxgstrs_finalize(pxgstrs_comm_t *);
extern int  zldperm_dist(int_t, int_t, int_t, int_t [], int_t [],
		    doublecomplex [], int_t *, double [], double []);
extern int  static_schedule(superlu_dist_options_t *, int, int, 
		            LUstruct_t *, gridinfo_t *, SuperLUStat_t *,
			    int_t *, int_t *, int *);
extern void LUstructInit(const int_t, LUstruct_t *);
extern void LUstructFree(LUstruct_t *);
extern void Destroy_LU(int_t, gridinfo_t *, LUstruct_t *);
extern void Destroy_Tree(int_t, gridinfo_t *, LUstruct_t *);

/* #define GPU_PROF
#define IPM_PROF */

extern int_t pzgstrf(superlu_dist_options_t *, int, int, double,
		    LUstruct_t*, gridinfo_t*, SuperLUStat_t*, int*);
extern void pzgstrs_Bglobal(int_t, LUstruct_t *, gridinfo_t *,
			     doublecomplex *, int_t, int, SuperLUStat_t *, int *);
extern void pzgstrs(int_t, LUstruct_t *, ScalePermstruct_t *, gridinfo_t *,
		    doublecomplex *, int_t, int_t, int_t, int, SOLVEstruct_t *,
		    SuperLUStat_t *, int *);
extern void zlsum_fmod(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int, int_t , int_t *, int_t, int_t, int_t,
		       int_t *, gridinfo_t *, LocalLU_t *, 
		       MPI_Request [], SuperLUStat_t *);
extern void zlsum_bmod(doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       MPI_Request [], SuperLUStat_t *);

extern void zlsum_fmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int_t , int_t *,
		       int_t *, gridinfo_t *, LocalLU_t *, 
		       SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_fmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
		       int, int, int_t , int_t *, int_t, 
		       int_t *, gridinfo_t *, LocalLU_t *, 
		       SuperLUStat_t **, int_t, int_t, int_t, int_t, int, int);
extern void zlsum_bmod_inv(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       MPI_Request [], SuperLUStat_t **, int_t *, int_t *, int_t, int_t, int, int);
extern void zlsum_bmod_inv_master(doublecomplex *, doublecomplex *, doublecomplex *, doublecomplex *,
                       int, int_t, int_t *, int_t *, int_t *, Ucb_indptr_t **,
                       int_t **, int_t *, gridinfo_t *, LocalLU_t *,
		       MPI_Request [], SuperLUStat_t **, int_t, int_t, int, int);			   
			   
extern void pzgsrfs(int_t, SuperMatrix *, double, LUstruct_t *,
		    ScalePermstruct_t *, gridinfo_t *,
		    doublecomplex [], int_t, doublecomplex [], int_t, int,
		    SOLVEstruct_t *, double *, SuperLUStat_t *, int *);
extern void pzgsrfs_ABXglobal(int_t, SuperMatrix *, double, LUstruct_t *,
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
extern int_t zQuerySpace_dist(int_t, LUstruct_t *, gridinfo_t *,
			      SuperLUStat_t *, superlu_dist_mem_usage_t *);

/* Auxiliary routines */

extern void zClone_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zCopy_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *);
extern void zZero_CompRowLoc_Matrix_dist(SuperMatrix *);
extern void zScaleAddId_CompRowLoc_Matrix_dist(SuperMatrix *, doublecomplex);
extern void zScaleAdd_CompRowLoc_Matrix_dist(SuperMatrix *, SuperMatrix *, doublecomplex);
extern void zZeroLblocks(int, int_t, gridinfo_t *, LUstruct_t *);
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
                                ScalePermstruct_t *, Pslu_freeable_t *, 
                                LUstruct_t *, gridinfo_t *);
extern void pzGetDiagU(int_t, LUstruct_t *, gridinfo_t *, doublecomplex *);


/* Routines for debugging */
extern void  zPrintLblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
		 	   LocalLU_t *);
extern void  zPrintUblocks(int, int_t, gridinfo_t *, Glu_persist_t *,
			   LocalLU_t *);
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
extern void ztrtri_(char*, char*, int*, doublecomplex*, int*,int*);				 

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


#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dDEFS */

