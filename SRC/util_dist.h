/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Header for utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 8.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * May 22, 2022
 * </pre>
 */

#ifndef __SUPERLU_DIST_UTIL /* allow multiple inclusions */
#define __SUPERLU_DIST_UTIL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "superlu_enum_consts.h"

/*
 * Macros
 */
#ifndef USER_ABORT
#define USER_ABORT(msg) superlu_abort_and_exit_dist(msg)
#endif

#define ABORT(err_msg) \
 { char msg[256];\
   sprintf(msg,"%s at line %d in file %s\n",err_msg,__LINE__, __FILE__);\
   USER_ABORT(msg); }


#ifndef USER_MALLOC
#define USER_MALLOC(size) superlu_malloc_dist(size)
#endif

#define SUPERLU_MALLOC(size) USER_MALLOC(size)

#ifndef USER_FREE
#define USER_FREE(addr) superlu_free_dist(addr)
#endif

#define SUPERLU_FREE(addr) USER_FREE(addr)

#define CHECK_MALLOC(pnum, where) {                 \
    extern long int superlu_malloc_total;        \
    printf("(%d) %s: superlu_malloc_total (MB) %.6f\n", \
	   pnum, where, superlu_malloc_total*1e-6); \
	fflush(stdout);        \
}

#define SUPERLU_MAX(x, y) 	( (x) > (y) ? (x) : (y) )
#define SUPERLU_MIN(x, y) 	( (x) < (y) ? (x) : (y) )

// allocating macros
#define MPI_REQ_ALLOC(x)  ((MPI_Request *) SUPERLU_MALLOC ( (x) * sizeof (MPI_Request)))
#define INT_T_ALLOC(x)  ((int_t *) SUPERLU_MALLOC ( (x) * sizeof (int_t)))
#define DOUBLE_ALLOC(x)  ((double *) SUPERLU_MALLOC ( (x) * sizeof (double)))

/* 
 * Constants 
 */
#define EMPTY	(-1)
#ifndef FALSE
#define FALSE	(0)
#endif
#ifndef TRUE
#define TRUE	(1)
#endif

/*==== For 3D code ====*/
#define MAX_3D_LEVEL 32 /*allows for z dimensions of 2^32*/
#define CBLOCK 192
#define CACHE_LINE_SIZE 8
#define CSTEPPING 8
/*=====================*/

/*
 * Type definitions
 */
typedef float    flops_t;
typedef unsigned char Logical;

/*
#ifdef _CRAY
#define int short
#endif
*/

typedef struct {
    int     *panel_histo; /* histogram of panel size distribution */
    double  *utime;       /* running time at various phases */
    flops_t *ops;         /* operation count at various phases */
    int     TinyPivots;   /* number of tiny pivots */
    int     RefineSteps;  /* number of iterative refinement steps */
    int     num_look_aheads; /* number of look ahead */
    /*-- new --*/
    float   current_buffer; /* bytes allocated for buffer in numerical factorization */
    float   peak_buffer;    /* monitor the peak buffer size (bytes) */
    float   gpu_buffer;     /* monitor the buffer allocated on GPU (bytes) */
    int_t MaxActiveBTrees;
    int_t MaxActiveRTrees;

#ifdef GPU_ACC  /*-- For GPU --*/
    double ScatterMOPCounter;
    double ScatterMOPTimer;
    double GemmFLOPCounter;
    double GemmFLOPTimer;

    double cPCIeH2D;
    double cPCIeD2H;
    double tHost_PCIeH2D;
    double tHost_PCIeD2H;

    /*GPU events to measure DGEMM and SCATTER timing */
    int *isOffloaded;  /* record whether any elimination step is offloaded or not */
    gpuEvent_t *GemmStart, *GemmEnd, *ScatterEnd;  /*GPU events to store gemm and scatter's begin and end*/
    gpuEvent_t *ePCIeH2D;
    gpuEvent_t *ePCIeD2H_Start;
    gpuEvent_t *ePCIeD2H_End;
#endif   /*-- end for GPU --*/
    
} SuperLUStat_t;


/* Headers for 2 types of dynamatically managed memory */
typedef struct e_node {
    int size;      /* length of the memory that has been used */
    void *mem;     /* pointer to the new malloc'd store */
} SuperLU_ExpHeader;

typedef struct {
    int  size;
    int  used;
    int  top1;  /* grow upward, relative to &array[0] */
    int  top2;  /* grow downward */
    void *array;
} SuperLU_LU_stack_t;

/* Constants */
#define SuperLU_GluIntArray(n)   (5 * (n) + 5)

#if 0 // defined in superlu_enum_consts.h -- 1/20/2018
#define SuperLU_NO_MEMTYPE  6      /* 0: lusup;
				      1: ucol;
				      2: lsub;
				      3: usub
				      4: llvl; level number in L for ILU(k)
				      5: ulvl; level number in U for ILU(k)
				   */
#endif	  

/* Macros to manipulate stack */
#define SuperLU_StackFull(x)         ( x + stack.used >= stack.size )
#define SuperLU_NotDoubleAlign(addr) ( (long)addr & 7 )
#define SuperLU_DoubleAlign(addr)    ( ((long)addr + 7) & ~7L )
#define SuperLU_TempSpace(n, w)      ( (2*w + 4 + NO_MARKER)*m*sizeof(int) + \
			      (w + 1)*n*sizeof(double) )
#define SuperLU_Reduce(alpha)        ((alpha + 1) / 2)  /* i.e. (alpha-1)/2 + 1 */

#define SuperLU_FIRSTCOL_OF_SNODE(i)	(xsup[i])

#if ( PROFlevel>=1 )
#define TIC(t)          t = SuperLU_timer_()
#define TOC(t2, t1)     t2 = SuperLU_timer_() - t1
#else
#define TIC(t)
#define TOC(t2, t1)
#endif

/*********************************************************
 * Macros used for easy access of sparse matrix entries. *
 *********************************************************/
#define SuperLU_L_SUB_START(col)     ( Lstore->rowind_colptr[col] )
#define SuperLU_L_SUB(ptr)           ( Lstore->rowind[ptr] )
#define SuperLU_L_NZ_START(col)      ( Lstore->nzval_colptr[col] )
#define SuperLU_L_FST_SUPC(superno)  ( Lstore->sup_to_col[superno] )
#define SuperLU_U_NZ_START(col)      ( Ustore->colptr[col] )
#define SuperLU_U_SUB(ptr)           ( Ustore->rowind[ptr] )

/***********************************************************************
 * For 3D code */
/* SCT_t was initially Schur-complement counter to compute different 
   metrics of Schur-complement Update.
   Later, it includes counters to keep track of many other metrics.
*/
typedef struct
{
    int_t datatransfer_count;
    int_t schurPhiCallCount;
    int_t PhiMemCpyCounter;
    double acc_load_imbal;
    double LookAheadGEMMFlOp;
    double PhiWaitTimer_2;
    double LookAheadGEMMTimer;
    double LookAheadRowSepTimer;
    double LookAheadScatterTimer;
    double GatherTimer ;
    double GatherMOP ;
    double scatter_mem_op_counter;
    double LookAheadRowSepMOP  ;
    double scatter_mem_op_timer;
    double schur_flop_counter;
    double schur_flop_timer;
    double CPUOffloadTimer;
    double PhiWaitTimer;
    double NetSchurUpTimer;
    double AssemblyTimer;
    double PhiMemCpyTimer;
    double datatransfer_timer;
    double LookAheadScatterMOP;
    double schurPhiCallTimer;
    double autotunetime;
    double *Predicted_acc_sch_time;
    double *Predicted_acc_gemm_time;
    double *Predicted_acc_scatter_time;

    double trf2_flops;
    double trf2_time;
    double offloadable_flops;   /*flops that can be done on ACC*/
    double offloadable_mops;    /*mops that can be done on ACC*/

    double *SchurCompUdtThreadTime;
    double *Predicted_host_sch_time;
    double *Measured_host_sch_time;

#ifdef SCATTER_PROFILE
    double *Host_TheadScatterMOP ;
    double *Host_TheadScatterTimer;
#endif

#ifdef OFFLOAD_PROFILE
    double *Predicted_acc_scatter_time_strat1;
    double *Predicted_host_sch_time_strat1;
    size_t pci_transfer_count[18];  /*number of transfers*/
    double pci_transfer_time[18];   /*time for each transfer */
    double pci_transfer_prediction_error[18];   /*error in prediction*/
    double host_sch_time[24][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING]; /**/
    double host_sch_flop[24][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING]; /**/
#endif

    double pdgstrs2_timer;
    double pdgstrf2_timer;
    double lookaheadupdatetimer;
    double pdgstrfTimer;

// new timers for different wait times
    //convention:  tl suffix  refers to times measured from rdtsc
    // td : suffix refers to times measured in SuerpLU_timer

    /* diagonal block factorization; part of pdgstrf2; called from thread*/
    // double Local_Dgstrf2_tl; 
    double *Local_Dgstrf2_Thread_tl;      
    /*wait for receiving U diagonal block: part of mpf*/
    double Wait_UDiagBlock_Recv_tl;
    /*wait for receiving L diagonal block: part of mpf*/
    double Wait_LDiagBlock_Recv_tl;
    

    /*Wait for U diagnal bloc kto receive; part of pdgstrf2 */
    double Recv_UDiagBlock_tl;
    /*wait for previous U block send to finish; part of pdgstrf2 */
    double Wait_UDiagBlockSend_tl;  
    /*after obtaining U block, time spent in calculating L panel*/
    double L_PanelUpdate_tl;
    /*Synchronous Broadcasting L and U panel*/
    double Bcast_UPanel_tl;
    double Bcast_LPanel_tl;
    /*Wait for L send to finish */
    double Wait_LSend_tl;

    /*Wait for U send to finish */
    double Wait_USend_tl;
    /*Wait for U receive */
    double Wait_URecv_tl;
    /*Wait for L receive */
    double Wait_LRecv_tl;

    /*time to get lock*/
    double *GetAijLock_Thread_tl;

    /*U_panelupdate*/
    double PDGSTRS2_tl;

    /*profiling by phases */
    double Phase_Factor_tl;
    double Phase_LU_Update_tl;
    double Phase_SC_Update_tl;
    
    /*3D timers*/
    double ancsReduce;  /*timer for reducing ancestors before factorization*/
    double gatherLUtimer; /*timer for gather LU factors into bottom layer*/
    double tFactor3D[MAX_3D_LEVEL];
    double tSchCompUdt3d[MAX_3D_LEVEL];

    /*ASync Profiler timing*/
    double tAsyncPipeTail;

    /*double t_Startup time before factorization starts*/
    double tStartup;

    /*keeping track of data sent*/
    double commVolFactor;
    double commVolRed;

} SCT_t;

#endif /* __SUPERLU_DIST_UTIL */

