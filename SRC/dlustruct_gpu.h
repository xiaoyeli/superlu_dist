

/*! @file
 * \brief Descriptions and declarations for structures used in GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * March 14, 2021 version 7.0.0
 *
 * Last update: December 12, 2021  v7.2.0
 * </pre>
 */

#pragma once // so that this header file is included onle once

#include "superlu_ddefs.h"

#ifdef GPU_ACC // enable GPU
#include "gpu_api_utils.h"
// #include "mkl.h"
// #include "sec_structs.h"
// #include "supernodal_etree.h"

/* Constants */
//#define SLU_TARGET_GPU 0
//#define MAX_BLOCK_SIZE 10000
#define MAX_NGPU_STREAMS 32

static
void check(gpuError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "GPU error at file %s: line %d code=(%s) \"%s\" \n",
                file, line, gpuGetErrorString(result), func);

        // Make sure we call GPU Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

#define checkGPUErrors(val)  check ( (val), #val, __FILE__, __LINE__ )

typedef struct //SCUbuf_gpu_
{
    /*Informations for various buffers*/
    double *bigV;
    double *bigU;
    double *bigU_host;      /*pinned location*/
    int_t *indirect;        /*for indirect address calculations*/
    int_t *indirect2;       /*for indirect address calculations*/

    double *Remain_L_buff;  /* on GPU */
    double *Remain_L_buff_host; /* Sherry: this memory is page-locked, why need another copy on GPU ? */
    
    int_t *lsub;
    int_t *usub;

    int_t *lsub_buf, *usub_buf;
    
    Ublock_info_t *Ublock_info; /* on GPU */
    Remain_info_t *Remain_info;
    Ublock_info_t *Ublock_info_host;
    Remain_info_t *Remain_info_host;

    int_t* usub_IndirectJ3;  /* on GPU */
    int_t* usub_IndirectJ3_host;

} dSCUbuf_gpu_t;

/* Holds the L & U data structures on the GPU side */
typedef struct //LUstruct_gpu_ 
{
    int_t   *LrowindVec;      /* A single vector */
    int_t   *LrowindPtr;      /* A single vector */

    double  *LnzvalVec;       /* A single vector */
    int_t   *LnzvalPtr;        /* A single vector */
    int_t   *LnzvalPtr_host;   /* A single vector */

    int_t   *UrowindVec;            /* A single vector */
    int_t   *UrowindPtr;            /* A single vector */
    int_t   *UrowindPtr_host;       /* A single vector */
    int_t   *UnzvalPtr_host;

    double  *UnzvalVec;       /* A single vector */
    int_t   *UnzvalPtr;        /* A single vector */
    
    /*gpu pointers for easy block accesses */
    local_l_blk_info_t *local_l_blk_infoVec;
    int_t *local_l_blk_infoPtr;
    int_t *jib_lookupVec;
    int_t *jib_lookupPtr;
    local_u_blk_info_t *local_u_blk_infoVec;

    int_t *local_u_blk_infoPtr;
    int_t *ijb_lookupVec;
    int_t *ijb_lookupPtr;

    // GPU buffers for performing Schur Complement Update on GPU
    dSCUbuf_gpu_t scubufs[MAX_NGPU_STREAMS];
    double *acc_L_buff, *acc_U_buff;

    /*Informations for various buffers*/
    int_t buffer_size;      /**/
    int_t nsupers;  /*should have number of supernodes*/
    int_t *xsup;
    gridinfo_t *grid;

#if 0 // Sherry: moved to 'SuperLUStat_t'
    double ScatterMOPCounter;
    double ScatterMOPTimer;
    double GemmFLOPCounter;
    double GemmFLOPTimer;

    double cPCIeH2D;
    double cPCIeD2H;
    double tHost_PCIeH2D;
    double tHost_PCIeD2H;

    /*GPU events to measure DGEMM and SCATTER timing */
    int *isOffloaded;  /*stores if any iteration is offloaded or not*/
    gpuEvent_t *GemmStart, *GemmEnd, *ScatterEnd;  /*GPU events to store gemm and scatter's begin and end*/
    gpuEvent_t *ePCIeH2D;
    gpuEvent_t *ePCIeD2H_Start;
    gpuEvent_t *ePCIeD2H_End;
#endif

    int_t *xsup_host;
    int_t* perm_c_supno;
    int_t first_l_block_gpu, first_u_block_gpu;
} dLUstruct_gpu_t;

typedef struct //sluGPU_t_
{
    //int gpuId;      // if there are multiple GPUs ( NOT USED )
    dLUstruct_gpu_t *A_gpu, *dA_gpu; // holds the LU structure on GPU
    gpuStream_t funCallStreams[MAX_NGPU_STREAMS], CopyStream;
    gpublasHandle_t gpublasHandles[MAX_NGPU_STREAMS];
    int lastOffloadStream[MAX_NGPU_STREAMS];
    int nGPUStreams;
    int* isNodeInMyGrid;
    double acc_async_cost;
} dsluGPU_t;


#ifdef __cplusplus
extern "C" {
#endif

extern int dsparseTreeFactor_ASYNC_GPU(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests,
                               // size = maxEtree level
    dscuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,          // size = num Look ahead
    dLUValSubBuf_t **LUvsbs, // size = num Look ahead
    ddiagFactBufs_t **dFBufs, // size = maxEtree level
    factStat_t *factStat,
    factNodelists_t *fNlists,
    gEtreeInfo_t *gEtreeInfo, // global etree info
    superlu_dist_options_t *options,
    int_t *gIperm_c_supno,
    int ldt,
    dsluGPU_t *sluGPU,
    d2Hreduce_t *d2Hred,
    HyP_t *HyP,
    dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, 
    SuperLUStat_t *stat,
    double thresh, SCT_t *SCT, int tag_ub,
    int *info);

int dinitD2Hreduce(
    int next_k,
    d2Hreduce_t* d2Hred,
    int last_flag,
    // int_t *perm_c_supno,
    HyP_t* HyP,
    dsluGPU_t *sluGPU,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct, SCT_t* SCT
);

extern int dreduceGPUlu(int last_flag, d2Hreduce_t* d2Hred,
   	dsluGPU_t *sluGPU, SCT_t *SCT, gridinfo_t *grid,
	 dLUstruct_t *LUstruct);

extern int dwaitGPUscu(int streamId, dsluGPU_t *sluGPU, SCT_t *SCT);
extern int dsendLUpanelGPU2HOST( int_t k0, d2Hreduce_t* d2Hred,
       	   dsluGPU_t *sluGPU, SuperLUStat_t *);
extern int dsendSCUdataHost2GPU(
    int_t streamId, int_t* lsub, int_t* usub, double* bigU, int_t bigu_send_size,
    int_t Remain_lbuf_send_size,  dsluGPU_t *sluGPU, HyP_t* HyP
);

extern int dinitSluGPU3D_t(
    dsluGPU_t *sluGPU,
    dLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno, int_t n, int_t buffer_size, int_t bigu_size, int_t ldt,
    SuperLUStat_t *
);
int dSchurCompUpdate_GPU(
    int_t streamId,
    int_t jj_cpu, int_t nub, int_t klst, int_t knsupc,
    int_t Rnbrow, int_t RemainBlk,
    int_t Remain_lbuf_send_size,
    int_t bigu_send_size, int_t ldu,
    int_t mcb,
    int_t buffer_size, int_t lsub_len, int_t usub_len,
    int_t ldt, int_t k0,
    dsluGPU_t *sluGPU, gridinfo_t *grid,
    SuperLUStat_t *
);


extern void dCopyLUToGPU3D (int* isNodeInMyGrid, dLocalLU_t *A_host,
           dsluGPU_t *sluGPU, Glu_persist_t *Glu_persist, int_t n,
	   gridinfo3d_t *grid3d, int_t buffer_size, int_t bigu_size, int_t ldt,
    	   SuperLUStat_t *
	   );

extern int dreduceAllAncestors3d_GPU(int_t ilvl, int_t* myNodeCount,
                              int_t** treePerm,    dLUValSubBuf_t*LUvsb,
                              dLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
                              dsluGPU_t *sluGPU,  d2Hreduce_t* d2Hred,
                              factStat_t *factStat, HyP_t* HyP, SCT_t* SCT,
    			      SuperLUStat_t *
			      );

extern void dsyncAllfunCallStreams(dsluGPU_t* sluGPU, SCT_t* SCT);
extern int dfree_LUstruct_gpu (dsluGPU_t *sluGPU, SuperLUStat_t *);

//int freeSluGPU(dsluGPU_t *sluGPU);

extern void dPrint_matrix( char *desc, int_t m, int_t n, double *dA, int_t lda );

#ifdef __cplusplus
}
#endif

#endif // matching: enable GPU
