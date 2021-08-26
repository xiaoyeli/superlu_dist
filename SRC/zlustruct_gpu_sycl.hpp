
/*! @file
 * \brief Descriptions and declarations for structures used in GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * March 14, 2021 version 7.0.0
 * </pre>
 */

#pragma once // so that this header file is included onle once

#include "superlu_zdefs.h"

#if defined(HAVE_SYCL) // enable GPU

// #include "mkl.h"

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

// #include "sec_structs.h"
// #include "supernodal_etree.h"

/* Constants */
//#define SLU_TARGET_GPU 0
//#define MAX_BLOCK_SIZE 10000
#define MAX_NCUDA_STREAMS 32

//typedef std::complex<double> doublecomplex;

typedef struct //SCUbuf_gpu_
{
    /*Informations for various buffers*/
    doublecomplex *bigV;
    doublecomplex *bigU;
    doublecomplex *bigU_host;      /*pinned location*/
    int_t *indirect;        /*for indirect address calculations*/
    int_t *indirect2;       /*for indirect address calculations*/

    doublecomplex *Remain_L_buff;  /* on GPU */
    doublecomplex *Remain_L_buff_host; /* Sherry: this memory is page-locked, why need another copy on GPU ? */

    int_t *lsub;
    int_t *usub;

    int_t *lsub_buf, *usub_buf;

    Ublock_info_t *Ublock_info; /* on GPU */
    Remain_info_t *Remain_info;
    Ublock_info_t *Ublock_info_host;
    Remain_info_t *Remain_info_host;

    int_t* usub_IndirectJ3;  /* on GPU */
    int_t* usub_IndirectJ3_host;

} zSCUbuf_gpu_t;


typedef struct //LUstruct_gpu_
{
    int_t   *LrowindVec;      /* A single vector */
    int_t   *LrowindPtr;      /* A single vector */

    doublecomplex  *LnzvalVec;       /* A single vector */
    int_t   *LnzvalPtr;       /* A single vector */
    int_t   *LnzvalPtr_host;  /* A single vector */

    int_t   *UrowindVec;            /* A single vector */
    int_t   *UrowindPtr;            /* A single vector */
    int_t   *UrowindPtr_host;       /* A single vector */
    int_t   *UnzvalPtr_host;

    doublecomplex  *UnzvalVec;       /* A single vector */
    int_t   *UnzvalPtr;      /* A single vector */
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
    zSCUbuf_gpu_t scubufs[MAX_NCUDA_STREAMS];
    doublecomplex *acc_L_buff, *acc_U_buff;

    /*Informations for various buffers*/
    int_t buffer_size;      /**/
    int_t nsupers;  /*should have number of supernodes*/
    int_t *xsup;
    gridinfo_t *grid;


    double ScatterMOPCounter;
    double ScatterMOPTimer;
    double GemmFLOPCounter;
    double GemmFLOPTimer;

    double cPCIeH2D;
    double cPCIeD2H;
    double tHost_PCIeH2D;
    double tHost_PCIeD2H;

    int *isOffloaded;  /*stores if any iteration is offloaded or not*/

    /*sycl events to measure DGEMM and SCATTER timing */
    sycl::event *GemmStart, *GemmEnd, *ScatterEnd;  /*sycl events to store gemm and scatter's begin and end*/
    sycl::event *ePCIeH2D;
    sycl::event *ePCIeD2H_Start;
    sycl::event *ePCIeD2H_End;

    int_t *xsup_host;
    int_t* perm_c_supno;
    int_t first_l_block_gpu, first_u_block_gpu;
} zLUstruct_gpu_t;

typedef struct //sluGPU_t_
{
    int_t gpuId;        // if there are multiple GPUs
    zLUstruct_gpu_t *A_gpu, *dA_gpu;

    sycl::queue *funCallStreams[MAX_NCUDA_STREAMS], *CopyStream;

    int_t lastOffloadStream[MAX_NCUDA_STREAMS];
    int_t nCudaStreams;
    int_t* isNodeInMyGrid;
    double acc_async_cost;
} zsluGPU_t;


#ifdef __cplusplus
extern "C" {
#endif

extern int zsparseTreeFactor_ASYNC_GPU(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests,
                               // size = maxEtree level
    zscuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,          // size = num Look ahead
    zLUValSubBuf_t **LUvsbs, // size = num Look ahead
    zdiagFactBufs_t **dFBufs, // size = maxEtree level
    factStat_t *factStat,
    factNodelists_t *fNlists,
    gEtreeInfo_t *gEtreeInfo, // global etree info
    superlu_dist_options_t *options,
    int_t *gIperm_c_supno,
    int ldt,
    zsluGPU_t *sluGPU,
    d2Hreduce_t *d2Hred,
    HyP_t *HyP,
    zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    SuperLUStat_t *stat,
    double thresh, SCT_t *SCT, int tag_ub,
    int *info);

int zinitD2Hreduce(
    int next_k,
    d2Hreduce_t* d2Hred,
    int last_flag,
    // int_t *perm_c_supno,
    HyP_t* HyP,
    zsluGPU_t *sluGPU,
    gridinfo_t *grid,
    zLUstruct_t *LUstruct, SCT_t* SCT
);

extern int zreduceGPUlu(int last_flag, d2Hreduce_t* d2Hred,
   	zsluGPU_t *sluGPU, SCT_t *SCT, gridinfo_t *grid,
	 zLUstruct_t *LUstruct);

extern int zwaitGPUscu(int streamId, zsluGPU_t *sluGPU, SCT_t *SCT);
extern int zsendLUpanelGPU2HOST( int_t k0, d2Hreduce_t* d2Hred, zsluGPU_t *sluGPU);
extern int zsendSCUdataHost2GPU(
    int_t streamId, int_t* lsub, int_t* usub, doublecomplex* bigU, int_t bigu_send_size,
    int_t Remain_lbuf_send_size,  zsluGPU_t *sluGPU, HyP_t* HyP
);

extern int zinitSluGPU3D_t(
    zsluGPU_t *sluGPU,
    zLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno, int_t n, int_t buffer_size, int_t bigu_size, int_t ldt
);
int zSchurCompUpdate_GPU(
    int_t streamId,
    int_t jj_cpu, int_t nub, int_t klst, int_t knsupc,
    int_t Rnbrow, int_t RemainBlk,
    int_t Remain_lbuf_send_size,
    int_t bigu_send_size, int_t ldu,
    int_t mcb,
    int_t buffer_size, int_t lsub_len, int_t usub_len,
    int_t ldt, int_t k0,
    zsluGPU_t *sluGPU, gridinfo_t *grid
);


extern void zCopyLUToGPU3D (int_t* isNodeInMyGrid, zLocalLU_t *A_host,
           zsluGPU_t *sluGPU, Glu_persist_t *Glu_persist, int_t n,
	   gridinfo3d_t *grid3d, int_t buffer_size, int_t bigu_size, int_t ldt);

extern int zreduceAllAncestors3d_GPU(int_t ilvl, int_t* myNodeCount,
                              int_t** treePerm,    zLUValSubBuf_t*LUvsb,
                              zLUstruct_t* LUstruct, gridinfo3d_t* grid3d,
                              zsluGPU_t *sluGPU,  d2Hreduce_t* d2Hred,
                              factStat_t *factStat, HyP_t* HyP, SCT_t* SCT );

extern void zsyncAllfunCallStreams(zsluGPU_t* sluGPU, SCT_t* SCT);
extern int zfree_LUstruct_gpu (zLUstruct_gpu_t *A_gpu
			       , zsluGPU_t *sluGPU
    );

//int freeSluGPU(zsluGPU_t *sluGPU);

extern void zPrint_matrix( char *desc, int_t m, int_t n, doublecomplex *dA, int_t lda );

/*to print out various statistics*/
void zprintGPUStats(zLUstruct_gpu_t *A_gpu);

#ifdef __cplusplus
}
#endif

#endif // matching: enable GPU
