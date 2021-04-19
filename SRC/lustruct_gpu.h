// This file contains descriptions and declarations for structures used
// in GPU
/*also declaration used for GPUs*/
#pragma once // Causes this source file to be included onle once

#define DEBUG
// #ifdef DEBUG
// #include <assert.h>
// #endif
// #include <math.h>
// #include "mkl.h"

// #define USE_VENDOR_BLAS

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include "superlu_ddefs.h"
// #include "sec_structs.h"
// #include "supernodal_etree.h"


#define SLU_TARGET_GPU 0

#define MAX_BLOCK_SIZE 10000

static void check(int result, char const *const func, const char *const file,
                  int_t const line)
{
}

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

typedef struct SCUbuf_gpu_
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

} SCUbuf_gpu_t;


#define MAX_NCUDA_STREAMS 32

typedef struct LUstruct_gpu_ 
{

    int_t   *LrowindVec;      /* A single vector */
    int_t   *LrowindPtr;      /* A single vector */

    double  *LnzvalVec;       /* A single vector */
    int_t   *LnzvalPtr;       /* A single vector */
    int_t   *LnzvalPtr_host;  /* A single vector */

    int_t   *UrowindVec;            /* A single vector */
    int_t   *UrowindPtr;            /* A single vector */
    int_t   *UrowindPtr_host;       /* A single vector */
    int_t   *UnzvalPtr_host;

    double  *UnzvalVec;       /* A single vector */
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
    SCUbuf_gpu_t scubufs[MAX_NCUDA_STREAMS];
    double *acc_L_buff, *acc_U_buff;

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


    /*cuda events to measure DGEMM and SCATTER timing */
    int_t *isOffloaded;       /*stores if any iteration is offloaded or not*/
    sycl::event *GemmStart, *GemmEnd, *ScatterEnd;
    std::chrono::time_point<std::chrono::steady_clock> GemmStart_ct1_k0;
    std::chrono::time_point<std::chrono::steady_clock> GemmEnd_ct1_k0;
    std::chrono::time_point<std::chrono::steady_clock>
        ScatterEnd_ct1_k0; /*cuda events to store gemm and scatter's begin and
                              end*/
    sycl::event *ePCIeH2D;
    std::chrono::time_point<std::chrono::steady_clock> ePCIeH2D_ct1_k0;
    sycl::event *ePCIeD2H_Start;
    std::chrono::time_point<std::chrono::steady_clock> ePCIeD2H_Start_ct1_k0;
    sycl::event *ePCIeD2H_End;
    std::chrono::time_point<std::chrono::steady_clock> ePCIeD2H_End_ct1_k0;

    int_t *xsup_host;
    int_t* perm_c_supno;
    int_t first_l_block_gpu, first_u_block_gpu;
} LUstruct_gpu;


typedef struct sluGPU_t_
{

    int_t gpuId;        // if there are multiple GPUs
    LUstruct_gpu *A_gpu, *dA_gpu;
    sycl::queue *funCallStreams[MAX_NCUDA_STREAMS], *CopyStream;
    sycl::queue *cublasHandles[MAX_NCUDA_STREAMS];
    int_t lastOffloadStream[MAX_NCUDA_STREAMS];
//    int_t nCudaStreams;
    int_t nGpuStreams;
    int_t* isNodeInMyGrid;
    double acc_async_cost;

} sluGPU_t;


#ifdef __cplusplus
extern "C" {
#endif



int_t initD2Hreduce(
    int_t next_k,
    d2Hreduce_t* d2Hred,
    int_t last_flag,
    // int_t *perm_c_supno,
    HyP_t* HyP,
    sluGPU_t *sluGPU,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct
    ,SCT_t* SCT
);

int_t reduceGPUlu(
    
    int_t last_flag,
    d2Hreduce_t* d2Hred,
    sluGPU_t *sluGPU,
    SCT_t *SCT,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct
);

int_t waitGPUscu(int_t streamId, sluGPU_t *sluGPU, SCT_t *SCT);
int_t sendLUpanelGPU2HOST( int_t k0, d2Hreduce_t* d2Hred, sluGPU_t *sluGPU);
int_t sendSCUdataHost2GPU(
    int_t streamId,
    int_t* lsub,
    int_t* usub,
    double* bigU,
    int_t bigu_send_size,
    int_t Remain_lbuf_send_size,
    sluGPU_t *sluGPU,
    HyP_t* HyP
);

int_t initSluGPU3D_t(
    
    sluGPU_t *sluGPU,
    dLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno,
    int_t n,
    int_t buffer_size,
    int_t bigu_size,
    int_t ldt
);
int_t SchurCompUpdate_GPU(
    int_t streamId,
    int_t jj_cpu, int_t nub, int_t klst, int_t knsupc,
    int_t Rnbrow, int_t RemainBlk,
    int_t Remain_lbuf_send_size,
    int_t bigu_send_size, int_t ldu,
    int_t mcb,
    int_t buffer_size, int_t lsub_len, int_t usub_len,
    int_t ldt, int_t k0,
    sluGPU_t *sluGPU, gridinfo_t *grid
);



void CopyLUToGPU3D (
    int_t* isNodeInMyGrid,
    dLocalLU_t *A_host,
    sluGPU_t *sluGPU,
    Glu_persist_t *Glu_persist, int_t n,
    gridinfo3d_t *grid3d,
    int_t buffer_size,
    int_t bigu_size,
    int_t ldt
);

int_t reduceAllAncestors3d_GPU(
    int_t ilvl, int_t* myNodeCount,
    int_t** treePerm,
    dLUValSubBuf_t*LUvsb,
    dLUstruct_t* LUstruct,
    gridinfo3d_t* grid3d,
    sluGPU_t *sluGPU,
    d2Hreduce_t* d2Hred,
    factStat_t *factStat,
    HyP_t* HyP,
    SCT_t* SCT );


void syncAllfunCallStreams(sluGPU_t* sluGPU, SCT_t* SCT);
int_t free_LUstruct_gpu (LUstruct_gpu *A_gpu);

int_t freeSluGPU(sluGPU_t *sluGPU);

int checkCublas(int result);
// cudaError_t checkCuda(cudaError_t result);

void dPrint_matrix( char *desc, int_t m, int_t n, double *dA, int_t lda );

/*to print out various statistics*/
void printGPUStats(LUstruct_gpu *A_gpu);

#ifdef __cplusplus
}
#endif

#undef DEBUG
