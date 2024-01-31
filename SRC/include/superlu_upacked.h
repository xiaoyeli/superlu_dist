#pragma once

#include "superlu_ddefs.h"
#include "superlu_zdefs.h"
#include "superlu_sdefs.h"

int_t pdgstrf3d_v100(superlu_dist_options_t *options, int m, int n, double anorm,
                     dtrf3Dpartition_t *trf3Dpartition, SCT_t *SCT,
                     dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                     SuperLUStat_t *stat, int *info);

#ifdef __cplusplus
extern "C"
{
#endif

    // Left for backward compatibility
    struct LUstruct_v100;
    typedef struct LUstruct_v100 *LUgpu_Handle;

    extern LUgpu_Handle createLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                                          dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                          SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                          double thresh_, int *info_);

    extern void destroyLUgpuHandle(LUgpu_Handle LuH);

    extern int dgatherFactoredLU3Dto2D(LUgpu_Handle LuH);

    extern int copyLUGPU2Host(LUgpu_Handle LuH, dLUstruct_t *LUstruct);

    extern int pdgstrf3d_LUpackedInterface(LUgpu_Handle LUHand);

    struct dLUstruct_v1;
    typedef struct dLUstruct_v1 *dLUgpu_Handle;

    dLUgpu_Handle dCreateLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat, double thresh_, int *info_);

    void dDestroyLUgpuHandle(dLUgpu_Handle LuH);

    int dGatherFactoredLU3Dto2D(dLUgpu_Handle LuH);

    int dCopyLUGPU2Host(dLUgpu_Handle LuH, dLUstruct_t *LUstruct);

    int pdgstrf3d_LUv1(dLUgpu_Handle LUHand);

    // Forward declaration of structs 
    // Forward declarations
// struct strf3Dpartition_t;
// typedef struct strf3Dpartition_t strf3Dpartition_t;
// struct sLUstruct_t;
// typedef struct sLUstruct_t sLUstruct_t;
struct ctrf3Dpartition_t;
typedef struct ctrf3Dpartition_t ctrf3Dpartition_t;
struct cLUstruct_t;
typedef struct cLUstruct_t cLUstruct_t;
// struct ztrf3Dpartition_t;
// typedef struct ztrf3Dpartition_t ztrf3Dpartition_t;
// struct zLUstruct_t;
// typedef struct zLUstruct_t zLUstruct_t;

//  Define the single precision real interface
    struct sLUstruct_v1;
    typedef struct sLUstruct_v1 *sLUgpu_Handle;

    // extern sLUgpu_Handle sCreateLUgpuHandle(int_t nsupers, int_t ldt_, 
    // strf3Dpartition_t *trf3Dpartition,
    //                                        sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    //                                        SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
    //                                        double thresh_, int *info_);
    extern sLUgpu_Handle sCreateLUgpuHandle(int_t nsupers, int_t ldt_, strf3Dpartition_t *trf3Dpartition,
                                     sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     float thresh_, int *info_);

    extern void sdestroyLUgpuHandle(sLUgpu_Handle LuH);

    extern int sgatherFactoredLU3Dto2D(sLUgpu_Handle LuH);

    extern int scopyLUGPU2Host(sLUgpu_Handle LuH, sLUstruct_t *LUstruct);

    extern int psgstrf3d_LUpackedInterface(sLUgpu_Handle LUHand);

    // struct cLUstruct_v1;
    // typedef struct cLUstruct_v1 *cLUgpu_Handle;

    // extern cLUgpu_Handle cCreateLUgpuHandle(int_t nsupers, int_t ldt_, ctrf3Dpartition_t *trf3Dpartition,
    //                                        cLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    //                                        SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
    //                                        float thresh_, int *info_);

    // extern void cdestroyLUgpuHandle(cLUgpu_Handle LuH);

    // extern int cgatherFactoredLU3Dto2D(cLUgpu_Handle LuH);

    // extern int ccopyLUGPU2Host(cLUgpu_Handle LuH, cLUstruct_t *LUstruct);

    // extern int pcgstrf3d_LUpackedInterface(cLUgpu_Handle LUHand);

    struct zLUstruct_v1;
    typedef struct zLUstruct_v1 *zLUgpu_Handle;

    extern zLUgpu_Handle zCreateLUgpuHandle(int_t nsupers, int_t ldt_, ztrf3Dpartition_t *trf3Dpartition,
                                           zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                           SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                           double thresh_, int *info_);

    extern void zdestroyLUgpuHandle(zLUgpu_Handle LuH);

    extern int zgatherFactoredLU3Dto2D(zLUgpu_Handle LuH);

    extern int zcopyLUGPU2Host(zLUgpu_Handle LuH, zLUstruct_t *LUstruct);

    extern int pzgstrf3d_LUpackedInterface(zLUgpu_Handle LUHand);

    struct BatchFactorizeWorkspace;
    typedef struct BatchFactorizeWorkspace* BatchFactorize_Handle;

    extern int dsparseTreeFactorBatchGPU(BatchFactorize_Handle ws, sForest_t *sforest);

    extern BatchFactorize_Handle getBatchFactorizeWorkspace(
        int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, 
        gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
    );

    extern void copyGPULUDataToHost(
        BatchFactorize_Handle ws, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
        SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
    );

    extern void freeBatchFactorizeWorkspace(BatchFactorize_Handle ws);

#ifdef __cplusplus
}
#endif
