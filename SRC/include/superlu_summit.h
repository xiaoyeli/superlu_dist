#pragma once 

#include "superlu_ddefs.h"



int_t pdgstrf3d_v100(superlu_dist_options_t *options, int m, int n, double anorm,
		dtrf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		dLUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info);

#ifdef __cplusplus
extern "C" {
#endif

struct LUstruct_v100;
typedef struct LUstruct_v100* LUgpu_Handle; 

extern LUgpu_Handle createLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                  dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                  SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                  double thresh_, int *info_); 

extern void destroyLUgpuHandle(LUgpu_Handle LuH);

extern int dgatherFactoredLU3Dto2D(LUgpu_Handle LuH);

extern int copyLUGPU2Host(LUgpu_Handle LuH, dLUstruct_t *LUstruct);

extern int pdgstrf3d_LUpackedInterface( LUgpu_Handle LUHand);

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
