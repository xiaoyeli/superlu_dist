#pragma once 

#include "superlu_ddefs.h"

int_t pdgstrf3d_v100(superlu_dist_options_t *options, int m, int n, double anorm,
		trf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		LUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info);

int_t dsparseTreeFactor_v100(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    dLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
);