#pragma once 

#include "superlu_ddefs.h"

int_t pdgstrf3d_v100(superlu_dist_options_t *options, int m, int n, double anorm,
		trf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		LUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info);

