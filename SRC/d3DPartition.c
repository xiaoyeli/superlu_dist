#include "superlu_ddefs.h"

sForest_t **compute_sForests(int_t nsupers, options_t options, LUstruct_t *LUstruct, grid3d_t *grid3d) {
    // Calculation of supernodal etree
    int_t *setree = supernodal_etree(nsupers, LUstruct->etree, LUstruct->Glu_persist->supno, LUstruct->Glu_persist->xsup);

    // Conversion of supernodal etree to list
    treeList_t *treeList = setree2list(nsupers, setree);

    // Calculation of tree weight
    calcTreeWeight(nsupers, setree, treeList, LUstruct->Glu_persist->xsup);

    // Calculation of maximum level
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    // Generation of forests
    sForest_t **sForests = getForests(maxLvl, nsupers, setree, treeList);

    // Allocate trf3d data structure
    LUstruct->trf3Dpart = (trf3Dpartition_t *) SUPERLU_MALLOC(sizeof(trf3Dpartition_t));
    trf3Dpartition_t *trf3Dpart = LUstruct->trf3Dpart;
    

    return sForests;
}
