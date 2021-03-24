#include <algorithm>
#include "lupanels.hpp"


LUstruct_v100::LUstruct_v100(int_t nsupers_, int_t ldt_,
            int_t *isNodeInMyGrid_,
            LUstruct_t *LUstruct,
            gridinfo3d_t *grid3d_in,
            SCT_t *SCT_, superlu_dist_options_t *options_, 
            SuperLUStat_t *stat_) : isNodeInMyGrid(isNodeInMyGrid_), 
            nsupers(nsupers_), ldt(ldt_), grid3d(grid3d_in), 
            SCT(SCT_), options(options_), stat(stat_)
{

    grid = &(grid3d->grid2d);
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    mycol = MYCOL(iam, grid);
    xsup = LUstruct->Glu_persist->xsup;
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    double **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;
    double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    lPanelVec = new lpanel_t[CEILING(nsupers, Pc)];
    uPanelVec = new upanel_t[CEILING(nsupers, Pr)];
    // create the lvectors
    maxLvalCount =0;
    maxLidxCount =0;
    maxUvalCount =0;
    maxUidxCount =0;
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        int_t k0 = i * Pc + mycol;
        if (Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[k0] == 1)
        {
            int_t isDiagIncluded = 0;
            
            if (myrow == krow(k0))
                isDiagIncluded = 1;
            lpanel_t lpanel(k0, Lrowind_bc_ptr[i], Lnzval_bc_ptr[i], xsup, isDiagIncluded);
            lPanelVec[i] = lpanel;
            maxLvalCount = std::max(lPanelVec[i].nzvalSize(),maxLvalCount );
            maxLidxCount = std::max(lPanelVec[i].indexSize(),maxLidxCount );
        }
    }

    // create the vectors
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            upanel_t upanel(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
            uPanelVec[i] = upanel;
            maxUvalCount = std::max(uPanelVec[i].nzvalSize(),maxUvalCount );
            maxUidxCount = std::max(uPanelVec[i].indexSize(),maxUidxCount );
        }
    }

    // Allocate bigV, indirect
    nThreads = getNumThreads(iam);
    bigV = dgetBigV(ldt, nThreads);
    indirect = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));
    indirectRow = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));
    indirectCol = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));


    // allocating communication buffers 
    LvalRecvBufs.resize(options->num_lookaheads);
    UvalRecvBufs.resize(options->num_lookaheads);
    LidxRecvBufs.resize(options->num_lookaheads);
    UidxRecvBufs.resize(options->num_lookaheads);

    for(int_t i=0; i<options->num_lookaheads; i++)
    {
        LvalRecvBufs[i] = (double*) SUPERLU_MALLOC(sizeof(double)*maxLvalCount);
        UvalRecvBufs[i] = (double*) SUPERLU_MALLOC(sizeof(double)*maxUvalCount);
        LidxRecvBufs[i] = (int_t*) SUPERLU_MALLOC(sizeof(int_t)*maxLidxCount);
        UidxRecvBufs[i] = (int_t*) SUPERLU_MALLOC(sizeof(int_t)*maxUidxCount);
    }

    
}

int_t LUstruct_v100::dSchurComplementUpdate(int_t k, lpanel_t &lpanel, upanel_t &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        /* code */
        int_t thread_id;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#else
        thread_id = 0;
#endif

        double *V = bigV + thread_id * ldt * ldt;
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;
        double alpha = 1.0;
        double beta = 0.0;
        superlu_dgemm("N", "N",
                      lpanel.nbrow(ii), upanel.nbcol(jj), supersize(k), alpha,
                      lpanel.blkPtr(ii), lpanel.LDA(),
                      upanel.blkPtr(jj), upanel.LDA(), beta,
                      V, lpanel.nbrow(ii));

        // now do the scatter
        int_t ib = lpanel.gid(ii);
        int_t jb = upanel.gid(jj);

        dScatter(lpanel.nbrow(ii), upanel.nbcol(jj),
                 ib, jb, V, lpanel.nbrow(ii),
                 lpanel.rowList(ii), upanel.colList(jj));
    }

    return 0;
}

// should be called from an openMP region
int_t *LUstruct_v100::computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        return srcVec;
    }
    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = indirect + thread_id * ldt;
    for (int_t i = 0; i < dstLen; i++)
    {
        dstIdx[dstVec[i]] = i;
    }

    int_t *RCmap = (direction == ROW_MAP) ? indirectRow : indirectCol;
    RCmap += thread_id * ldt;

    for (int_t i = 0; i < srcLen; i++)
    {
        RCmap[i] = dstIdx[srcVec[i]];
    }

    return RCmap;
}

int_t LUstruct_v100::dScatter(int_t m, int_t n,
                              int_t gi, int_t gj,
                              double *Src, int_t ldsrc,
                              int_t *srcRowList, int_t *srcColList)
{

    double *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    if (gj > gi) // its in upanel
    {
        int li = g2lRow(gi);
        int lj = uPanelVec[li].find(gj);
        Dst = uPanelVec[li].blkPtr(lj);
        lddst = supersize(gi);
        dstRowLen = supersize(gi);
        dstRowList = NULL;
        dstColLen = uPanelVec[li].nbcol(lj);
        dstColList = uPanelVec[li].colList(lj);
    }
    else
    {
        int lj = g2lCol(gj);
        int li = lPanelVec[lj].find(gi);
        Dst = lPanelVec[lj].blkPtr(li);
        lddst = lPanelVec[lj].LDA();
        dstRowLen = lPanelVec[lj].nbrow(li);
        dstRowList = lPanelVec[lj].rowList(li);
        dstColLen = supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int_t *rowS2D = computeIndirectMap(ROW_MAP, m, srcRowList,
                                       dstRowLen, dstRowList);
    // compute source col to dest col mapping
    int_t *colS2D = computeIndirectMap(COL_MAP, n, srcColList,
                                       dstColLen, dstColList);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

int_t LUstruct_v100::packedU2skyline(LUstruct_t *LUstruct)
{

    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            uPanelVec[i].packed2skyline(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
        }
    }
}
