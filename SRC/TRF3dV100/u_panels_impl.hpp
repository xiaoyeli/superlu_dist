#pragma once 
#include "lupanels.hpp"

template <typename Ftype>
xupanel_t<Ftype>::xupanel_t(int_t k, int_t *usub, Ftype *uval, int_t *xsup)
{
    int_t kSupSz = SuperSize(k);
    int_t kLastRow = xsup[k + 1];
    /* compute number of columns */
    int_t nonZeroCols = 0;
    int_t usubPtr = BR_HEADER;
    int_t nub = usub[0];

    for (int_t ub = 0; ub < nub; ub++)
    {
        int_t gblockId = usub[usubPtr];
        int_t gsupc = SuperSize(gblockId);
        for (int_t col = 0; col < gsupc; col++)
        {
            int_t segsize = kLastRow - usub[usubPtr + UB_DESCRIPTOR + col];
            if (segsize)
                nonZeroCols++;
        }

        usubPtr += UB_DESCRIPTOR + gsupc;
    }

    int_t uIndexSize = UPANEL_HEADER_SIZE + 2 * nub + 1 + nonZeroCols;
    //Allocating the index and val
    index = (int_t*) SUPERLU_MALLOC(sizeof(int_t) * uIndexSize);
    val = (Ftype *)SUPERLU_MALLOC(sizeof(Ftype) * nonZeroCols * kSupSz);
    index[0] = nub;
    index[1] = nonZeroCols;
    index[2] = kSupSz;
    index[UPANEL_HEADER_SIZE + nub] = 0; // starting of prefix sum is zero
    // now start the loop
    int_t blkIdPtr   = UPANEL_HEADER_SIZE;
    int_t pxSumPtr   = UPANEL_HEADER_SIZE + nub + 1;
    int_t colIdxPtr  = UPANEL_HEADER_SIZE + 2 * nub + 1;
    int_t srcUvalPtr = 0;
    int_t dstUvalPtr = 0;
    // reset the USUB ptr
    usubPtr = BR_HEADER;
    for (int_t ub = 0; ub < nub; ub++)
    {
        int_t gblockId = usub[usubPtr];
        index[blkIdPtr++] = gblockId;
        int_t local_nzcols = 0;
        int_t gsupc = SuperSize(gblockId);
        for (int_t col = 0; col < gsupc; col++)
        {
            int_t segsize = kLastRow - usub[usubPtr + UB_DESCRIPTOR + col];
            if (segsize)
            {
                for(int row=0; row<kSupSz; row++)
                {
                    if(row<kSupSz-segsize)
                        val[dstUvalPtr++] = zeroT<Ftype>();
                    else 
                        val[dstUvalPtr++] =uval[srcUvalPtr++];
                }
                
                index[colIdxPtr++] = col; 
                local_nzcols++;
            }
        }
        index[pxSumPtr] = index[pxSumPtr - 1] + local_nzcols;
        pxSumPtr++;
        usubPtr += UB_DESCRIPTOR + gsupc;
    }

    return;
}

template <typename Ftype>
int_t xupanel_t<Ftype>::packed2skyline(int_t k, int_t *usub, Ftype *uval, int_t*xsup)
{
    int_t kSupSz = SuperSize(k);
    int_t kLastRow = xsup[k + 1];
    int_t srcUvalPtr = 0;
    int_t dstUvalPtr = 0;
    // reset the USUB ptr
    int_t usubPtr = BR_HEADER;
    int_t nub = nblocks();

    for (int_t ub = 0; ub < nub; ub++)
    {
        int_t gblockId = usub[usubPtr];
        int_t gsupc = SuperSize(gblockId);
        for (int_t col = 0; col < gsupc; col++)
        {
            int_t segsize = kLastRow - usub[usubPtr + UB_DESCRIPTOR + col];
            if (segsize)
            {
                for(int row=0; row<kSupSz; row++)
                {
                    if(row<kSupSz-segsize)
                        dstUvalPtr++;
                    else 
                        uval[srcUvalPtr++] =val[dstUvalPtr++];
                }
                
            }
        }
        
        usubPtr += UB_DESCRIPTOR + gsupc;
    }   
    return 0;
}

template <typename Ftype>
int_t xupanel_t<Ftype>::find(int_t k)
{
    //TODO: possible to optimize
    for (int_t i = 0; i < nblocks(); i++)
    {
        if (k == gid(i))
            return i;
    }
    //TODO: it shouldn't come here
    return GLOBAL_BLOCK_NOT_FOUND;
}
template <typename Ftype>
int_t xupanel_t<Ftype>::panelSolve(int_t ksupsz, Ftype *DiagBlk, int_t LDD)
{
    if (isEmpty()) return 0;
    
    superlu_trsm<Ftype>("L", "L", "N", "U",
                  ksupsz, nzcols(), one<Ftype>(), DiagBlk, LDD, val, LDA());
    return 0;
}

template <typename Ftype>
int xupanel_t<Ftype>::getEndBlock(int iSt, int maxCols)
{
    int nlb = nblocks();
    if(iSt >= nlb )
        return nlb; 
    int iEnd = iSt; 
    int ii = iSt +1;

    while (
        stCol(ii) - stCol(iSt) <= maxCols &&
        ii < nlb)
        ii++;

#if 1
    if (stCol(ii) - stCol(iSt) > maxCols)
        iEnd = ii-1;
    else 
        iEnd =ii; 
#else 
    if (ii == nlb)
    {
        if (stCol(ii) - stCol(iSt) <= maxCols)
            iEnd = nlb;
        else
            iEnd = nlb - 1;
    }
    else
        iEnd = ii - 1;
#endif 
    return iEnd; 
}