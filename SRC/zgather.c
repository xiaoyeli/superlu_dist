/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Various gather routines.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 */
#include <stdio.h>
#include "superlu_zdefs.h"
#if 0
#include "scatter.h"
#include "sec_structs.h"
#include "superlu_defs.h"
#include "gather.h"
#endif

int_t zprintMatrix(char*s, int n, int m, doublecomplex* A, int LDA)
{
    printf("%s\n", s );
    for(int i=0; i<n; i++)
    {
        for(int j =0; j<m; j++)
        {
            printf("%g %g\n", A[j*LDA +i].r, A[j*LDA +i].r);
        }
        printf("\n");
    }
    return 0;
}

void zgather_u(int_t num_u_blks,
                Ublock_info_t *Ublock_info, int_t * usub,
              	doublecomplex *uval, doublecomplex *bigU, int_t ldu,
              	int_t *xsup, int_t klst                /* for SuperSize */
             )
{
    // return;
    //  private(j,iukp,rukp,tempu, jb, nsupc,ljb,segsize,lead_zero, \
    // jj, i)
    doublecomplex zero = {0.0, 0.0};

#ifdef _OPENMP    
#pragma omp parallel for default (shared) schedule(dynamic)
#endif
    for (int_t j = 0; j < num_u_blks; ++j)
    {
        doublecomplex *tempu;
        if (j == 0) tempu = bigU;
        else tempu = bigU + ldu * Ublock_info[j - 1].full_u_cols;

        int_t iukp = Ublock_info[j].iukp ;
        int_t rukp = Ublock_info[j].rukp ;
        int_t jb = Ublock_info[j].jb ;
        int_t nsupc = SuperSize(jb);

        for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
        {
            int_t segsize = klst - usub[jj];
            if ( segsize )
            {
                int_t lead_zero = ldu - segsize;
                for (int_t i = 0; i < lead_zero; ++i) tempu[i] = zero;
                tempu += lead_zero;
                for (int_t i = 0; i < segsize; ++i)
                {
                    // printf("%d %d %d\n",i,rukp,segsize );
                    tempu[i] = uval[rukp + i];
                }
                rukp += segsize;
                tempu += segsize;
            }
        }
        rukp -= usub[iukp - 1]; /* Return to start of U(k,j). */
    }
}


void zgather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               doublecomplex * lval, int_t LD_lval, doublecomplex * L_buff )
{
    if (num_LBlk < 1)
    {
        return;
    }

    int_t LD_LBuff = L_info[num_LBlk - 1].FullRow;  /*leading dimension of buffer*/
#ifdef _OPENMP    
#pragma omp parallel for
#endif
    for (int_t i = 0; i < num_LBlk; ++i)
    {
        int_t StRowDest  = 0;
        int_t temp_nbrow;
        if (i == 0)
        {
            temp_nbrow = L_info[0].FullRow;
        }
        else
        {
            StRowDest   = L_info[i - 1].FullRow;
            temp_nbrow  = L_info[i].FullRow - L_info[i - 1].FullRow;
        }

        int_t StRowSource = L_info[i].StRow;
#if 0
        LAPACKE_dlacpy (LAPACK_COL_MAJOR, 'A', temp_nbrow, knsupc, &lval[StRowSource], LD_lval, &L_buff[StRowDest], LD_LBuff);
#else  /* Sherry */
	for (int j = 0; j < knsupc; ++j) {
            memcpy( &L_buff[StRowDest + j * LD_LBuff], 
	            &lval[StRowSource + j * LD_lval],
	            temp_nbrow * sizeof(doublecomplex) );
        }
#endif
    } /* end for i ... */
}

// Rearragnes and gatehrs L blocks
void zRgather_L( int_t k, int_t *lsub, doublecomplex *lusup,
                gEtreeInfo_t* gEtreeInfo, Glu_persist_t *Glu_persist,
                gridinfo_t *grid, HyP_t *HyP, int_t *myIperm, int_t *iperm_c_supno )
{

    int_t temp_nbrow;
    int_t cum_nrow = 0;
    int_t ib;
    int_t *xsup = Glu_persist->xsup;
    int_t knsupc = SuperSize (k);
    int_t krow = PROW (k, grid);
    int_t nlb, lptr0, luptr0;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);

    HyP->lookAheadBlk = 0, HyP->RemainBlk = 0;

    int_t nsupr = lsub[1];  /* LDA of lusup. */
    if (myrow == krow)  /* Skip diagonal block L(k,k). */
    {
        lptr0 = BC_HEADER + LB_DESCRIPTOR + lsub[BC_HEADER + 1];
        luptr0 = knsupc;
        nlb = lsub[0] - 1;
    }
    else
    {
        lptr0 = BC_HEADER;
        luptr0 = 0;
        nlb = lsub[0];
    }
    // printf("nLb =%d ", nlb );

    int_t lptr = lptr0;
    int_t luptr = luptr0;
    for (int_t i = 0; i < nlb; ++i)
    {
        ib = lsub[lptr];        /* Row block L(i,k). */
        temp_nbrow = lsub[lptr + 1]; /* Number of full rows. */

        int_t look_up_flag = 1;

        // if elimination order is greater than first block stored on GPU
        if (iperm_c_supno[ib] < HyP->first_u_block_acc) look_up_flag = 0;

        // if it myIperm[ib] is within look ahead window
        if (myIperm[ib]< myIperm[k] + HyP->nGPUStreams && myIperm[ib]>0) look_up_flag = 0;        

        if (k <= HyP->nsupers - 2 && gEtreeInfo->setree[k] > 0 )
        {
            int_t k_parent = gEtreeInfo->setree[k];
            if (ib == k_parent && gEtreeInfo->numChildLeft[k_parent]==1 )
            {
                look_up_flag = 0;
            }
        }
        // look_up_flag = 0;
        if (!look_up_flag)
        {
            /* ib is within look up window */
            HyP->lookAhead_info[HyP->lookAheadBlk].nrows = temp_nbrow;
            if (HyP->lookAheadBlk == 0)
            {
                HyP->lookAhead_info[HyP->lookAheadBlk].FullRow = temp_nbrow;
            }
            else
            {
                HyP->lookAhead_info[HyP->lookAheadBlk].FullRow
                    = temp_nbrow + HyP->lookAhead_info[HyP->lookAheadBlk - 1].FullRow;
            }
            HyP->lookAhead_info[HyP->lookAheadBlk].StRow = cum_nrow;
            HyP->lookAhead_info[HyP->lookAheadBlk].lptr = lptr;
            HyP->lookAhead_info[HyP->lookAheadBlk].ib = ib;
            HyP->lookAheadBlk++;
        }
        else
        {
            /* ib is not in look up window */
            HyP->Remain_info[HyP->RemainBlk].nrows = temp_nbrow;
            if (HyP->RemainBlk == 0)
            {
                HyP->Remain_info[HyP->RemainBlk].FullRow = temp_nbrow;
            }
            else
            {
                HyP->Remain_info[HyP->RemainBlk].FullRow
                    = temp_nbrow + HyP->Remain_info[HyP->RemainBlk - 1].FullRow;
            }
            HyP->Remain_info[HyP->RemainBlk].StRow = cum_nrow;
            HyP->Remain_info[HyP->RemainBlk].lptr = lptr;
            HyP->Remain_info[HyP->RemainBlk].ib = ib;
            HyP->RemainBlk++;
        }

        cum_nrow += temp_nbrow;

        lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
        lptr += temp_nbrow;
        luptr += temp_nbrow;
    }
    lptr = lptr0;
    luptr = luptr0;

    zgather_l( HyP->lookAheadBlk, knsupc, HyP->lookAhead_info,
               &lusup[luptr], nsupr, HyP->lookAhead_L_buff);

    zgather_l( HyP->RemainBlk, knsupc, HyP->Remain_info,
               &lusup[luptr], nsupr, HyP->Remain_L_buff);

    assert(HyP->lookAheadBlk + HyP->RemainBlk ==nlb );
    HyP->Lnbrow = HyP->lookAheadBlk == 0 ? 0 : HyP->lookAhead_info[HyP->lookAheadBlk - 1].FullRow;
    HyP->Rnbrow = HyP->RemainBlk == 0 ? 0 : HyP->Remain_info[HyP->RemainBlk - 1].FullRow;

    // zprintMatrix("LookAhead Block", HyP->Lnbrow, knsupc, HyP->lookAhead_L_buff, HyP->Lnbrow);
    // zprintMatrix("Remaining Block", HyP->Rnbrow, knsupc, HyP->Remain_L_buff, HyP->Rnbrow);
}

// void Rgather_U(int_t k,
//                 HyP_t *HyP,
//                int_t st, int_t end,
//                int_t *usub, double *uval, double *bigU,
//                Glu_persist_t *Glu_persist, gridinfo_t *grid,
//                int_t *perm_u)

void zRgather_U( int_t k, int_t jj0, int_t *usub,	doublecomplex *uval,
                 doublecomplex *bigU, gEtreeInfo_t* gEtreeInfo,	
                 Glu_persist_t *Glu_persist, gridinfo_t *grid, HyP_t *HyP,
                 int_t* myIperm, int_t *iperm_c_supno, int_t *perm_u)
{
    HyP->ldu   = 0;
    HyP->num_u_blks = 0;
    HyP->ldu_Phi = 0;
    HyP->num_u_blks_Phi = 0;

    int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
    int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
    int_t     nub = usub[0];      /* Number of blocks in the block row U(k,:) */
    int_t *xsup = Glu_persist->xsup;
    // int_t k = perm_c_supno[k0];
    int_t klst = FstBlockC (k + 1);
    int_t iukp0 = iukp;
    int_t rukp0 = rukp;
    int_t jb, ljb;
    int_t nsupc;
    int_t full = 1;
    int_t full_Phi = 1;
    int_t temp_ncols = 0;
    int_t segsize;
    HyP->num_u_blks = 0;
    HyP->ldu = 0;

    for (int_t j = jj0; j < nub; ++j)
    {
        temp_ncols = 0;
        arrive_at_ublock(
            j, &iukp, &rukp, &jb, &ljb, &nsupc,
            iukp0, rukp0, usub, perm_u, xsup, grid
        );

        for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
        {
            segsize = klst - usub[jj];
            if ( segsize ) ++temp_ncols;
        }
        /*here goes the condition wether jb block exists on Phi or not*/
        int_t u_blk_acc_cond = 0;
        // if (j == jj0) u_blk_acc_cond = 1;   /* must schedule first colum on cpu */
        if (iperm_c_supno[jb] < HyP->first_l_block_acc) 
        {
            // printf("k=%d jb=%d got at condition-1:%d, %d \n",k,jb, iperm_c_supno[jb] , HyP->first_l_block_acc);
            u_blk_acc_cond = 1;
        }
        // if jb is within lookahead window
        if (myIperm[jb]< myIperm[k] + HyP->nGPUStreams && myIperm[jb]>0)
        {
            // printf("k=%d jb=%d got at condition-2:%d, %d\n ",k,jb, myIperm[jb] , myIperm[k]);
            u_blk_acc_cond = 1;
        }
 
        if (k <= HyP->nsupers - 2 && gEtreeInfo->setree[k] > 0 )
        {
            int_t k_parent = gEtreeInfo->setree[k];
            if (jb == k_parent && gEtreeInfo->numChildLeft[k_parent]==1 )
            {
                u_blk_acc_cond = 1;
                // printf("k=%d jb=%d got at condition-3\n",k,jb);
                u_blk_acc_cond = 1;
            }
        }


        if (u_blk_acc_cond)
        {
            HyP->Ublock_info[HyP->num_u_blks].iukp = iukp;
            HyP->Ublock_info[HyP->num_u_blks].rukp = rukp;
            HyP->Ublock_info[HyP->num_u_blks].jb = jb;

            for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
            {
                segsize = klst - usub[jj];
                if ( segsize )
                {

                    if ( segsize != HyP->ldu ) full = 0;
                    if ( segsize > HyP->ldu ) HyP->ldu = segsize;
                }
            }

            HyP->Ublock_info[HyP->num_u_blks].ncols = temp_ncols;
            // ncols += temp_ncols;
            HyP->num_u_blks++;
        }
        else
        {
            HyP->Ublock_info_Phi[HyP->num_u_blks_Phi].iukp = iukp;
            HyP->Ublock_info_Phi[HyP->num_u_blks_Phi].rukp = rukp;
            HyP->Ublock_info_Phi[HyP->num_u_blks_Phi].jb = jb;
            HyP->Ublock_info_Phi[HyP->num_u_blks_Phi].eo =  HyP->nsupers - iperm_c_supno[jb]; /*since we want it to be in descending order*/

            /* Prepare to call DGEMM. */


            for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
            {
                segsize = klst - usub[jj];
                if ( segsize )
                {

                    if ( segsize != HyP->ldu_Phi ) full_Phi = 0;
                    if ( segsize > HyP->ldu_Phi ) HyP->ldu_Phi = segsize;
                }
            }

            HyP->Ublock_info_Phi[HyP->num_u_blks_Phi].ncols = temp_ncols;
            // ncols_Phi += temp_ncols;
            HyP->num_u_blks_Phi++;
        }
    }

    /* Now doing prefix sum on  on ncols*/
    HyP->Ublock_info[0].full_u_cols = HyP->Ublock_info[0 ].ncols;
    for (int_t j = 1; j < HyP->num_u_blks; ++j)
    {
        HyP->Ublock_info[j].full_u_cols = HyP->Ublock_info[j ].ncols + HyP->Ublock_info[j - 1].full_u_cols;
    }

    /*sorting u blocks based on elimination order */
    // sort_U_info_elm(HyP->Ublock_info_Phi,HyP->num_u_blks_Phi );
    HyP->Ublock_info_Phi[0].full_u_cols = HyP->Ublock_info_Phi[0 ].ncols;
    for ( int_t j = 1; j < HyP->num_u_blks_Phi; ++j)
    {
        HyP->Ublock_info_Phi[j].full_u_cols = HyP->Ublock_info_Phi[j ].ncols + HyP->Ublock_info_Phi[j - 1].full_u_cols;
    }

    HyP->bigU_Phi = bigU;
    if ( HyP->num_u_blks_Phi == 0 )  // Sherry fix
	HyP->bigU_host = bigU;
    else
	HyP->bigU_host = bigU + HyP->ldu_Phi * HyP->Ublock_info_Phi[HyP->num_u_blks_Phi - 1].full_u_cols;

    zgather_u(HyP->num_u_blks, HyP->Ublock_info, usub, uval, HyP->bigU_host,
               HyP->ldu, xsup, klst );

    zgather_u(HyP->num_u_blks_Phi, HyP->Ublock_info_Phi, usub, uval,
               HyP->bigU_Phi,  HyP->ldu_Phi, xsup, klst );

} /* zRgather_U */
