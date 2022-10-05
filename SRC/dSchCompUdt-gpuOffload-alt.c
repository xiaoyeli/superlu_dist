/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief This file contains the main loop of pdgstrf which involves
 *        rank k update of the Schur complement.
 *        Uses GPU.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 */

#define SCHEDULE_STRATEGY dynamic

int full;
double gemm_timer = 0.0;
double scatter_timer = 0.0;

if (msg0 && msg2)
{ /* L(:,k) and U(k,:) are not empty. */
    int ldu = 0;
    int full = 1;
    int cum_nrow;
    int temp_nbrow;

    int_t lptr = lptr0;
    int_t luptr = luptr0;

    nbrow = lsub[1];
    if (myrow == krow)
        nbrow = lsub[1] - lsub[3];

    double* L_mat;
    int ldl;

    if (nbrow > 0)
    {

        RemainBlk = 0;
        // get information of lblack
        for (int i = 0; i < nlb; ++i)
        {
            ib = lsub[lptr];             /* Block number of L(i,k). */
            temp_nbrow = lsub[lptr + 1]; /* Number of full rows. */

            if (RemainBlk == 0)
            {
                Remain_info[RemainBlk].FullRow = temp_nbrow;
            }
            else
            {
                Remain_info[RemainBlk].FullRow =
                    temp_nbrow + Remain_info[RemainBlk - 1].FullRow;
            }

            Remain_info[RemainBlk].lptr = lptr;
            Remain_info[RemainBlk].ib = ib;
            RemainBlk++;
            lptr += LB_DESCRIPTOR; /* Skip descriptor. */
            lptr += temp_nbrow;    /* Move to next block */

        } /* for i ... set up pointers for all blocks in L(:,k) */

        int num_u_blks = nub - jj0;
        for (int j = jj0; j < nub; ++j)
        { /* jj0 starts after look-ahead window. */
            temp_ncols = 0;

            arrive_at_ublock(
                j, &iukp, &rukp, &jb, &ljb, &nsupc,
                iukp0, rukp0, usub, perm_u, xsup, grid);

            Ublock_info[j - jj0].iukp = iukp;
            Ublock_info[j - jj0].rukp = rukp;
            Ublock_info[j - jj0].jb = jb;

            /* if ( iam==0 )
            printf("j %d: Ublock_info[j].iukp %d, Ublock_info[j].rukp %d,"
               "Ublock_info[j].jb %d, nsupc %d\n",
               j, Ublock_info[j].iukp, Ublock_info[j].rukp,
               Ublock_info[j].jb, nsupc); */

            /* Prepare to call GEMM. */
            jj = iukp;
            for (; jj < iukp + nsupc; ++jj)
            {
                segsize = klst - usub[jj];
                if (segsize)
                {
                    ++temp_ncols;
                    if (segsize > ldu)
                        ldu = segsize;
                }
            }

            Ublock_info[j - jj0].full_u_cols = temp_ncols;
            ncols += temp_ncols;

        } /* end for j ... compute ldu & ncols */

        /* Now doing prefix sum on full_u_cols.
         * After this, full_u_cols is the number of nonzero columns
         * from block 0 to block j.
         */
        for (int j = 1; j < num_u_blocks; ++j)
        {
            Ublock_info[j].full_u_cols += Ublock_info[j - 1].full_u_cols;
        }

        // get information of ublocks and pack it
        dgather_u(num_u_blks, Ublock_info, usub,
                  uval, bigU, ldu, xsup, klst);

        for (int ij = 0; ij < RemainBlk * num_u_blocks; ++ij)
        {
            /* jj0 starts after look-ahead window. */
            int j = ij / RemainBlk;
            int lb = ij % RemainBlk;

            int thread_id =0; 
            
            dblock_gemm_scatter(lb,j, Ublock_info, Remain_info,
                    L_mat, ldl,
                    bigU, ldu,
                    bigV, knsupc, klst,
                   lsub, usub,ldt,
                   thread_id,
                    indirect,
                    indirect2,
                   Lrowind_bc_ptr, Lnzval_bc_ptr,
                   Ufstnz_br_ptr, Unzval_br_ptr,
                   xsup, grid,
                    stat
#ifdef SCATTER_PROFILE
                    , Host_TheadScatterMOP, Host_TheadScatterTimer
#endif
                  )
        } /* if nbrow>0 */

    } /* if msg1 and msg 2 */
