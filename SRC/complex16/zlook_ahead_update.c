/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/************************************************************************/
/*! @file
 * \brief Look-ahead update of the Schur complement.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 5.4) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 * Modified:
 *  September 18, 2017
 *  June 1, 2018  add parallel AWPM pivoting; add back arrive_at_ublock()
 *
 */

#include <assert.h>  /* assertion doesn't work if NDEBUG is defined */

iukp = iukp0; /* point to the first block in index[] */
rukp = rukp0; /* point to the start of nzval[] */
j = jj0 = 0;  /* After the j-loop, jj0 points to the first block in U
                 outside look-ahead window. */

#if 0
for (jj = 0; jj < nub; ++jj) assert(perm_u[jj] == jj); /* Sherry */
#endif

#ifdef ISORT
while (j < nub && iperm_u[j] <= k0 + num_look_aheads)
#else
while (j < nub && perm_u[2 * j] <= k0 + num_look_aheads)
#endif
{
    doublecomplex zero = {0.0, 0.0};

#if 1
    /* Search is needed because a permutation perm_u is involved for j  */
    /* Search along the row for the pointers {iukp, rukp} pointing to
     * block U(k,j).
     * j    -- current block in look-ahead window, initialized to 0 on entry
     * iukp -- point to the start of index[] metadata
     * rukp -- point to the start of nzval[] array
     * jb   -- block number of block U(k,j), update destination column
     */
    arrive_at_ublock(
		     j, &iukp, &rukp, &jb, &ljb, &nsupc,
         	     iukp0, rukp0, usub, perm_u, xsup, grid
		    );
#else
    jb = usub[iukp];
    ljb = LBj (jb, grid);     /* Local block number of U(k,j). */
    nsupc = SuperSize(jb);
    iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
#endif

    j++;
    jj0++;
    jj = iukp;

    while (usub[jj] == klst) ++jj; /* Skip zero segments */

    ldu = klst - usub[jj++];
    ncols = 1;

    /* This loop computes ldu. */
    for (; jj < iukp + nsupc; ++jj) { /* for each column jj in block U(k,j) */
        segsize = klst - usub[jj];
        if (segsize) {
            ++ncols;
            if (segsize > ldu)  ldu = segsize;
        }
    }
#if ( DEBUGlevel>=3 )
    ++num_update;
#endif

#if ( DEBUGlevel>=3 )
    printf ("(%d) k=%d,jb=%d,ldu=%d,ncols=%d,nsupc=%d\n",
	    iam, k, jb, ldu, ncols, nsupc);
    ++num_copy;
#endif

    /* Now copy one block U(k,j) to bigU for GEMM, padding zeros up to ldu. */
    tempu = bigU; /* Copy one block U(k,j) to bigU for GEMM */
    for (jj = iukp; jj < iukp + nsupc; ++jj) {
        segsize = klst - usub[jj];
        if (segsize) {
            lead_zero = ldu - segsize;
            for (i = 0; i < lead_zero; ++i) tempu[i] = zero;
            tempu += lead_zero;
            for (i = 0; i < segsize; ++i) {
                tempu[i] = uval[rukp + i];
            }
            rukp += segsize;
            tempu += segsize;
        }
    }
    tempu = bigU; /* set back to the beginning of the buffer */

    nbrow = lsub[1]; /* number of row subscripts in L(:,k) */
    if (myrow == krow) nbrow = lsub[1] - lsub[3]; /* skip diagonal block for those rows. */
    // double ttx =SuperLU_timer_();

    int current_b = 0; /* Each thread starts searching from first block.
                          This records the moving search target.           */
    lptr = lptr0; /* point to the start of index[] in supernode L(:,k) */
    luptr = luptr0;

#ifdef _OPENMP
    /* Sherry -- examine all the shared variables ??
       'firstprivate' ensures that the private variables are initialized
       to the values before entering the loop.  */
#pragma omp parallel for \
    firstprivate(lptr,luptr,ib,current_b) private(lb) \
    default(shared) schedule(dynamic)
#endif
    for (lb = 0; lb < nlb; lb++) { /* Loop through each block in L(:,k) */
        int temp_nbrow; /* automatic variable is private */

        /* Search for the L block that my thread will work on.
           No need to search from 0, can continue at the point where
           it is left from last iteration.
           Note: Blocks may not be sorted in L. Different thread picks up
	   different lb.   */
        for (; current_b < lb; ++current_b) {
            temp_nbrow = lsub[lptr + 1];    /* Number of full rows. */
            lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
            lptr += temp_nbrow;   /* move to next block */
            luptr += temp_nbrow;  /* move to next block */
        }

#ifdef _OPENMP
        int_t thread_id = omp_get_thread_num ();
#else
        int_t thread_id = 0;
#endif
        doublecomplex * tempv = bigV + ldt*ldt*thread_id;

        int *indirect_thread  = indirect + ldt * thread_id;
        int *indirect2_thread = indirect2 + ldt * thread_id;
        ib = lsub[lptr];        /* block number of L(i,k) */
        temp_nbrow = lsub[lptr + 1];    /* Number of full rows. */
	/* assert (temp_nbrow <= nbrow); */

        lptr += LB_DESCRIPTOR;  /* Skip descriptor. */

	/*if (thread_id == 0) tt_start = SuperLU_timer_();*/

        /* calling gemm */
	stat->ops[FACT] += 8.0 * (flops_t)temp_nbrow * ldu * ncols;
#if defined (USE_VENDOR_BLAS)
        zgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                   &lusup[luptr + (knsupc - ldu) * nsupr], &nsupr,
                   tempu, &ldu, &beta, tempv, &temp_nbrow, 1, 1);
#else
        zgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                   &lusup[luptr + (knsupc - ldu) * nsupr], &nsupr,
                   tempu, &ldu, &beta, tempv, &temp_nbrow );
#endif

#if 0
	if (thread_id == 0) {
	    tt_end = SuperLU_timer_();
	    LookAheadGEMMTimer += tt_end - tt_start;
	    tt_start = tt_end;
	}
#endif
        /* Now scattering the output. */
        if (ib < jb) {    /* A(i,j) is in U. */
            zscatter_u (ib, jb,
                       nsupc, iukp, xsup,
                       klst, temp_nbrow,
                       lptr, temp_nbrow, lsub,
                       usub, tempv, Ufstnz_br_ptr, Unzval_br_ptr, grid);
        } else {          /* A(i,j) is in L. */
            zscatter_l (ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
                       temp_nbrow, usub, lsub, tempv,
                       indirect_thread, indirect2_thread,
                       Lrowind_bc_ptr, Lnzval_bc_ptr, grid);
        }

        ++current_b;         /* Move to next block. */
        lptr += temp_nbrow;
        luptr += temp_nbrow;

#if 0
	if (thread_id == 0) {
	    tt_end = SuperLU_timer_();
	    LookAheadScatterTimer += tt_end - tt_start;
	}
#endif
    } /* end parallel for lb = 0, nlb ... all blocks in L(:,k) */

    iukp += nsupc; /* Mov to block U(k,j+1) */

    /* =========================================== *
     * == factorize L(:,j) and send if possible == *
     * =========================================== */
    kk = jb; /* destination column that is just updated */
    kcol = PCOL (kk, grid);
#ifdef ISORT
    kk0 = iperm_u[j - 1];
#else
    kk0 = perm_u[2 * (j - 1)];
#endif
    look_id = kk0 % (1 + num_look_aheads);

    if (look_ahead[kk] == k0 && kcol == mycol) {
        /* current column is the last dependency */
        look_id = kk0 % (1 + num_look_aheads);

        /* Factor diagonal and subdiagonal blocks and test for exact
           singularity.  */
        factored[kk] = 0;

        double tt1 = SuperLU_timer_();

        PZGSTRF2(options, kk0, kk, thresh, Glu_persist, grid, Llu,
                  U_diag_blk_send_req, tag_ub, stat, info);

        pdgstrf2_timer += SuperLU_timer_() - tt1;

        /* stat->time7 += SuperLU_timer_() - ttt1; */

        /* Multicasts numeric values of L(:,kk) to process rows. */
        send_req = send_reqs[look_id];
        msgcnt = msgcnts[look_id];

        lk = LBj (kk, grid);    /* Local block number. */
        lsub1 = Lrowind_bc_ptr[lk];
        lusup1 = Lnzval_bc_ptr[lk];
        if (lsub1) {
            msgcnt[0] = lsub1[1] + BC_HEADER + lsub1[0] * LB_DESCRIPTOR;
            msgcnt[1] = lsub1[1] * SuperSize (kk);
        } else {
            msgcnt[0] = 0;
            msgcnt[1] = 0;
        }

        scp = &grid->rscp;      /* The scope of process row. */
        for (pj = 0; pj < Pc; ++pj) {
            if (ToSendR[lk][pj] != EMPTY) {
#if ( PROFlevel>=1 )
                TIC (t1);
#endif
                MPI_Isend (lsub1, msgcnt[0], mpi_int_t, pj,
                           SLU_MPI_TAG (0, kk0) /* (4*kk0)%tag_ub */ ,
                           scp->comm, &send_req[pj]);
                MPI_Isend (lusup1, msgcnt[1], SuperLU_MPI_DOUBLE_COMPLEX, pj,
                           SLU_MPI_TAG (1, kk0) /* (4*kk0+1)%tag_ub */ ,
                           scp->comm, &send_req[pj + Pc]);
#if ( PROFlevel>=1 )
                TOC (t2, t1);
                stat->utime[COMM] += t2;
                msg_cnt += 2;
                msg_vol += msgcnt[0] * iword + msgcnt[1] * dword;
#endif
#if ( DEBUGlevel>=2 )
                printf ("[%d] -2- Send L(:,%4d): #lsub %4d, #lusup %4d to Pj %2d, tags %d:%d \n",
                        iam, kk, msgcnt[0], msgcnt[1], pj,
			SLU_MPI_TAG(0,kk0), SLU_MPI_TAG(1,kk0));
#endif
            }  /* end if ( ToSendR[lk][pj] != EMPTY ) */
        } /* end for pj ... */
    } /* end if( look_ahead[kk] == k0 && kcol == mycol ) */
} /* end while j < nub and perm_u[j] <k0+NUM_LOOK_AHEAD */

