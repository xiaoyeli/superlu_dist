/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief This file contains the main loop of pdgstrf which involves rank k
 *        update of the Schur complement.
 *        Uses 2D partitioning for the scatter phase.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 5.4) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 * Modified:
 *   September 14, 2017
 *   - First gather U-panel, then depending on "ldu" (excluding leading zeros),
 *     gather only trailing columns of the L-panel corresponding to the nonzero
 *     of U-rows.
 *   - Padding zeros for nice dimensions of GEMM.
 *
 *  June 1, 2018  add parallel AWPM pivoting; add back arrive_at_ublock()
 */

#define SCHEDULE_STRATEGY guided

/*
 * Buffers:
 *     [ lookAhead_L_buff | Remain_L_buff ] : stores the gathered L-panel
 *                                            (A matrix in C := A*B )
 *     bigU : stores the U-panel (B matrix in C := A*B)
 *     bigV : stores the block GEMM result (C matrix in C := A*B)
 */

if ( msg0 && msg2 ) { /* L(:,k) and U(k,:) are not empty. */
    int cum_nrow = 0; /* cumulative number of nonzero rows in L(:,k) */
    int temp_nbrow;   /* nonzero rows in current block L(i,k) */
    lptr  = lptr0;
    luptr = luptr0;
    int Lnbrow, Rnbrow; /* number of nonzero rows in look-ahead window,
			   and remaining part.  */

    /*******************************************************************
     * Separating L blocks into the top part within look-ahead window
     * and the remaining ones.
     *******************************************************************/

     int lookAheadBlk=0, RemainBlk=0;

     tt_start = SuperLU_timer_();

     /* Sherry -- can this loop be threaded?? */
     /* Loop through all blocks in L(:,k) to set up pointers to the start
      * of each block in the data arrays.
      *   - lookAheadFullRow[i] := number of nonzero rows from block 0 to i
      *   - lookAheadStRow[i] := number of nonzero rows before block i
      *   - lookAhead_lptr[i] := point to the start of block i in L's index[]
      *   - (ditto Remain_Info[i])
      */
     for (int i = 0; i < nlb; ++i) {
	 ib = lsub[lptr];            /* Block number of L(i,k). */
	 temp_nbrow = lsub[lptr+1];  /* Number of full rows. */

	 int look_up_flag = 1; /* assume ib is outside look-up window */
	 for (int j = k0+1; j < SUPERLU_MIN (k0 + num_look_aheads+2, nsupers );
	      ++j) {
		 if ( ib == perm_c_supno[j] ) {
		     look_up_flag = 0; /* flag ib within look-up window */
                     break;            /* Sherry -- can exit the loop?? */
                 }
	 }

	 if ( look_up_flag == 0 ) { /* ib is within look-up window */
	     if (lookAheadBlk==0) {
		 lookAheadFullRow[lookAheadBlk] = temp_nbrow;
	     } else {
		 lookAheadFullRow[lookAheadBlk] =
		     temp_nbrow + lookAheadFullRow[lookAheadBlk-1];
	     }
	     lookAheadStRow[lookAheadBlk] = cum_nrow;
	     lookAhead_lptr[lookAheadBlk] = lptr;
	     lookAhead_ib[lookAheadBlk] = ib;
	     lookAheadBlk++;
	 } else { /* ib is not in look-up window */
	     if ( RemainBlk==0 ) {
		 Remain_info[RemainBlk].FullRow = temp_nbrow;
	     } else {
		 Remain_info[RemainBlk].FullRow =
		     temp_nbrow + Remain_info[RemainBlk-1].FullRow;
	     }
             RemainStRow[RemainBlk] = cum_nrow;
             // Remain_lptr[RemainBlk] = lptr;
	     Remain_info[RemainBlk].lptr = lptr;
	     // Remain_ib[RemainBlk] = ib;
	     Remain_info[RemainBlk].ib = ib;
	     RemainBlk++;
	 }

         cum_nrow += temp_nbrow;

	 lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
	 lptr += temp_nbrow;     /* Move to next block */
	 luptr += temp_nbrow;
     }  /* for i ... set up pointers for all blocks in L(:,k) */

     lptr = lptr0;
     luptr = luptr0;

     /* leading dimension of L look-ahead buffer, same as Lnbrow */
     //int LDlookAhead_LBuff = lookAheadBlk==0 ? 0 :lookAheadFullRow[lookAheadBlk-1];
     Lnbrow = lookAheadBlk==0 ? 0 : lookAheadFullRow[lookAheadBlk-1];
     /* leading dimension of L remaining buffer, same as Rnbrow */
     //int LDRemain_LBuff = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;
     Rnbrow = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;
     /* assert( cum_nrow == (LDlookAhead_LBuff + LDRemain_LBuff) );*/
     /* Piyush fix */
     //int LDlookAhead_LBuff = lookAheadBlk==0? 0 : lookAheadFullRow[lookAheadBlk-1];

     nbrow = Lnbrow + Rnbrow; /* total number of rows in L */
     LookAheadRowSepMOP += 2*knsupc*(nbrow);

     /***********************************************
      * Gather U blocks (AFTER LOOK-AHEAD WINDOW)   *
      ***********************************************/
     tt_start = SuperLU_timer_();

     if ( nbrow > 0 ) { /* L(:,k) is not empty */
	 /*
	  * Counting U blocks
	  */
     	 ldu = 0; /* Calculate ldu for U(k,:) after look-ahead window. */
	 ncols = 0; /* Total number of nonzero columns in U(k,:) */
	 int temp_ncols = 0;

	 /* jj0 contains the look-ahead window that was updated in
	    dlook_ahead_update.c. Now the search can continue from that point,
	    not to start from block 0. */
#if 0 // Sherry comment out 5/21/2018
	 /* Save pointers at location right after look-ahead window
	    for later restart. */
	 iukp0 = iukp;
	 rukp0 = rukp;
#endif

	 /* if ( iam==0 ) printf("--- k0 %d, k %d, jj0 %d, nub %d\n", k0, k, jj0, nub);*/

         /*
	  * Loop through all blocks in U(k,:) to set up pointers to the start
          * of each block in the data arrays, store them in Ublock_info[j]
          * for block U(k,j).
  	  */
	 for (j = jj0; j < nub; ++j) { /* jj0 starts after look-ahead window. */
	     temp_ncols = 0;
#if 1
	     /* Cannot remove following call, since perm_u != Identity  */
	     arrive_at_ublock(
			      j, &iukp, &rukp, &jb, &ljb, &nsupc,
			      iukp0, rukp0, usub, perm_u, xsup, grid
			      );
#else
	     jb = usub[iukp];
	     /* ljb = LBj (jb, grid);   Local block number of U(k,j). */
	     nsupc = SuperSize(jb);
	     iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
#endif
	     Ublock_info[j].iukp = iukp;
	     Ublock_info[j].rukp = rukp;
	     Ublock_info[j].jb = jb;

	     /* if ( iam==0 )
		 printf("j %d: Ublock_info[j].iukp %d, Ublock_info[j].rukp %d,"
			"Ublock_info[j].jb %d, nsupc %d\n",
			j, Ublock_info[j].iukp, Ublock_info[j].rukp,
			Ublock_info[j].jb, nsupc); */

	     /* Prepare to call GEMM. */
	     jj = iukp;
	     for (; jj < iukp+nsupc; ++jj) {
		 segsize = klst - usub[jj];
		 if ( segsize ) {
                    ++temp_ncols;
                    if ( segsize > ldu ) ldu = segsize;
		 }
	     }

	     Ublock_info[j].full_u_cols = temp_ncols;
	     ncols += temp_ncols;
#if 0 // Sherry comment out 5/31/2018 */
	     /* Jump number of nonzeros in block U(k,jj);
		Move to block U(k,j+1) in nzval[] array.  */
	     rukp += usub[iukp - 1];
	     iukp += nsupc;
#endif
         } /* end for j ... compute ldu & ncols */

	 /* Now doing prefix sum on full_u_cols.
	  * After this, full_u_cols is the number of nonzero columns
          * from block 0 to block j.
          */
	 for ( j = jj0+1; j < nub; ++j) {
	     Ublock_info[j].full_u_cols += Ublock_info[j-1].full_u_cols;
	 }

	 /* Padding zeros to make {m,n,k} multiple of vector length. */
	 jj = 8; //n;
	 if (gemm_padding > 0 && Rnbrow > jj && ncols > jj && ldu > jj) {
	     gemm_m_pad = Rnbrow + (Rnbrow % GEMM_PADLEN);
	     gemm_n_pad = ncols + (ncols % GEMM_PADLEN);
	     //gemm_n_pad = ncols;
	     //gemm_k_pad = ldu + (ldu % GEMM_PADLEN);
	     gemm_k_pad = ldu;

	     for (i = Rnbrow; i < gemm_m_pad; ++i)  // padding A matrix
		 for (j = 0; j < gemm_k_pad; ++j)
		     Remain_L_buff[i + j*gemm_m_pad] = zero;
	     for (i = 0; i < Rnbrow; ++i)
		 for (j = ldu; j < gemm_k_pad; ++j)
		     Remain_L_buff[i + j*gemm_m_pad] = zero;
	     for (i = ldu; i < gemm_k_pad; ++i)     // padding B matrix
		 for (j = 0; j < gemm_n_pad; ++j)
		     bigU[i + j*gemm_k_pad] = zero;
	     for (i = 0; i < ldu; ++i)
		 for (j = ncols; j < gemm_n_pad; ++j)
		     bigU[i + j*gemm_k_pad] = zero;
	 } else {
	     gemm_m_pad = Rnbrow;
	     gemm_n_pad = ncols;
	     gemm_k_pad = ldu;
	 }

	 tempu = bigU; /* buffer the entire row block U(k,:) */

         /* Gather U(k,:) into buffer bigU[] to prepare for GEMM */
#ifdef _OPENMP
#pragma omp parallel for firstprivate(iukp, rukp) \
    private(j,tempu, jb, nsupc,ljb,segsize, lead_zero, jj, i) \
    default (shared) schedule(SCHEDULE_STRATEGY)
#endif
        for (j = jj0; j < nub; ++j) { /* jj0 starts after look-ahead window. */

            if (j==jj0) tempu = bigU;
            //else tempu = bigU + ldu * Ublock_info[j-1].full_u_cols;
            else tempu = bigU + gemm_k_pad * Ublock_info[j-1].full_u_cols;

            /* == processing each of the remaining columns in parallel == */
#if 0
	    /* Can remove following call, since search was already done.  */
            arrive_at_ublock(j, &iukp, &rukp, &jb, &ljb, &nsupc,
			     iukp0, rukp0, usub,perm_u, xsup, grid);
#else
	    iukp = Ublock_info[j].iukp;
	    rukp = Ublock_info[j].rukp;
	    jb = Ublock_info[j].jb;
	    nsupc = SuperSize (jb );
#endif
            /* Copy from U(k,j) to tempu[], padding zeros.  */
            for (jj = iukp; jj < iukp+nsupc; ++jj) {
                segsize = klst - usub[jj];
                if ( segsize ) {
                    lead_zero = ldu - segsize;
                    for (i = 0; i < lead_zero; ++i) tempu[i] = zero;
		    //tempu += lead_zero;
#if (_OPENMP>=201307)
#pragma omp simd
#endif
		    for (i = 0; i < segsize; ++i)
                    	tempu[i+lead_zero] = uval[rukp+i];
                    rukp += segsize;
                    tempu += gemm_k_pad;
                }
	    }
        }   /* parallel for j = jj0 .. nub */

#if 0
	if (ldu==0) printf("[%d] .. k0 %d, before updating: ldu %d, Lnbrow %d, Rnbrow %d, ncols %d\n",iam,k0,ldu,Lnbrow,Rnbrow, ncols);
	fflush(stdout);
#endif

        GatherMOP += 2*ldu*ncols;

    }  /* end if (nbrow>0), end gather U blocks */

    GatherUTimer += SuperLU_timer_() - tt_start;
    int jj_cpu = nub;       /* limit between CPU and GPU */
    int thread_id;
    /*tempv = bigV;*/

    /**********************
     * Gather L blocks    *
     **********************/
     tt_start = SuperLU_timer_();

     /* Loop through the look-ahead blocks to copy Lval into the buffer */
#ifdef _OPENMP
#pragma omp parallel for private(j,jj,tempu,tempv) default (shared)
#endif
     for (i = 0; i < lookAheadBlk; ++i) {
	 int StRowDest, temp_nbrow;
	 if ( i==0 ) {
	     StRowDest = 0;
	     temp_nbrow = lookAheadFullRow[0];
	 } else {
	     StRowDest   = lookAheadFullRow[i-1];
	     temp_nbrow  = lookAheadFullRow[i]-lookAheadFullRow[i-1];
	 }

	 int StRowSource = lookAheadStRow[i];

	 /* Now copying one block into L lookahead buffer */
	 /* #pragma omp parallel for (gives slow down) */
	 // for (int j = 0; j < knsupc; ++j) {
	 for (j = knsupc-ldu; j < knsupc; ++j) { /* skip leading columns
						    corresponding to zero U rows */
#if 1
	     /* Better let compiler generate memcpy or vectorized code. */
	     //tempu = &lookAhead_L_buff[StRowDest + j*LDlookAhead_LBuff];
	     //tempu = &lookAhead_L_buff[StRowDest + j * Lnbrow];
	     tempu = &lookAhead_L_buff[StRowDest + (j - (knsupc-ldu)) * Lnbrow];
	     tempv = &lusup[luptr+j*nsupr + StRowSource];
#if (_OPENMP>=201307)
#pragma omp simd
#endif
	     for (jj = 0; jj < temp_nbrow; ++jj) tempu[jj] = tempv[jj];
#else
	     //memcpy(&lookAhead_L_buff[StRowDest + j*LDlookAhead_LBuff],
	     memcpy(&lookAhead_L_buff[StRowDest + (j - (knsupc-ldu)) * Lnbrow],
		    &lusup[luptr+j*nsupr + StRowSource],
		    temp_nbrow * sizeof(doublecomplex) );
#endif
	 } /* end for j ... */
     } /* parallel for i ... gather Lval blocks from lookahead window */

     /* Loop through the remaining blocks to copy Lval into the buffer */
#ifdef _OPENMP
#pragma omp parallel for private(i,j,jj,tempu,tempv) default (shared) \
    schedule(SCHEDULE_STRATEGY)
#endif
     for (int i = 0; i < RemainBlk; ++i) {
         int StRowDest, temp_nbrow;
         if ( i==0 )  {
	     StRowDest  = 0;
	     temp_nbrow = Remain_info[0].FullRow;
	 } else  {
	     StRowDest   = Remain_info[i-1].FullRow;
	     temp_nbrow  = Remain_info[i].FullRow - Remain_info[i-1].FullRow;
	 }

	 int StRowSource = RemainStRow[i];

	 /* Now copying a block into L remaining buffer */
	 // #pragma omp parallel for (gives slow down)
	 // for (int j = 0; j < knsupc; ++j) {
	 for (int j = knsupc-ldu; j < knsupc; ++j) {
	     // printf("StRowDest %d Rnbrow %d StRowSource %d \n", StRowDest,Rnbrow ,StRowSource);
#if 1
	     /* Better let compiler generate memcpy or vectorized code. */
	     //tempu = &Remain_L_buff[StRowDest + j*LDRemain_LBuff];
	     //tempu = &Remain_L_buff[StRowDest + (j - (knsupc-ldu)) * Rnbrow];
	     tempu = &Remain_L_buff[StRowDest + (j - (knsupc-ldu)) * gemm_m_pad];
	     tempv = &lusup[luptr + j*nsupr + StRowSource];
#if (_OPENMP>=201307)
#pragma omp simd
#endif
	     for (jj = 0; jj < temp_nbrow; ++jj) tempu[jj] = tempv[jj];
#else
	     //memcpy(&Remain_L_buff[StRowDest + j*LDRemain_LBuff],
	     memcpy(&Remain_L_buff[StRowDest + (j - (knsupc-ldu)) * gemm_m_pad],
		    &lusup[luptr+j*nsupr + StRowSource],
                    temp_nbrow * sizeof(doublecomplex) );
#endif
	 } /* end for j ... */
     } /* parallel for i ... copy Lval into the remaining buffer */

     tt_end = SuperLU_timer_();
     GatherLTimer += tt_end - tt_start;


     /*************************************************************************
      * Perform GEMM (look-ahead L part, and remain L part) followed by Scatter
      *************************************************************************/
     tempu = bigU;  /* setting to the start of padded U(k,:) */

     if ( Lnbrow>0 && ldu>0 && ncols>0 ) { /* Both L(:,k) and U(k,:) nonempty */
	 /***************************************************************
	  * Updating blocks in look-ahead window of the LU(look-ahead-rows,:)
	  ***************************************************************/

	 /* Count flops for total GEMM calls */
	 ncols = Ublock_info[nub-1].full_u_cols;
	 flops_t flps = 8.0 * (flops_t)Lnbrow * ldu * ncols;
	 LookAheadScatterMOP += 3 * Lnbrow * ncols; /* scatter-add */
	 schur_flop_counter += flps;
	 stat->ops[FACT]    += flps;
	 LookAheadGEMMFlOp  += flps;

#ifdef _OPENMP
#pragma omp parallel default (shared) private(thread_id)
	 {
#ifdef _OPENMP	 
	   thread_id = omp_get_thread_num();
#else	   
	   thread_id = 0;
#endif

	   /* Ideally, should organize the loop as:
	      for (j = 0; j < nub; ++j) {
	          for (lb = 0; lb < lookAheadBlk; ++lb) {
	               L(lb,k) X U(k,j) -> tempv[]
		  }
	      }
	      But now, we use collapsed loop to achieve more parallelism.
	      Total number of block updates is:
	      (# of lookAheadBlk in L(:,k)) X (# of blocks in U(k,:))
	   */

	   int i = sizeof(int);
	   int* indirect_thread    = indirect + (ldt + CACHELINE/i) * thread_id;
	   int* indirect2_thread   = indirect2 + (ldt + CACHELINE/i) * thread_id;

#pragma omp for \
    private (nsupc,ljb,lptr,ib,temp_nbrow,cum_nrow)	\
    schedule(dynamic)
#else /* not use _OPENMP */
	   thread_id = 0;
	   int* indirect_thread    = indirect;
	   int* indirect2_thread   = indirect2;
#endif
	   /* Each thread is assigned one loop index ij, responsible for
	      block update L(lb,k) * U(k,j) -> tempv[]. */
	   for (int ij = 0; ij < lookAheadBlk*(nub-jj0); ++ij) {
	       /* jj0 starts after look-ahead window. */
            int j   = ij/lookAheadBlk + jj0;
            int lb  = ij%lookAheadBlk;

            /* Getting U block U(k,j) information */
            /* unsigned long long ut_start, ut_end; */
            int_t rukp =  Ublock_info[j].rukp;
            int_t iukp =  Ublock_info[j].iukp;
            int jb   =  Ublock_info[j].jb;
            int nsupc = SuperSize(jb);
            int ljb = LBj (jb, grid);  /* destination column block */
            int st_col;
            int ncols;  /* Local variable counts only columns in the block */
            if ( j > jj0 ) { /* jj0 starts after look-ahead window. */
                ncols  = Ublock_info[j].full_u_cols-Ublock_info[j-1].full_u_cols;
                st_col = Ublock_info[j-1].full_u_cols;
            } else {
                ncols  = Ublock_info[j].full_u_cols;
                st_col = 0;
            }

            /* Getting L block L(i,k) information */
            int_t lptr = lookAhead_lptr[lb];
            int ib   = lookAhead_ib[lb];
            int temp_nbrow = lsub[lptr+1];
            lptr += LB_DESCRIPTOR;
            int cum_nrow = (lb==0 ? 0 : lookAheadFullRow[lb-1]);

	    /* Block-by-block GEMM in look-ahead window */
#if 0
	    i = sizeof(doublecomplex);
	    doublecomplex* tempv1 = bigV + thread_id * (ldt*ldt + CACHELINE/i);
#else
	    doublecomplex* tempv1 = bigV + thread_id * (ldt*ldt);
#endif

#if ( PRNTlevel>= 1)
	    if (thread_id == 0) tt_start = SuperLU_timer_();
	    gemm_max_m = SUPERLU_MAX(gemm_max_m, temp_nbrow);
	    gemm_max_n = SUPERLU_MAX(gemm_max_n, ncols);
	    gemm_max_k = SUPERLU_MAX(gemm_max_k, ldu);
#endif

#if defined (USE_VENDOR_BLAS)
            zgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
		   //&lookAhead_L_buff[(knsupc-ldu)*Lnbrow+cum_nrow], &Lnbrow,
		   &lookAhead_L_buff[cum_nrow], &Lnbrow,
		   &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow, 1, 1);
#else
            zgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
		   //&lookAhead_L_buff[(knsupc-ldu)*Lnbrow+cum_nrow], &Lnbrow,
		   &lookAhead_L_buff[cum_nrow], &Lnbrow,
		   &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow);
#endif

#if (PRNTlevel>=1 )
	    if (thread_id == 0) {
		tt_end = SuperLU_timer_();
		LookAheadGEMMTimer += tt_end - tt_start;
		tt_start = tt_end;
	    }
#endif
            if ( ib < jb ) {
                zscatter_u (
				 ib, jb,
				 nsupc, iukp, xsup,
				 klst, temp_nbrow,
				 lptr, temp_nbrow, lsub,
				 usub, tempv1,
				 Ufstnz_br_ptr, Unzval_br_ptr,
				 grid
			        );
            } else {
#if 0
		//#ifdef USE_VTUNE
	    __SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
	    __itt_resume(); // start VTune, again use 2 underscores
#endif
                zscatter_l (
				 ib, ljb,
				 nsupc, iukp, xsup,
 				 klst, temp_nbrow,
				 lptr, temp_nbrow,
				 usub, lsub, tempv1,
				 indirect_thread, indirect2_thread,
				 Lrowind_bc_ptr, Lnzval_bc_ptr,
				 grid
				);
#if 0
		//#ifdef USE_VTUNE
		__itt_pause(); // stop VTune
		__SSC_MARK(0x222); // stop SDE tracing
#endif
            }

#if ( PRNTlevel>=1 )
	    if (thread_id == 0)
		LookAheadScatterTimer += SuperLU_timer_() - tt_start;
#endif
	   } /* end omp for ij = ... */

#ifdef _OPENMP
	 } /* end omp parallel */
#endif
     } /* end if Lnbrow>0 ... look-ahead GEMM and scatter */

    /***************************************************************
     * Updating remaining rows and columns on CPU.
     ***************************************************************/
    ncols = jj_cpu==0 ? 0 : Ublock_info[jj_cpu-1].full_u_cols;

    if ( Rnbrow>0 && ldu>0 ) { /* There are still blocks remaining ... */
	double flps = 8.0 * (double)Rnbrow * ldu * ncols;
	schur_flop_counter  += flps;
	stat->ops[FACT]     += flps;

#if ( PRNTlevel>=1 )
	RemainGEMM_flops += flps;
	gemm_max_m = SUPERLU_MAX(gemm_max_m, Rnbrow);
	gemm_max_n = SUPERLU_MAX(gemm_max_n, ncols);
	gemm_max_k = SUPERLU_MAX(gemm_max_k, ldu);
	tt_start = SuperLU_timer_();
	/* printf("[%d] .. k0 %d, before large GEMM: %d-%d-%d, RemainBlk %d\n",
	   iam, k0,Rnbrow,ldu,ncols,RemainBlk);  fflush(stdout);
	assert( Rnbrow*ncols < bigv_size ); */
#endif
	/* calling aggregated large GEMM, result stored in bigV[]. */
#if defined (USE_VENDOR_BLAS)
	//zgemm_("N", "N", &Rnbrow, &ncols, &ldu, &alpha,
	zgemm_("N", "N", &gemm_m_pad, &gemm_n_pad, &gemm_k_pad, &alpha,
	       //&Remain_L_buff[(knsupc-ldu)*Rnbrow], &Rnbrow,
	       &Remain_L_buff[0], &gemm_m_pad,
	       &bigU[0], &gemm_k_pad, &beta, bigV, &gemm_m_pad, 1, 1);
#else
	//zgemm_("N", "N", &Rnbrow, &ncols, &ldu, &alpha,
	zgemm_("N", "N", &gemm_m_pad, &gemm_n_pad, &gemm_k_pad, &alpha,
	       //&Remain_L_buff[(knsupc-ldu)*Rnbrow], &Rnbrow,
	       &Remain_L_buff[0], &gemm_m_pad,
	       &bigU[0], &gemm_k_pad, &beta, bigV, &gemm_m_pad);
#endif

#if ( PRNTlevel>=1 )
	tt_end = SuperLU_timer_();
	RemainGEMMTimer += tt_end - tt_start;
#if ( PROFlevel>=1 )
	//fprintf(fgemm, "%8d%8d%8d %16.8e\n", Rnbrow, ncols, ldu,
	// (tt_end - tt_start)*1e6); // time in microsecond
	//fflush(fgemm);
	gemm_stats[gemm_count].m = Rnbrow;
	gemm_stats[gemm_count].n = ncols;
	gemm_stats[gemm_count].k = ldu;
	gemm_stats[gemm_count++].microseconds = (tt_end - tt_start) * 1e6;
#endif
	tt_start = SuperLU_timer_();
#endif

#ifdef USE_VTUNE
	__SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
	__itt_resume(); // start VTune, again use 2 underscores
#endif

	/* Scatter into destination block-by-block. */
#ifdef _OPENMP
#pragma omp parallel default(shared) private(thread_id)
	{
#ifdef _OPENMP	
	    thread_id = omp_get_thread_num();
#else	    
	    thread_id = 0;
#endif

	    /* Ideally, should organize the loop as:
               for (j = 0; j < jj_cpu; ++j) {
	           for (lb = 0; lb < RemainBlk; ++lb) {
	               L(lb,k) X U(k,j) -> tempv[]
                   }
               }
	       But now, we use collapsed loop to achieve more parallelism.
	       Total number of block updates is:
	       (# of RemainBlk in L(:,k)) X (# of blocks in U(k,:))
	    */

	    int i = sizeof(int);
	    int* indirect_thread = indirect + (ldt + CACHELINE/i) * thread_id;
	    int* indirect2_thread = indirect2 + (ldt + CACHELINE/i) * thread_id;

#pragma omp for \
    private (j,lb,rukp,iukp,jb,nsupc,ljb,lptr,ib,temp_nbrow,cum_nrow)	\
    schedule(dynamic)
#else /* not use _OPENMP */
	    thread_id = 0;
	    int* indirect_thread = indirect;
	    int* indirect2_thread = indirect2;
#endif
	    /* Each thread is assigned one loop index ij, responsible for
	       block update L(lb,k) * U(k,j) -> tempv[]. */
	    for (int ij = 0; ij < RemainBlk*(jj_cpu-jj0); ++ij) {
		/* jj_cpu := nub, jj0 starts after look-ahead window. */
		int j   = ij / RemainBlk + jj0; /* j-th block in U panel */
		int lb  = ij % RemainBlk;       /* lb-th block in L panel */

		/* Getting U block U(k,j) information */
		/* unsigned long long ut_start, ut_end; */
		int_t rukp =  Ublock_info[j].rukp;
		int_t iukp =  Ublock_info[j].iukp;
		int jb   =  Ublock_info[j].jb;
		int nsupc = SuperSize(jb);
		int ljb = LBj (jb, grid);
		int st_col;
		int ncols;
		if ( j>jj0 ) {
		    ncols = Ublock_info[j].full_u_cols - Ublock_info[j-1].full_u_cols;
		    st_col = Ublock_info[j-1].full_u_cols;
		} else {
		    ncols = Ublock_info[j].full_u_cols;
		    st_col = 0;
		}

		/* Getting L block L(i,k) information */
		int_t lptr = Remain_info[lb].lptr;
		int ib   = Remain_info[lb].ib;
		int temp_nbrow = lsub[lptr+1];
		lptr += LB_DESCRIPTOR;
		int cum_nrow = (lb==0 ? 0 : Remain_info[lb-1].FullRow);

		/* tempv1 points to block(i,j) in bigV : LDA == Rnbrow */
		//double* tempv1 = bigV + (st_col * Rnbrow + cum_nrow); Sherry
		doublecomplex* tempv1 = bigV + (st_col * gemm_m_pad + cum_nrow); /* Sherry */

		// printf("[%d] .. before scatter: ib %d, jb %d, temp_nbrow %d, Rnbrow %d\n", iam, ib, jb, temp_nbrow, Rnbrow); fflush(stdout);

		/* Now scattering the block */

		if ( ib < jb ) {
		    zscatter_u (
				ib, jb,
				nsupc, iukp, xsup,
				//klst, Rnbrow, /*** klst, temp_nbrow, Sherry */
				klst, gemm_m_pad, /*** klst, temp_nbrow, Sherry */
				lptr, temp_nbrow, /* row dimension of the block */
				lsub, usub, tempv1,
				Ufstnz_br_ptr, Unzval_br_ptr,
				grid
				);
		} else {
		    zscatter_l(
			       ib, ljb,
			       nsupc, iukp, xsup,
			       //klst, temp_nbrow, Sherry
			       klst, gemm_m_pad, /*** temp_nbrow, Sherry */
			       lptr, temp_nbrow, /* row dimension of the block */
			       usub, lsub, tempv1,
			       indirect_thread, indirect2_thread,
			       Lrowind_bc_ptr,Lnzval_bc_ptr,
			       grid
			       );
		}

	    } /* end omp for (int ij =...) */

#ifdef _OPENMP
	} /* end omp parallel region */
#endif

#if ( PRNTlevel>=1 )
	RemainScatterTimer += SuperLU_timer_() - tt_start;
#endif

#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

    } /* end if Rnbrow>0 ... update remaining block */

}  /* end if L(:,k) and U(k,:) are not empty */
