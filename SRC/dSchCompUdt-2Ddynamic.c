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
 * -- Distributed SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 */

#define SCHEDULE_STRATEGY guided 

double tt_start;
double tt_end;
if ( msg0 && msg2 ) {  /* L(:,k) and U(k,:) are not empty. */
    int cum_nrow=0; /* cumulative number of nonzero rows in L(:,k) */
    int temp_nbrow; /* nonzero rows in current block L(i,k) */
    lptr = lptr0;
    luptr = luptr0;
    /**
     * Seperating L blocks into the top part within look-ahead window
     * and the remaining ones.
     */
     int lookAheadBlk=0, RemainBlk=0;

     tt_start = SuperLU_timer_();

     /* Loop through all blocks in L(:,k) to set up pointers to the start 
      * of each block in the data arrays.
      *   - lookAheadFullRow[i] := number of nonzero rows from block 0 to i
      *   - lookAheadStRow[i] := number of nonzero rows before block i
      *   - lookAhead_lptr[i] := point to the start of block i in L's index[] 
      *   - (ditto Remain_Info[i])
      */
     for (int i = 0; i < nlb; ++i) {
	 ib = lsub[lptr];            /* block number of L(i,k). */
	 temp_nbrow = lsub[lptr+1];  /* Number of full rows. */
        
	 int look_up_flag = 1; /* assume ib is outside look-up window */
	 for (int j = k0+1; j < SUPERLU_MIN (k0 + num_look_aheads+2, nsupers ); ++j)
	     {
		 if(ib == perm_c_supno[j]) {
		     look_up_flag=0; /* flag ib is within look-up window */
                     break; /* Sherry -- can exit the loop?? */
                 }
	     }
	 
	 if( look_up_flag == 0 ) { /* ib is within look up window */
	     if (lookAheadBlk==0) {
		 lookAheadFullRow[lookAheadBlk] = temp_nbrow;
	     } else {
		 lookAheadFullRow[lookAheadBlk] = temp_nbrow+lookAheadFullRow[lookAheadBlk-1];   
	     }
	     lookAheadStRow[lookAheadBlk] = cum_nrow;
	     lookAhead_lptr[lookAheadBlk] = lptr;
	     lookAhead_ib[lookAheadBlk] = ib; 
	     lookAheadBlk++;
	 } else { /* ib is not in look up window */

	     if (RemainBlk==0) {
		 Remain_info[RemainBlk].FullRow = temp_nbrow;
	     } else {
		 Remain_info[RemainBlk].FullRow = temp_nbrow+Remain_info[RemainBlk-1].FullRow;   
	     }

             RemainStRow[RemainBlk] = cum_nrow;
             // Remain_lptr[RemainBlk] = lptr;
	     Remain_info[RemainBlk].lptr = lptr;
	     // Remain_ib[RemainBlk] = ib; 
	     Remain_info[RemainBlk].ib = ib; 
	     RemainBlk++;
	 }
	 
         cum_nrow +=temp_nbrow;
	 
	 lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
	 lptr += temp_nbrow;     /* Move to next block */
	 luptr += temp_nbrow;
     }  /* for i ... all blocks in L(:,k) */

     lptr = lptr0;
     luptr = luptr0;

     /* leading dimension of L buffer */
#if 0
     int LDlookAhead_LBuff = lookAheadFullRow[lookAheadBlk-1]; /* may go negative.*/
#else /* Piyush fix */
     int LDlookAhead_LBuff = lookAheadBlk==0? 0 :lookAheadFullRow[lookAheadBlk-1];
#endif

     /* Loop through the look-ahead blocks to copy Lval into the buffer */
#ifdef __OPENMP
     /* #pragma omp parallel for -- why not?? Sherry */
#endif
     for (int i = 0; i < lookAheadBlk; ++i) {
	 int StRowDest  = 0;
	 int temp_nbrow;
	 if (i==0) {
	     temp_nbrow = lookAheadFullRow[0];
	 } else {
	     StRowDest   = lookAheadFullRow[i-1];
	     temp_nbrow  = lookAheadFullRow[i]-lookAheadFullRow[i-1];
	 }
	 
	 int StRowSource=lookAheadStRow[i];
	 
	 /* Now copying the matrix*/
	 // #pragma omp parallel for (gives slow down)
	 for (int j = 0; j < knsupc; ++j) {
	     memcpy(&lookAhead_L_buff[StRowDest+j*LDlookAhead_LBuff],
		    &lusup[luptr+j*nsupr+StRowSource],
		    temp_nbrow * sizeof(double) );
	 }
     }

     int LDRemain_LBuff = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;

    /* Loop through the remaining blocks to copy Lval into the buffer */
#ifdef _OPENMP
#pragma omp parallel for 
#endif
     for (int i = 0; i < RemainBlk; ++i) {
	 int StRowDest  = 0;
	 int temp_nbrow;
         if (i==0)  {
	     temp_nbrow = Remain_info[0].FullRow;
	 } else  {
	     StRowDest   = Remain_info[i-1].FullRow;
	     temp_nbrow  = Remain_info[i].FullRow-Remain_info[i-1].FullRow;
	 }

	 int StRowSource=RemainStRow[i];

	 /* Now copying the matrix*/
	 // #pragma omp parallel for (gives slow down)
	 for (int j = 0; j < knsupc; ++j) {
	     // printf("StRowDest %d LDRemain_LBuff %d StRowSource %d \n", StRowDest ,LDRemain_LBuff ,StRowSource );
	     memcpy(&Remain_L_buff[StRowDest+j*LDRemain_LBuff],
		    &lusup[luptr+j*nsupr+StRowSource],
                    temp_nbrow * sizeof(double) );
	 }
     } /* parallel for i ... */

     tt_end = SuperLU_timer_();

     LookAheadRowSepTimer += tt_end-tt_start;
#if 0
     LookAheadRowSepMOP  +=  2*knsupc*(lookAheadFullRow[lookAheadBlk-1]+Remain_info[RemainBlk-1].FullRow );
#else
     int_t lnbrow, rnbrow; /* number of nonzero rows in look-ahead window
                              or remaining part.  */
     lnbrow = lookAheadBlk==0 ? 0  : lookAheadFullRow[lookAheadBlk-1];
     rnbrow = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;
     nbrow = lnbrow + rnbrow; /* total number of rows in L */
     LookAheadRowSepMOP += 2*knsupc*(nbrow);
#endif     
     ldu   =0;
     full  =1;
     
     /*updating lookahead rows */
     
     tt_start = SuperLU_timer_();
#if 0     
     nbrow = lookAheadFullRow[lookAheadBlk-1]+Remain_info[RemainBlk-1].FullRow;
#endif

     if ( nbrow > 0 ) { /* L(:,k) is not empty */
	 /*
	  * Counting U blocks
	  */
	 ncols=0; /* total number of nonzero columns in U(k,:) */
	 ldu=0;
	 full=1; /* flag the U block is indeed 'full', containing segments
	            of same length. No need padding 0 */
	 int temp_ncols=0;

         /* Loop through all blocks in U(k,:) to set up pointers to the start
          * of each block in the data arrays, store them in Ublock_info[j]
          * for block U(k,j).
  	  */
	 for (j = jj0; j < nub; ++j) { /* jj0 was set to 0 */
	     temp_ncols = 0;
	     arrive_at_ublock(
			      j, &iukp, &rukp, &jb, &ljb, &nsupc,
			      iukp0, rukp0, usub, perm_u, xsup, grid
			      );
	     Ublock_info[j].iukp = iukp;
	     Ublock_info[j].rukp = rukp;
	     Ublock_info[j].jb = jb;
	     
	     /* Prepare to call GEMM. */
	     jj = iukp;
	     
	     for (; jj < iukp+nsupc; ++jj) {
		 segsize = klst - usub[jj];
		 if ( segsize ) {
                    ++temp_ncols;
                    if ( segsize != ldu ) full = 0; /* need padding 0 */
                    if ( segsize > ldu ) ldu = segsize;
		 }
	     }

	     Ublock_info[j].full_u_cols = temp_ncols;
	     ncols += temp_ncols;
	 }

	 /* Now doing prefix sum on full_u_cols.
	  * After this, full_u_cols is the number of nonzero columns
          * from block 0 to block j.
          */
	 for ( j = jj0+1; j < nub; ++j) {
	     Ublock_info[j].full_u_cols += Ublock_info[j-1].full_u_cols;
	 }
            
	 tempu = bigU; /* buffer the entire row block U(k,:) */

         /* Gather U(k,:) into buffer bigU[] to prepare for GEMM */
#ifdef _OPENMP        
#pragma omp parallel for private(j,iukp,rukp,tempu, jb, nsupc,ljb,segsize,\
	lead_zero, jj, i) \
        default (shared) schedule(SCHEDULE_STRATEGY)
#endif
        for (j = jj0; j < nub; ++j) { /* jj0 was set to 0 */

            if(j==jj0) tempu = bigU;
            else tempu = bigU + ldu*Ublock_info[j-1].full_u_cols;

            /* == processing each of the remaining columns == */
            arrive_at_ublock(j, &iukp, &rukp, &jb, &ljb, &nsupc,
			     iukp0, rukp0, usub,perm_u, xsup, grid);

            /* Copy from U(k,:) to tempu[], padding zeros.  */            
            for (jj = iukp; jj < iukp+nsupc; ++jj) {
                segsize = klst - usub[jj];
                if ( segsize ) {
                    lead_zero = ldu - segsize;
                    for (i = 0; i < lead_zero; ++i) tempu[i] = zero;
                    tempu += lead_zero;
                    for (i = 0; i < segsize; ++i) tempu[i] = uval[rukp+i];
                    rukp += segsize;
                    tempu += segsize;
                }
            }

            rukp -= usub[iukp - 1]; /* Return to start of U(k,j). */

        }   /* parallel for j:jjj_st..jjj */

        tempu = bigU;  /* setting to the start of padded U(k,:) */

    }  /* end if (nbrow>0) */

    tt_end = SuperLU_timer_();
    GatherTimer += tt_end-tt_start;
    GatherMOP += 2*ldu*ncols;

    int Lnbrow   = lookAheadBlk==0 ? 0 :lookAheadFullRow[lookAheadBlk-1];
    int Rnbrow   = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;
    int jj_cpu=nub;       /*limit between CPU and GPU */
    int thread_id;
    tempv = bigV;

    if ( Lnbrow>0 && ldu>0 && ncols>0 ) { /* Both L(:,k) and U(k,:) nonempty */
        /* Perform a large GEMM call */
        ncols = Ublock_info[nub-1].full_u_cols;
        schur_flop_counter += 2 * (double)Lnbrow * (double)ldu * (double)ncols;
        stat->ops[FACT]    += 2 * (double)Lnbrow * (double)ldu * (double)ncols;

        tt_start = SuperLU_timer_();

        /***************************************************************
         * Updating look-ahead blocks in both L and U look-ahead windows.
         ***************************************************************/
#ifdef _OPENMP
#pragma omp parallel default (shared) private(thread_id)
     {
 	thread_id = omp_get_thread_num();
 
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
#pragma omp for \
    private (j,i,lb,rukp,iukp,jb,nsupc,ljb,lptr,ib,temp_nbrow,cum_nrow)	\
    schedule(dynamic)
#else /* not use _OPENMP */
 	thread_id = 0;
#endif
 	/* Each thread is assigned one loop index ij, responsible for 
 	   block update L(lb,k) * U(k,j) -> tempv[]. */
        for (int ij = 0; ij < lookAheadBlk*(nub-jj0); ++ij) {
            int j   = ij/lookAheadBlk + jj0; /* jj0 was set to 0 */
            int lb  = ij%lookAheadBlk;

            int* indirect_thread    = indirect + ldt*thread_id;
            int* indirect2_thread   = indirect2 + ldt*thread_id;
            double* tempv1 = bigV + thread_id*ldt*ldt; 

            /* Getting U block U(k,j) information */
            /* unsigned long long ut_start, ut_end; */
            int_t rukp =  Ublock_info[j].rukp;
            int_t iukp =  Ublock_info[j].iukp;
            int jb   =  Ublock_info[j].jb;
            int nsupc = SuperSize(jb);
            int ljb = LBj (jb, grid);  /* destination column block */
            int st_col;
            int ncols;
            if ( j>jj0 ) { /* jj0 was set to 0 */
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

#if defined (USE_VENDOR_BLAS)            
            dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                  &lookAhead_L_buff[(knsupc-ldu)*Lnbrow+cum_nrow], &Lnbrow,
                  &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow, 1, 1);
#else
            dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                  &lookAhead_L_buff[(knsupc-ldu)*Lnbrow+cum_nrow], &Lnbrow,
                  &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow);
#endif
            if ( ib < jb ) {
                dscatter_u (
				 ib, jb,
				 nsupc, iukp, xsup,
				 klst, temp_nbrow,
				 lptr, temp_nbrow, lsub,
				 usub, tempv1,
				 Ufstnz_br_ptr, Unzval_br_ptr,
				 grid
			        );
            } else {
                dscatter_l (
				 ib, ljb, 
				 nsupc, iukp, xsup,
 				 klst, temp_nbrow,
				 lptr, temp_nbrow,
				 usub, lsub, tempv1,
				 indirect_thread, indirect2_thread,
				 Lrowind_bc_ptr, Lnzval_bc_ptr,
				 grid
				);
            }
        } /* end omp for ij = ... */
#ifdef _OPENMP
    } /* end omp parallel */
#endif
        tt_end = SuperLU_timer_();
        LookAheadGEMMTimer += tt_end- tt_start;
        LookAheadGEMMFlOp  += 2*(double)Lnbrow * (double)ldu * (double)ncols;
        stat->ops[FACT]    += 2 * (double)Lnbrow * (double)ldu * (double)ncols;
        LookAheadScatterTimer += tt_end-tt_start;
        LookAheadScatterMOP   += 3*Lnbrow*ncols;
    } /* end if Lnbrow < ... */
    
    /***************************************************************
     * Updating remaining rows and columns on CPU.
     ***************************************************************/
    Rnbrow  = RemainBlk==0 ? 0 : Remain_info[RemainBlk-1].FullRow;
    ncols   = jj_cpu==0 ? 0 : Ublock_info[jj_cpu-1].full_u_cols;

    schur_flop_counter  += 2 * (double)Rnbrow * (double)ldu * (double)ncols;
    stat->ops[FACT]     += 2 * (double)Rnbrow * (double)ldu * (double)ncols;

    tt_start = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel default(shared) private(thread_id)
    {
	thread_id = omp_get_thread_num();
 
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
#pragma omp for \
    private (j,i,lb,rukp,iukp,jb,nsupc,ljb,lptr,ib,temp_nbrow,cum_nrow)	\
    schedule(dynamic)
#else /* not use _OPENMP */
    thread_id = 0;
#endif
	/* Each thread is assigned one loop index ij, responsible for 
	   block update L(lb,k) * U(k,j) -> tempv[]. */
    for (int ij = 0; ij < RemainBlk*(jj_cpu-jj0); ++ij) { /* jj_cpu := nub */
	int j   = ij / RemainBlk + jj0; 
	int lb  = ij % RemainBlk;

	int* indirect_thread = indirect + ldt*thread_id;
	int* indirect2_thread = indirect2 + ldt*thread_id;
	double* tempv1 = bigV + thread_id*ldt*ldt; 

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
	    ncols  = Ublock_info[j].full_u_cols-Ublock_info[j-1].full_u_cols;
	    st_col = Ublock_info[j-1].full_u_cols;
	} else {
	    ncols  = Ublock_info[j].full_u_cols;
	    st_col = 0;   
	}

	/* Getting L block L(i,k) information */
	int_t lptr = Remain_info[lb].lptr;
	int ib   = Remain_info[lb].ib;
	int temp_nbrow = lsub[lptr+1];
	lptr += LB_DESCRIPTOR;
	int cum_nrow = (lb==0 ? 0 : Remain_info[lb-1].FullRow);

	/* calling GEMM */
#if defined (USE_VENDOR_BLAS)
	dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
	      &Remain_L_buff[(knsupc-ldu)*Rnbrow+cum_nrow], &Rnbrow,
	      &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow, 1, 1);
#else
	dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
	      &Remain_L_buff[(knsupc-ldu)*Rnbrow+cum_nrow], &Rnbrow,
	      &tempu[st_col*ldu], &ldu, &beta, tempv1, &temp_nbrow);
#endif

	/* Now scattering the block */
	if ( ib<jb ) {
	    dscatter_u(
			    ib, jb,
			    nsupc, iukp, xsup,
			    klst, temp_nbrow,
			    lptr, temp_nbrow,lsub,
			    usub, tempv1,
			    Ufstnz_br_ptr, Unzval_br_ptr,
			    grid
		           );
	} else {
	    dscatter_l(
			    ib, ljb,
			    nsupc, iukp, xsup,
			    klst, temp_nbrow,
			    lptr, temp_nbrow,
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
}  /* end if L(:,k) and U(k,:) are not empty */
