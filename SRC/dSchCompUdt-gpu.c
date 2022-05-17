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

if ( msg0 && msg2 ) {  /* L(:,k) and U(k,:) are not empty. */
    ldu   =0;
    full  =1;
    int cum_nrow;
    int temp_nbrow;

    lptr = lptr0;
    luptr = luptr0;

    nbrow= lsub[1];
    if (myrow==krow) nbrow = lsub[1]-lsub[3];

    if (nbrow>0) {

        // Maximum number of columns that can fit in dC[buffer_size] on GPU 
#if 0   // max_ldu can be < ldt, so bigu_size/ldt may be smaller, giving false alarm
        int ncol_max = SUPERLU_MIN(buffer_size/nbrow,bigu_size/ldt);
#else // Sherry fix
        int ncol_max = SUPERLU_MIN(buffer_size/nbrow, max_ncols);
#endif
	
        int num_streams_used, /* number of streams that will be used*/
        ncpu_blks;            /* the leading number of CPU dgemm blks
			         in each partition */
        int jjj, jjj_st,jjj_global;
        for (j = jj0; j < nub; ++j) {
            arrive_at_ublock( j,&iukp,&rukp,&jb,&ljb,&nsupc,
	    		      iukp0,rukp0,usub,perm_u,xsup,grid );

            ncols =0 ;  //initialize at 0
            jj = iukp;
            int temp_ldu=0;
            for (; jj < iukp+nsupc; ++jj) {
                segsize = klst - usub[jj];
                if ( segsize ) {
		    ++ncols;
		}
                temp_ldu = SUPERLU_MAX(temp_ldu, segsize);
            }

            full_u_cols[j] = ncols;
            blk_ldu[j] = temp_ldu;
        } /* end for j = jj0..nub */

        jjj = jj0; /* jj0 is the first block column after look-ahead window */

	int parts = 1;
        // #pragma omp barrier
        while ( jjj < nub ) {
#if ( PRNTlevel>=1 )	
           if ( parts>1 ){
	      printf("warning: number of partitions %d > 1, try increasing MAX_BUFFER_SIZE.\n",
	              parts);
            }
#endif
            parts++;
            jjj_st=jjj;
#ifdef _OPENMP
#pragma omp single
#endif
            {
                ldu = blk_ldu[jjj_st];
                for (j = jjj_st; j < nub ; ++j) {

                    /* prefix sum */
                    if (j != jjj_st) full_u_cols[j] += full_u_cols[j-1];

                    ldu = SUPERLU_MAX(ldu, blk_ldu[j]);

                    /* break condition */
                    /* the number of columns that can be processed on GPU is
		       limited by buffer size */
                    if (full_u_cols[j]+((j+1==nub)?0:full_u_cols[j+1]) > ncol_max) {
                        break; // block column j+1 does not fit in GPU memory */
                    }
                } /* end for j=jjj_st to nub */

                jjj_global = SUPERLU_MIN(nub, j+1); /* Maximum value of jjj < nub */

                // TAU_STATIC_TIMER_START("work_divison");
                /* Divide CPU-GPU gemm here.
		 * If there is only one block, we leave it on CPU.
		 */
                gemm_division_cpu_gpu(
		       options,
		       &num_streams_used,/*number of streams that will be used*/
		       stream_end_col,   /*array holding last column blk for each partition*/
		       &ncpu_blks,       /*number of CPU gemm blks*/
		       		// Following are inputs
		       nbrow,            /*number of rows in A matrix*/
		       ldu,              /*value of k in dgemm*/
		       nstreams,
		       full_u_cols + jjj_st, /*array containing prefix sum of GPU workload*/
		       jjj_global - jjj_st, /*number of block columns on GPU.
		       		             If only one block, leave it on CPU*/
		       buffer_size
                );
                // TAU_STATIC_TIMER_STOP("work_divison");

            } /* pragma omp single */

            jjj = jjj_global; /* Move to the next [ CPU : GPU ] partition */
	    
#if 0 // !!Sherry: this test is not necessary
	    // if jjj_global - jjj_st == 1, everything is on CPU.
	    // bigv_size is calculated sufficiently large.
            if (jjj == jjj_st+1 && full_u_cols[jjj_st] > ncol_max) {
                printf("allocate more memory for buffer !!!!\n"
		       ".. jjj_st %d, nbrow %d, full_u_cols[jjj_st] %d, ncol_max %d\n",
		       jjj_st, nbrow, full_u_cols[jjj_st], ncol_max);
                if(nbrow * full_u_cols[jjj_st] > buffer_size)
                    printf("[%d] needed %d > buffer_size %d\n",iam,nbrow*full_u_cols[jjj_st],buffer_size );
		fflush(stdout);
            }
#endif

            // #pragma omp barrier
            /* gathering circuit */
            assert(jjj_st<nub);
            assert(jjj-1<nub);
            // TAU_STATIC_TIMER_START("GATHER_U");
	    
	    tt_start = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp for schedule( SCHEDULE_STRATEGY )
#endif
	    // Copy U segments into tempu, up to jjj_global block */	
            for (j = jjj_st; j < jjj; ++j) {
                if (j==jjj_st) tempu = bigU; /* leading block(s) on CPU */
                else tempu = bigU + ldu*full_u_cols[j-1];

                /* == processing each of the remaining columns == */
                arrive_at_ublock(j,&iukp,&rukp,&jb,&ljb,&nsupc,
				 iukp0,rukp0,usub,perm_u,xsup,grid);

                // copy block j into tempu
                for (jj = iukp; jj < iukp+nsupc; ++jj) {
                    segsize = klst - usub[jj];
                    if ( segsize ) {
                        lead_zero = ldu - segsize;
                        for (i = 0; i < lead_zero; ++i) tempu[i] = zero;
                        tempu += lead_zero;
                        for (i = 0; i < segsize; ++i)
                            tempu[i] = uval[rukp+i];
                        rukp += segsize;
                        tempu += segsize;
                    }
                }

                rukp -= usub[iukp - 1]; /* Return to start of U(k,j). */

            } /* end for j=jjj_st to jjj */

	    tt_end = SuperLU_timer_();
	    GatherUTimer += tt_end - tt_start;
	    
	    if ( num_streams_used > 0 ) {
#ifdef PI_DEBUG
		printf("nbrow %d *ldu %d  =%d < ldt %d * max_row_size %d =%d \n",nbrow,ldu,nbrow*ldu,ldt,max_row_size,ldt*max_row_size ); fflush(stdout);
		assert(nbrow*ldu<=ldt*max_row_size);
#endif
		gpuMemcpy2DAsync(dA, nbrow*sizeof(double),
				  &lusup[luptr+(knsupc-ldu)*nsupr],
				  nsupr*sizeof(double), nbrow*sizeof(double),
				  ldu, gpuMemcpyHostToDevice, streams[0]);
	    }

	    for (int i = 0; i < num_streams_used; ++i) { // streams on GPU
		int st = (i==0) ? ncpu_blks+jjj_st : jjj_st+stream_end_col[i-1];
		// st starts after the leading ncpu_blks
		int st_col = full_u_cols[st-1];
		int num_col_stream = full_u_cols[jjj_st+stream_end_col[i]-1]-full_u_cols[st-1];
		tempu = bigU;

		double *tempv1 = bigV + full_u_cols[st-1]*nbrow;

		/* Following is for testing purpose */
		if ( num_col_stream > 0 ) {
		
#ifdef GPU_ACC  /* Sherry: this file is not used if GPU_ACC is not defined. */
		    int stream_id = i;
		    int b_offset  = ldu * st_col;
		    int c_offset  = st_col * nbrow;
		    size_t B_stream_size = ldu * num_col_stream * sizeof(double);
		    size_t C_stream_size = nbrow * num_col_stream * sizeof(double);

		    // Sherry: Check dC buffer of *buffer_size* is large enough
		    assert(nbrow*(st_col+num_col_stream) < buffer_size);

		    gpuMemcpyAsync(dB+b_offset, tempu+b_offset, B_stream_size,
		    		    gpuMemcpyHostToDevice, streams[stream_id]);

		    gpublasCheckErrors(
				  gpublasSetStream(handle[stream_id],
						  streams[stream_id])
				     );

		    gpublasCheckErrors(
				  gpublasDgemm(handle[stream_id],
					      GPUBLAS_OP_N, GPUBLAS_OP_N,
					      nbrow, num_col_stream, ldu,
                                              &alpha, dA, nbrow,
					      &dB[b_offset], ldu,
					      &beta, &dC[c_offset],
                                              nbrow)
				  );

		    checkGPU( gpuMemcpyAsync(tempv1, dC+c_offset,
					   C_stream_size,
					   gpuMemcpyDeviceToHost,
					   streams[stream_id]) );
#else /*-- on CPU --*/
		} else { // num_col_stream == 0  Sherry: how can get here?
                    // Sherry: looks like a batched GEMM 
	            my_dgemm_("N", "N", &nbrow, &num_col_stream, &ldu,
			      &alpha, &lusup[luptr+(knsupc-ldu)*nsupr],
			      &nsupr, tempu+ldu*st_col, &ldu, &beta,
			      tempv1, &nbrow, 1, 1);
	        }
#endif /*-- end ifdef GPU_ACC --*/

   	      } // end if num_col_stream > 0

	    } /* end for i = 1 to num_streams used */

	    /* Special case for CPU -- leading block columns are computed 
	       on CPU in order to mask the GPU data transfer latency */
	    int num_col = full_u_cols[jjj_st+ncpu_blks-1];
	    int st_col = 0; /* leading part on CPU */
	    tempv = bigV + nbrow * st_col;
	    tempu = bigU;

	    double tstart = SuperLU_timer_();
#if defined (USE_VENDOR_BLAS)
	    dgemm_("N", "N", &nbrow, &num_col, &ldu, &alpha,
		  &lusup[luptr+(knsupc-ldu)*nsupr], &nsupr,
		  tempu+ldu*st_col, &ldu, &beta, tempv, &nbrow, 1, 1);
#else
	    dgemm_("N", "N", &nbrow, &num_col, &ldu, &alpha,
		  &lusup[luptr+(knsupc-ldu)*nsupr], &nsupr,
		  tempu+ldu*st_col, &ldu, &beta, tempv, &nbrow);
#endif
	    gemm_timer += SuperLU_timer_() -tstart;

	    /* The following counts both CPU and GPU parts.
	       full_u_cols[jjj-1] contains both CPU and GPU. */
	    stat->ops[FACT] += 2.0 * nbrow * ldu * full_u_cols[jjj-1];

            /* Now scattering blocks computed by CPU */
            int temp_ncol;

            /* scatter leading blocks which CPU has computated */
            tstart = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel  \
    private(j,iukp,rukp, tempu, tempv, cum_nrow, jb, nsupc,ljb,	\
	    segsize,lead_zero,					\
	    ib, temp_nbrow,ilst,lib,index,			\
	    ijb,fnz,ucol,rel,ldv,lptrj,luptrj,			\
	    nzval,     lb ,                     jj, i)		\
    firstprivate(luptr,lptr) default (shared)
#endif
            {
#ifdef _OPENMP	    
                int thread_id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
#else
                int thread_id = 0;
		int num_threads = 1;
#endif		

                int* indirect_thread = indirect + ldt*thread_id;
                int* indirect2_thread = indirect2 + ldt*thread_id;
                double* tempv1;

                if ( ncpu_blks < num_threads ) {
                    // TAU_STATIC_TIMER_START("SPECIAL_CPU_SCATTER");

                    for (j = jjj_st; j < jjj_st+ncpu_blks; ++j) {
                        /* code */
#ifdef PI_DEBUG
			printf("scattering block column %d, jjj_st, jjj_st+ncpu_blks\n",j,jjj_st,jjj_st+ncpu_blks);
#endif

                        /* == processing each of the remaining columns == */

                        if(j==jjj_st) tempv1 = bigV;
                        else tempv1 = bigV + full_u_cols[j-1]*nbrow;

                        arrive_at_ublock( j,&iukp,&rukp,&jb,&ljb,&nsupc,
					  iukp0,rukp0,usub,perm_u,xsup,grid );

                        cum_nrow =0 ;

                        /* do update with the kth column of L and (k,j)th block of U */
                        lptr = lptr0;
                        luptr = luptr0;

#ifdef _OPENMP
#pragma omp for schedule( SCHEDULE_STRATEGY ) nowait
#endif
                        for (lb = 0; lb < nlb; lb++ ) {
                            int cum_nrow = 0;
                            int temp_nbrow;
                            lptr = lptr0;
                            luptr = luptr0;
                            for (int i = 0; i < lb; ++i) {
                                ib = lsub[lptr];        /* Row block L(i,k). */
                                temp_nbrow = lsub[lptr+1];   /* Number of full rows. */
                                lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
                                lptr += temp_nbrow;
                                luptr += temp_nbrow;
                                cum_nrow +=temp_nbrow;
                            }

                            ib = lsub[lptr];       /* Row block L(i,k). */
                            temp_nbrow = lsub[lptr+1];  /* Number of full rows. */
                            assert(temp_nbrow<=nbrow);

                            lptr += LB_DESCRIPTOR; /* Skip descriptor. */

                            /* Now gather the result into the destination block. */
                            if ( ib < jb ) {  /* A(i,j) is in U. */
                                #ifdef PI_DEBUG
                                    printf("cpu scatter \n");
                                    printf("A(%d,%d) goes to U block %d \n", ib,jb,ljb);
                                #endif

                                tempv = tempv1+cum_nrow;
                                dscatter_u (
						 ib,jb,
						 nsupc,iukp,xsup,
						 klst,nbrow,
						 lptr,temp_nbrow,lsub,
						 usub,tempv,
						 Ufstnz_br_ptr,
						 Unzval_br_ptr,
						 grid
						 );
                            } else {    /* A(i,j) is in L. */
#ifdef PI_DEBUG
                                printf("cpu scatter \n");
                                printf("A(%d,%d) goes to L block %d \n", ib,jb,ljb);
#endif

                                tempv = tempv1+cum_nrow;

                                dscatter_l (
						 ib, ljb,nsupc,iukp,xsup,klst,nbrow,lptr,
						 temp_nbrow,usub,lsub,tempv,
						 indirect_thread,indirect2_thread,
						 Lrowind_bc_ptr,Lnzval_bc_ptr,grid
						 );
                            } /* if ib < jb ... */

                            lptr += temp_nbrow;
                            luptr += temp_nbrow;
                            cum_nrow += temp_nbrow;

                        } /* for lb ... */

                        luptr=luptr0;
                    } /* for j = jjj_st ... */

                    // TAU_STATIC_TIMER_STOP("SPECIAL_CPU_SCATTER");
                } else { // ncpu_blks >= omp_get_num_threads()
#ifdef _OPENMP
#pragma omp for schedule(SCHEDULE_STRATEGY) nowait
#endif
                    for (j = jjj_st; j < jjj_st+ncpu_blks; ++j) {
                        /* code */
#ifdef PI_DEBUG
			printf("scattering block column %d\n",j);
#endif

                        /* == processing each of the remaining columns == */
                        if(j==jjj_st) tempv1 = bigV;
                        else tempv1 = bigV + full_u_cols[j-1]*nbrow;

                        arrive_at_ublock( j,&iukp,&rukp,&jb,&ljb,&nsupc,
					  iukp0,rukp0,usub,perm_u,xsup,grid );
                        cum_nrow =0 ;

                        /* do update with the kth column of L and (k,j)th block of U */
                        lptr = lptr0;
                        luptr = luptr0;

                        for (lb = 0; lb < nlb; lb++ ) {
                            ib = lsub[lptr];       /* Row block L(i,k). */
                            temp_nbrow = lsub[lptr+1];  /* Number of full rows. */
                            assert(temp_nbrow<=nbrow);

                            lptr += LB_DESCRIPTOR; /* Skip descriptor. */
#ifdef DGEMM_STAT
			    if(j==jjj_st) {
				temp_ncol = full_u_cols[j];
			    } else {
				temp_ncol = full_u_cols[j]- full_u_cols[j-1];
			    }
			    printf("%d %d %d \n",temp_nbrow, temp_ncol,ldu);
#endif

			    /* Now gather the result into the destination block. */
			    if ( ib < jb ) {  /* A(i,j) is in U. */
#ifdef PI_DEBUG
				printf("cpu scatter \n");
				printf("A(%d,%d) goes to U block %d \n", ib,jb,ljb);
#endif

				tempv = tempv1+cum_nrow;
                                dscatter_u (
						 ib,jb,
						 nsupc,iukp,xsup,
						 klst,nbrow,
						 lptr,temp_nbrow,lsub,
						 usub,tempv,
						 Ufstnz_br_ptr,
						 Unzval_br_ptr,
						 grid
						 );
			    } else {    /* A(i,j) is in L. */
#ifdef PI_DEBUG
                                printf("cpu scatter \n");
                                printf("A(%d,%d) goes to L block %d \n", ib,jb,ljb);
#endif
                                tempv = tempv1+cum_nrow;

                                dscatter_l (
						 ib, ljb,nsupc,iukp,xsup,klst,nbrow,lptr,
						 temp_nbrow,usub,lsub,tempv,
						 indirect_thread,indirect2_thread,
						 Lrowind_bc_ptr,Lnzval_bc_ptr,grid
						 );
			    } /* if ib < jb ... */

			    lptr += temp_nbrow;
			    luptr += temp_nbrow;
			    cum_nrow += temp_nbrow;

			} /* for lb ... */

			luptr=luptr0;
		    } /* for j = jjj_st ... */
		}     /* else (ncpu_blks >= omp_get_num_threads()) */
	    }         /* parallel region */

	    scatter_timer += SuperLU_timer_() - tstart;
	    
	    // Scatter tempv(:, (jjj_st1 : jjj_global)) computed on GPU.
#ifdef _OPENMP
#pragma omp parallel							\
    private(j,iukp,rukp, tempu, tempv, cum_nrow, jb, nsupc,ljb,		\
	    segsize,lead_zero,						\
	    ib, temp_nbrow,ilst,lib,index,				\
	    ijb,fnz,ucol,rel,ldv,lptrj,luptrj,				\
	    nzval,     lb ,                     jj, i)			\
    firstprivate(luptr,lptr) default (shared)
#endif
            {
#ifdef _OPENMP	    
                int thread_id = omp_get_thread_num();
#else		
                int thread_id = 0;
#endif
                int* indirect_thread = indirect + ldt*thread_id;
                int* indirect2_thread = indirect2 + ldt*thread_id;
                double* tempv1;
                for(i = 0; i < num_streams_used; i++) { /* i is private variable */
                    checkGPU(gpuStreamSynchronize (streams[i]));
		    // jjj_st1 := first block column on GPU stream[i]
		    int jjj_st1 = (i==0) ? jjj_st + ncpu_blks : jjj_st + stream_end_col[i-1];
                    int jjj_end = jjj_st + stream_end_col[i];
                    assert(jjj_end-1<nub);
                    assert(jjj_st1>jjj_st) ;

                    /* now scatter it */
#pragma omp for schedule( SCHEDULE_STRATEGY ) nowait
                    for (j = jjj_st1; j < jjj_end; ++j) {
                        /* code */
#ifdef PI_DEBUG
			printf("scattering block column %d, jjj_end %d, nub %d\n",j,jjj_end,nub); fflush(stdout);
#endif
                        /* == processing each of the remaining columns == */

                        if(j==jjj_st) tempv1 = bigV;
                        else tempv1 = bigV + full_u_cols[j-1]*nbrow;

                        arrive_at_ublock( j,&iukp,&rukp,&jb,&ljb,&nsupc,
					  iukp0,rukp0,usub,perm_u,xsup,grid );
                        cum_nrow =0 ;

                        /* do update with the kth column of L and (k,j)th
			   block of U */
                        lptr = lptr0;
                        luptr = luptr0;
                        for (lb = 0; lb < nlb; lb++) {
                            ib = lsub[lptr];       /* Row block L(i,k). */
                            temp_nbrow = lsub[lptr+1];  /* Number of full rows. */
                            assert(temp_nbrow<=nbrow);

                            lptr += LB_DESCRIPTOR; /* Skip descriptor. */
#ifdef DGEMM_STAT
			    if(j==jjj_st) {
				temp_ncol = full_u_cols[j];
			    } else {
				temp_ncol = full_u_cols[j]- full_u_cols[j-1];
			    }
			    printf("%d %d %d \n",temp_nbrow, temp_ncol,ldu);
#endif

                            /* Now scatter result into destination block. */
                            if ( ib < jb ) { /* A(i,j) is in U. */
#ifdef PI_DEBUG
				printf("gpu scatter \n");
				printf("A(%d,%d) goes to U block %d \n", ib,jb,ljb);
				fflush(stdout);
#endif
                                tempv = tempv1+cum_nrow;
                                dscatter_u (
						 ib,jb,
						 nsupc,iukp,xsup,
						 klst,nbrow,
						 lptr,temp_nbrow,lsub,
						 usub,tempv,
						 Ufstnz_br_ptr,
						 Unzval_br_ptr,
						 grid
						 );
                            } else {    /* A(i,j) is in L. */
#ifdef PI_DEBUG
                                printf("gpu scatter \n");
                                printf("A(%d,%d) goes to L block %d \n", ib,jb,ljb);
				fflush(stdout);
#endif
                                tempv = tempv1+cum_nrow;

                                dscatter_l (
						 ib, ljb,nsupc,iukp,xsup,klst,nbrow,lptr,
						 temp_nbrow,usub,lsub,tempv,
						 indirect_thread,indirect2_thread,
						 Lrowind_bc_ptr,Lnzval_bc_ptr,grid
						 );
                            } /* if ib < jb ... */

                            lptr += temp_nbrow;
                            luptr += temp_nbrow;
                            cum_nrow += temp_nbrow;

                        } /* for lb ... */

                        luptr=luptr0;
                    } /* for j = jjj_st ... */

                } /* end for i = 0 to num_streams_used  */
		
                // TAU_STATIC_TIMER_STOP("GPU_SCATTER");
                // TAU_STATIC_TIMER_STOP("INSIDE_OMP");
		
            } /* end pragma omp parallel */
            // TAU_STATIC_TIMER_STOP("OUTSIDE_OMP");
	    
	    RemainScatterTimer += SuperLU_timer_() - tstart;

        }  /* end while(jjj<nub) */

    } /* if nbrow>0 */

 }   /* if msg1 and msg 2 */



