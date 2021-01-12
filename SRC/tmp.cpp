//
// Created by NanDing on 9/28/20.
//

#include "tmp.h"

if(bid<nbcol_loc){


    if(Lrowind_bc_offset[bid]==-1){
    return;
    }

    lk=bid;
    iam = grid->iam;
    mycol = MYCOL( iam, grid );
    myrow = MYROW( iam, grid );
    k = mycol+lk*grid->npcol;
    knsupc = SuperSize( k );
    lsub = &Lrowind_bc_dat[Lrowind_bc_offset[lk]];
    iam = grid->iam;
    krow = PROW( k, grid );
    lusup = &Lnzval_bc_dat[Lnzval_bc_offset[lk]];
    lloc = &Lindval_loc_bc_dat[Lindval_loc_bc_offset[lk]];
    nsupr = lsub[1];

    if(myrow==krow){
        nlb = lsub[0] - 1;
        idx_n = 1;
        idx_i = nlb+2;
        idx_v = 2*nlb+3;
        luptr_tmp = lloc[idx_v];
        m = nsupr-knsupc;
    }else{
        nlb = lsub[0];
        idx_n = 0;
        idx_i = nlb;
        idx_v = 2*nlb;
        luptr_tmp = lloc[idx_v];
        m = nsupr;
    }

    //printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx_x, blockIdx_x, grid->npcol, nsupers,myrow,krow,bid,tid);

    if(myrow==krow){   /* diagonal block performs trsm and forward the message*/
        if(tid==0){  /*only the first thread in a block handles the lock */
            //printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
            lib = LBi( k, grid ); /* Local block number, row-wise. */
            do{
                tmp=fmod[lib*aln_i];
                __threadfence();
            }while(tmp>0);
        } //if tid==0
        __syncthreads();

        lib = LBi( k, grid ); /* Local block number, row-wise. */
        il = LSUM_BLK( lib );
        ii = X_BLK( lib );

        RHS_ITERATE(j)
            for (i = tid; i < knsupc; i+=block_size)
                x[i + ii + j*knsupc] += lsum[i + il + j*knsupc ];
        __syncthreads();

        Linv = &Linv_bc_dat[Linv_bc_offset[lk]];

        if(nrhs==1){
            for (i = tid; i < knsupc; i+=block_size){
                temp1=zero;
                for (l=0 ; l<knsupc ; l++){
                    temp1+=  Linv[l*knsupc+i]*x[ii+l];
                }
                lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
            }
            //__syncthreads();

            for (i = tid; i < knsupc; i+=block_size){
                x[i + ii] = lsum[il+i];
                //printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
            }
            // __syncthreads();
            // RHS_ITERATE(j){

            // for (i = tid; i < knsupc; i+=block_size)
            // rtemp_loc[i]=zero;
            // __syncthreads();


            // gemv_device_dlsum_fmod(
            // knsupc, knsupc, alpha,
            // Linv, knsupc,
            // &x[ii+j*knsupc], 1, beta,
            // rtemp_loc, 1);

            // __syncthreads();
            // // printf("tid %5d knsupc %5d block_size %5d\n",tid,knsupc,block_size);
            // for (i = tid; i < knsupc; i+=block_size){
            // x[i + ii + j*knsupc] = rtemp_loc[i];
            // // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
            // }
            // }
            // __syncthreads();

        }else{
            __syncthreads();
            for (int_t blx = 0; blx*BLK_M < knsupc; blx++){
                for (int_t bly = 0; bly*BLK_N < nrhs; bly++){
                    gemm_device_dlsum_fmod(knsupc, nrhs, knsupc, blx, bly,
                                            Linv, knsupc, &x[ii], knsupc, rC,
                                            alpha, beta);
#pragma unroll
                    for (ni = 0; ni < THR_N; ni++) {
                        int_t coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
#pragma unroll
                        for (mi = 0; mi < THR_M; mi++) {
                            int_t coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                            if (coord_dCm < knsupc && coord_dCn < nrhs) {
                                double &regC = rC[ni][mi];
                                lsum[coord_dCm + il + coord_dCn*knsupc ]=regC;  //reuse lsum as temporary output as it's no longer accessed
                            }//if (coord_dCm < knsupc && coord_dCn < nrhs)
                        }
                    }
                }
            }
            __syncthreads();

        RHS_ITERATE(j)
            for (i = tid; i < knsupc; i+=block_size)
                x[i + ii + j*knsupc] = lsum[i + il + j*knsupc ];
        __syncthreads();
        }//if(nrhs==1)


        RHS_ITERATE(j)
            for (i = tid; i < knsupc; i+=block_size)
                ready_x[i + maxrecvsz*lk + j*knsupc ] = x[i + ii + j*knsupc];
        __syncthreads();
    }else{   /* off-diagonal block forward the message*/
    /* waiting for the x subvector and forward*/
        if(tid==0){  //YL: only the first thread in a block spin-waits for the coming x subvector message using NVSHMEM, put the message into ready_x[maxrecvsz*lk]

        }
    }


    if(tid==0){  //YL: only the first thread in a block forwards the x subvector using NVSHMEM
        cnt=LBtree_ptr[lk].destCnt_;
        //printf("good1 %5d%5d\n",lk,cnt);
        if(cnt>0){
            cnt=LBtree_ptr[lk].msgSize_;
            C_BcTree_forwardMessageSimple_Device(&LBtree_ptr[lk],&ready_x[maxrecvsz*lk],cnt*nrhs+XK_H);
        }
    }


    if(nlb>0){
        lib = LBi( k, grid ); /* Local block number, row-wise. */
        ii = X_BLK( lib );

        if(nrhs==1){
            luptr_tmp1 = lloc[idx_v];
            lb = 0;
            nbrow=0;
            lptr1_tmp = lloc[lb+idx_i];
            lptr= lptr1_tmp+2;
            nbrow1 = lsub[lptr1_tmp+1];
            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
            rel = xsup[ik]; /* Global row index of block ik. */
            lk = LBi( ik, grid ); /* Local block number, row-wise. */
            iknsupc = SuperSize( ik );
            il = LSUM_BLK( lk );

            for (i = tid; i < m; i+=block_size){
                while(nbrow+lsub[lptr1_tmp+1]<=i){
                    lb++;
                    nbrow +=lsub[lptr1_tmp+1];
                    lptr1_tmp = lloc[lb+idx_i];
                    lptr= lptr1_tmp+2;
                    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                    rel = xsup[ik]; /* Global row index of block ik. */
                    lk = LBi( ik, grid ); /* Local block number, row-wise. */
                    iknsupc = SuperSize( ik );
                    il = LSUM_BLK( lk );
                }
                irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
                RHS_ITERATE(j){
                        temp1=zero;
                        for (l=0 ; l<knsupc ; l++){
                            temp1+= lusup[luptr_tmp1+l*nsupr+i]*x[ii+j*knsupc+l];
                        }
                        temp=atomicAdd(&lsum[il+irow + j*iknsupc],-temp1);
                }
                if(i==nbrow+lsub[lptr1_tmp+1]-1){
                    fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
                    // __threadfence();
                }
            }
            __syncthreads();

        }else{ //if nrhs ==1
            for (lb = 0; lb < nlb; lb++){
                luptr_tmp1 = lloc[lb+idx_v];
                lib = LBi( k, grid ); /* Local block number, row-wise. */
                ii = X_BLK( lib );

                lptr1_tmp = lloc[lb+idx_i];
                lptr= lptr1_tmp+2;
                nbrow1 = lsub[lptr1_tmp+1];
                ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                rel = xsup[ik]; /* Global row index of block ik. */

                lk = LBi( ik, grid ); /* Local block number, row-wise. */

                iknsupc = SuperSize( ik );
                il = LSUM_BLK( lk );

                for (int_t blx = 0; blx*BLK_M < nbrow1; blx++){
                    for (int_t bly = 0; bly*BLK_N < nrhs; bly++){
                        gemm_device_dlsum_fmod(nbrow1, nrhs, knsupc, blx, bly,
                                                &lusup[luptr_tmp1], nsupr, &x[ii], knsupc, rC,
                                                alpha, beta);
#pragma unroll
                        for (ni = 0; ni < THR_N; ni++) {
                            int_t coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
#pragma unroll
                            for (mi = 0; mi < THR_M; mi++) {
                                int_t coord_dCm = blx*BLK_M + mi*DIM_X + idx;
                                if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
                                    irow = lsub[lptr+coord_dCm] - rel; /* Relative row. */
                                    double &regC = rC[ni][mi];
                                    temp=atomicAdd(&lsum[il+irow + coord_dCn*iknsupc],-regC);
                                }
                            }
                        }
                    }
                }

                if(tid==0)fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
            }
        }//if(nrhs==1)

        __syncthreads();
    } /* if nlb>0*/

// printf("nimbgood \n");

}else if(bid<nbcol_loc+nblock_ex){  //the next nblock_ex blocks handle all reduction communication
    int_t bid1 = bid-nbcol_loc;

    iam = grid->iam;
    mycol = MYCOL( iam, grid );
    myrow = MYROW( iam, grid );

    lib = bid1*block_size+tid; // the local numbering of my block row
    k = myrow+lib*grid->nprow;
    knsupc = SuperSize( k );
    il = LSUM_BLK( lk );


    if(lib>=CEILING(nsupers, grid->nprow) return;
    if(LRtree_ptr[lib].empty_==YES) return;

    cnt = LRtree_ptr[lib].destCnt_;

    //YL: wait for the one or two coming messages to complete using NVSHMEM, the received data is in ready_lsum[maxrecvsz*lib*2]

    for (ii = 0; ii < cnt; ++ii){
        RHS_ITERATE(j) {
            for (i = 0; i < knsupc; ++i)
                temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
        }
        fmod_tmp=atomicSub(&fmod[lib*aln_i],1);
    }

    do{
        tmp=fmod[lib*aln_i];
        __threadfence();
    }while(tmp>0);


    //YL: this thread forwards the lsum subvector using NVSHMEM
    if(LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_){
        cnt=LRtree_ptr[lib].msgSize_;
    	int tmp_myoff;
    	if(LRtree_ptr[lib].myIdx %2 ==0){
    	   tmp_myoff = lib*RDMA_FLAG_SIZE*2;
    	}else{
    	   tmp_myoff = lib*RDMA_FLAG_SIZE*2+RDMA_FLAG_SIZE;
    	}
        C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib],&lsum[il - LSUM_H ],cnt*nrhs+LSUM_H);
    }

} // else if (bid<)
