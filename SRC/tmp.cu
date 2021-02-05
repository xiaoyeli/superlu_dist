if (bid==1) { // for RD
int j, iam, lib, mycol, myrow, k, knsupc, il, cnt;
int_t fmod_tmp, aln_i;
aln_i = 1;
double temp;
if (WAIT_NUM_THREADS >= d_nfrecvmod[1]) { // one thread wait for one col
if (tid < d_nfrecvmod[1]) {
printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]]);
for (int i = 0; i < d_recv_cnt[d_colnummod[tid]]; i++) {
//printf("(%d,%d,%d) d_colnummod=%d,recv_cnt=%d,i=%d,wait_off=%d,%d,status=%d,%d\n", mype, bid, tid, d_colnummod[tid], d_recv_cnt[d_colnummod[tid]],i,d_colnummod[tid]*2, d_colnummod[tid]*2+1,d_statusmod[d_colnummod[tid]*2], d_statusmod[d_colnummod[tid]*2+1]);
int wm_val = nvshmem_int_wait_until_any(flag_rd_q + d_colnummod[tid] * 2, 2,
                                        d_statusmod + d_colnummod[tid] * 2, NVSHMEM_CMP_EQ, 1);
d_statusmod[d_colnummod[tid] * 2 + wm_val] = 1;
lib=(d_colnummod[tid] * 2 + wm_val)/2;

iam = grid->iam;
mycol = MYCOL(iam, grid);
myrow = MYROW(iam, grid);

k = myrow + lib * grid->nprow; // global block row
knsupc = SuperSize(k);
il = LSUM_BLK(lib);
printf("HERE2-(%d,%d,%d),lib=%d,k=%d\n", mype, bid, tid, lib, k);

cnt = LRtree_ptr[lib].destCnt_;
if (d_statusmod[lib*2]+d_statusmod[lib*2+1]==cnt) {
double tmp_sum = 0;
int ii=0;
////YL: wait for the one or two coming messages to complete using NVSHMEM, the received data is in ready_lsum[maxrecvsz*lib*2]
if (cnt == 2) {
for (ii = 0; ii < cnt; ++ii) {
tmp_sum = 0;
RHS_ITERATE(j) {
        for (i = 0; i < knsupc; ++i) {
            //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
            temp = atomicAdd(&lsum[il + i + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i +
                                        j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc];
            //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid,
            //       lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

        printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f\n", mype, bid, tid, lib, k,
        tmp_sum);
        fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
}
}
}
if (cnt == 1) {
if (flag_rd_q[k * 2 + 1] == 1) ii = 1;
RHS_ITERATE(j) {
        for (i = 0; i < knsupc; ++i) {
            //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
            temp = atomicAdd(&lsum[il + i + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i + j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc];
            //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

}
printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f\n", mype, bid, tid, lib, k, tmp_sum);
fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
}
}

if (fmod[lib*aln_i]==0){
//YL: this thread forwards the lsum subvector using NVSHMEM
if(LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_){
//cnt=LRtree_ptr[lib].msgSize_;
my_flag_rd[k*RDMA_FLAG_SIZE]=k;
my_flag_rd[k*RDMA_FLAG_SIZE+1]=LRtree_ptr[lib].msgSize_;
RHS_ITERATE(j) {
        for (int i = 0; i < knsupc; i++) {
            ready_lsum[k * maxrecvsz * 2 + i +j * knsupc] = lsum[il + i+j * knsupc];
            //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
            //       k * maxrecvsz * 2 + i +j * knsupc,
            //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

        }
}
printf("(%d,%d,%d),lib=%d,k=%d,myflagrd=%d,%d\n",mype,bid,tid,lib,k,my_flag_rd[k*RDMA_FLAG_SIZE],my_flag_rd[k*RDMA_FLAG_SIZE+1]);
C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &ready_lsum[0],maxrecvsz);
}
}
}//for
} else {
int delta = d_nfrecvmod[1] % WAIT_NUM_THREADS;
//d_mynummod: #col I wait.
if (tid < delta) {
d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS + 1;
} else {
d_mynummod[tid] = d_nfrecvmod[1] / WAIT_NUM_THREADS;
}
__syncthreads();

d_mymaskstartmod[tid] = 0;
d_msgnum[tid]=0;

//d_mymaskstartmod: start offset of d_colnummod
for (int i = 0; i < tid; i++) {
d_mymaskstartmod[tid] += d_mynummod[i];
//printf("(%d,%d,%d),i=%d,d_mynummod=%d,d_mymaskstartmod=%d\n",
//       mype,bid,tid,i,
//       d_mynummod[i],d_mymaskstartmod[tid]);
}
d_mymasklengthmod[tid] = 2* (d_colnummod[d_mymaskstartmod[tid] + d_mynummod[tid] - 1]
- d_colnummod[d_mymaskstartmod[tid]] + 1);
__syncthreads();
for (int i=d_mymaskstartmod[tid];i<d_mymaskstartmod[tid]+d_mynummod[tid];i++){
d_msgnum[tid]+= d_recv_cnt[d_colnummod[i]];
//printf("(%d,%d,%d),i=%d,d_recv_cnt=%d\n",mype,bid,tid,i,d_recv_cnt[d_colnummod[i]]);
}
//printf("(%d,%d,%d) waitcol=%d,msgnum=%d,masklength=%d,start=%d\n",mype,bid,tid,d_mynummod[tid],d_msgnum[tid],d_mymasklengthmod[tid],d_mymaskstartmod[tid]);
for (int i = 0; i < d_msgnum[tid]; i++) {
int wm_val = nvshmem_int_wait_until_any(flag_rd_q + d_colnummod[d_mymaskstartmod[tid]] * 2,
                                        d_mymasklengthmod[tid],
                                        d_statusmod + d_colnummod[d_mymaskstartmod[tid]] * 2,
                                        NVSHMEM_CMP_EQ, 1);
d_statusmod[d_colnummod[d_mymaskstartmod[tid]] + wm_val] = 1;
lib=(d_colnummod[d_mymaskstartmod[tid]] + wm_val)/2;
iam = grid->iam;
mycol = MYCOL(iam, grid);
myrow = MYROW(iam, grid);

k = myrow + lib * grid->nprow; // global block row
knsupc = SuperSize(k);
il = LSUM_BLK(lib);
printf("HERE2-(%d,%d,%d),lib=%d,k=%d\n", mype, bid, tid, lib, k);

cnt = LRtree_ptr[lib].destCnt_;
if (d_statusmod[lib*2]+d_statusmod[lib*2+1]==cnt) {
double tmp_sum = 0;
int ii=0;
////YL: wait for the one or two coming messages to complete using NVSHMEM, the received data is in ready_lsum[maxrecvsz*lib*2]
if (cnt == 2) {
for (ii = 0; ii < cnt; ++ii) {
tmp_sum = 0;
RHS_ITERATE(j) {
        for (i = 0; i < knsupc; ++i) {
            //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
            temp = atomicAdd(&lsum[il + i + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i +
                                        j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc];
            //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid,
            //       lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

        printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f\n", mype, bid, tid, lib, k,
        tmp_sum);
        fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
}
}
}
if (cnt == 1) {
if (flag_rd_q[k * 2 + 1] == 1) ii = 1;
RHS_ITERATE(j) {
        for (i = 0; i < knsupc; ++i) {
            //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
            temp = atomicAdd(&lsum[il + i + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i + j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc];
            //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

}
printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f\n", mype, bid, tid, lib, k, tmp_sum);
fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
}
}

if (fmod[lib*aln_i]==0){
//YL: this thread forwards the lsum subvector using NVSHMEM
if(LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_){
//cnt=LRtree_ptr[lib].msgSize_;
my_flag_rd[k*RDMA_FLAG_SIZE]=k;
my_flag_rd[k*RDMA_FLAG_SIZE+1]=LRtree_ptr[lib].msgSize_;
RHS_ITERATE(j) {
        for (int i = 0; i < knsupc; i++) {
            ready_lsum[k * maxrecvsz * 2 + i +j * knsupc] = lsum[il + i+j * knsupc];
            //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
            //       k * maxrecvsz * 2 + i +j * knsupc,
            //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

        }
}
printf("(%d,%d,%d),lib=%d,k=%d,myflagrd=%d,%d\n",mype,bid,tid,lib,k,my_flag_rd[k*RDMA_FLAG_SIZE],my_flag_rd[k*RDMA_FLAG_SIZE+1]);
C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &ready_lsum[0],maxrecvsz);
}
}

}//for

} // else WAIT_NUM_THREAD<recv
}
}



while(mydone!=d_rownum[tid]){
for (int i=d_rowstart[tid]; i<d_rowstart[tid]+d_rownum[tid] && d_validrows[i]!=-1; i++){
lib=i;
if (lib >= CEILING(nsupers, grid->nprow)) continue;
if (LRtree_ptr[lib].empty_ == YES) continue;
if (fmod[lib*aln_i]=!0) continue;

iam = grid->iam;
mycol = MYCOL(iam, grid);
myrow = MYROW(iam, grid);

k = myrow + lib * grid->nprow; // global block row
knsupc = SuperSize(k);
il = LSUM_BLK(lib);

if(LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_){
//cnt=LRtree_ptr[lib].msgSize_;
my_flag_rd[k*RDMA_FLAG_SIZE]=k;
my_flag_rd[k*RDMA_FLAG_SIZE+1]=LRtree_ptr[lib].msgSize_;
RHS_ITERATE(j) {
        for (int i = 0; i < knsupc; i++) {
            ready_lsum[k * maxrecvsz * 2 + i +j * knsupc] = lsum[il + i+j * knsupc];
            //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
            //       k * maxrecvsz * 2 + i +j * knsupc,
            //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

        }
}
printf("(%d,%d,%d),lib=%d,k=%d,myflagrd=%d,%d\n",mype,bid,tid,lib,k,my_flag_rd[k*RDMA_FLAG_SIZE],my_flag_rd[k*RDMA_FLAG_SIZE+1]);
C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int*)flag_rd_q, &my_flag_rd[RDMA_FLAG_SIZE*k], mype, bid, tid, &ready_lsum[0],maxrecvsz);
}


d_validrows[i]=-1;
mydone+=1;
}
}





