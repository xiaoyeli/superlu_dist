
for (int i = 0; i < d_msgnum[tid]; i++) {
if (tid<2) printf("(%d,%d,%d)--before wait any,i=%d/%d\n",mype,bid,tid,i,d_msgnum[tid]);
int wm_val = nvshmem_int_wait_until_any(&flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                        d_mymasklengthmod[tid],
                                        &d_statusmod[d_colnummod[d_mymaskstartmod[tid]] * 2],
                                        NVSHMEM_CMP_EQ, 1);
d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val] = 1;
lib = (d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val) / 2;
if (tid<2) printf("(%d,%d,%d)--recv a msg, offset=%d,base=%d, lib=%d, flag=%d,status=%d\n",mype,bid,tid,wm_val,d_colnummod[d_mymaskstartmod[tid]]*2,
lib,flag_rd_q[d_colnummod[d_mymaskstartmod[tid]] * 2+wm_val],d_statusmod[d_colnummod[d_mymaskstartmod[tid]]*2 + wm_val]);
iam = grid->iam;
mycol = MYCOL(iam, grid);
myrow = MYROW(iam, grid);

k = myrow + lib * grid->nprow; // global block row
knsupc = SuperSize(k);
il = LSUM_BLK(lib);
cnt = LRtree_ptr[lib].destCnt_;
if (tid<2) printf("HERE2-(%d,%d,%d),lib=%d,k=%d,wm_val=%d,cnt=%d,%d, mycnt=%d\n", mype, bid, tid, lib, k,
wm_val,cnt,d_recv_cnt[lib],d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1]);

if (d_statusmod[lib * 2] + d_statusmod[lib * 2 + 1] == cnt) {
double tmp_sum = 0;
int ii = 0;
if (cnt == 2) {
for (ii = 0; ii < cnt; ++ii) {
tmp_sum = 0;
RHS_ITERATE(j) {
        for (int aab = 0; aab < knsupc; aab++) {
            //temp=atomicAdd(&lsum[il+i + j*knsupc], ready_lsum[maxrecvsz*lib*2+ii*maxrecvsz + i + j*knsupc]  );
            temp = atomicAdd(&lsum[il + aab + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + aab +
                                        j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + aab + j * knsupc];
            //printf("data2-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid,
            //       lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

        // atomic return old val
        fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
        if (tid<2) printf("sum2-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k,
        tmp_sum,fmod_tmp);

}
}
}
if (cnt == 1) {
if (flag_rd_q[k * 2 + 1] == 1) ii = 1;
RHS_ITERATE(j) {
        for (i = 0; i < knsupc; ++i) {
            temp = atomicAdd(&lsum[il + i + j * knsupc],
                             ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i + j * knsupc]);
            tmp_sum += ready_lsum[maxrecvsz * k * 2 + ii * maxrecvsz + i + j * knsupc];
            //printf("data1-(%d,%d,%d),lib=%d,k=%d,ii=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, ii,
            //       maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc,
            //       ready_lsum[maxrecvsz * lib * 2 + ii * maxrecvsz + i + j * knsupc]);
        }

}
// atomic return old val
fmod_tmp = atomicSub(&fmod[lib * aln_i], 1);
if (tid<2) printf("sum1-(%d,%d,%d),lib=%d,k=%d,sum=%f,fmod_tmp=%d\n", mype, bid, tid, lib, k, tmp_sum,fmod_tmp);
}

if (fmod_tmp <= 1) {// forward RD
if (tid<2) printf("sum1-(%d,%d,%d),lib=%d, myRoot=%d\n", mype, bid, tid, lib,LRtree_ptr[lib].myRoot_);
if (LRtree_ptr[lib].myRoot_ != LRtree_ptr[lib].myRank_) {
//cnt=LRtree_ptr[lib].msgSize_;
my_flag_rd[k * RDMA_FLAG_SIZE] = k;
my_flag_rd[k * RDMA_FLAG_SIZE + 1] = LRtree_ptr[lib].msgSize_;
RHS_ITERATE(j) {
        for (int aab = 0; aab < knsupc; aab++) {
            ready_lsum[k * maxrecvsz * 2 + aab + j * knsupc] = lsum[il + aab + j * knsupc];
            //printf("data3-(%d,%d,%d),lib=%d,k=%d,i=%d,ready_lsum[%d]=%f\n", mype, bid, tid, lib, k, i,
            //       k * maxrecvsz * 2 + i +j * knsupc,
            //       ready_lsum[k * maxrecvsz * 2 + i +j * knsupc]);

        }
}
printf("(%d,%d,%d),in wait lib=%d,k=%d,myflagrd=%d,%d\n", mype, bid, tid, lib, k,
my_flag_rd[k * RDMA_FLAG_SIZE], my_flag_rd[k * RDMA_FLAG_SIZE + 1]);
C_RdTree_forwardMessageSimple_Device(&LRtree_ptr[lib], (int *) flag_rd_q,
&my_flag_rd[RDMA_FLAG_SIZE * k], mype, bid, tid,
&ready_lsum[0], maxrecvsz);
}
}
}
}//for