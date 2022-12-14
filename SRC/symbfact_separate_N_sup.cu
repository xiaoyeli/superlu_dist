#ifndef SYMBFACT_SEPARATE_H
#define SYMBFACT_SEPARATE_H
// #include "wtime.h"
#include <omp.h>
#include <cuda.h>
#include <math.h> 
#include <cstdio> 
#include <iostream>
#include <vector> 
#include <fstream>
#include <algorithm> 
#include <fstream>
#include <cstring>
#include <string>
#include "superlu_ddefs.h"
#include "symbfact_separate_N_sup.cuh"
// #define parallel_prefix_sum 1
// #define print_fillins 1
// #define DEBUG_relaxation 1
  
  
using namespace std;
typedef std::vector<int> vec;

#define PROW(bnum,grid) ( (bnum) % grid->nprow )
#define PCOL(bnum,grid) ( (bnum) % grid->npcol )
#define PNUM(i,j,grid)  ( (i)*grid->npcol + j ) /* Process number at coord(i,j) */
#define MAX_VAL UINT_MAX
#define MIN_VAL 0
#define initializeCSR 1
//  #define SortLsubUsubIndicies 1 
// #define pass_throughLogic 1
// #define pass_throughLogic1 1

#define GPUOffset 0
#define _OPENMP 1
#define enable_memadvise 1
#define create_supernodal_LU 1
#define detect_supernodes 1
#define write_LU_struct 1
#define switchLU_write 1
#define debug_fill_count
// #define enable_original_nz_flag 1
// #define supernode_relaxation 2
#define Enable_supernodal_relaxation 1
// #define debug_gsofa_supernode 1


// #define lambda 1
// #define profileLsubUsubClocks 1
// #define enable_actual_UCount 1
// #define _M_LN2  0.693147180559945309417 // Natural log of 2
// #define log_2(x) (log(x)/_M_LN2)
// #define enable_debug 1
////////////-----------------------//////////////
//////////// Only 1 of the following two variables should be defined. Not both///////////////////////////////
// #define enable_fillins_filter_FQ_No_Max_id_update 1

int compare (const void * a, const void * b)
{
    if ( *(int*)a <  *(int*)b ) return -1;
    if ( *(int*)a == *(int*)b ) return 0;
    if ( *(int*)a >  *(int*)b ) return 1;
    return 0;
} 

void qsort_cuda(int* RecvRepresentativeBuffer, int TotalNSupernode)
{

    qsort(RecvRepresentativeBuffer, TotalNSupernode, sizeof(int), compare);
}

__global__ void PrefixAdd(int* g_idata,int* g_odata,int* BlockSum,int n)
{
    __shared__ int temp[256];//variable that stores the product of an element in M and element in B(vector)
    //	extern  __shared__ int mark[];
    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int count=0;
    // if (thid>=n) return;
    while(thid<n)
    {
        int pout =0, pin =1;

        temp[thid%blockDim.x] = g_idata[thid];// (thid > 0) ? g_idata[thid-1] : 0;
        temp[thid%blockDim.x+blockDim.x]=0;
        //  int pout=0,pin=1;
        __syncthreads();

        for (int offset = 1; offset < n; offset <<= 1)
        {
            pout=1-pout;
            pin=1- pout;
            if (thid%blockDim.x >= offset)
            {

                temp[pout*blockDim.x+thid%blockDim.x] = temp[pin*blockDim.x+thid%blockDim.x] + temp[pin*blockDim.x+thid%blockDim.x - offset];

            }
            else
            {
                temp[pout*blockDim.x+thid%blockDim.x]=temp[pin*blockDim.x+thid%blockDim.x];

            }
            __syncthreads();
            //	thid+=offset;
        }
        //	__syncthreads();
        g_odata[thid]=temp[pout*blockDim.x+thid%blockDim.x];

        __syncthreads();
        if (threadIdx.x==blockDim.x-1)
        {
            BlockSum[blockIdx.x+count*gridDim.x]=temp[thid%blockDim.x];
        }
        thid+=blockDim.x*gridDim.x;
        count++;
    }

    return;
}

__global__ void PrefixAddIntraBlock(int* g_odata,int* BlockPrefixSum,int n)
{

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int count = 0;
    while(thid<n)
    {
        int Bid=(count)*gridDim.x+blockIdx.x;//thid/blockDim.x;

        if (thid!=threadIdx.x)
        {
            g_odata[thid]+=BlockPrefixSum[Bid-1];
        }
        thid+=blockDim.x*gridDim.x;
        count++;			
    }

    return;
}


void prefix_sum(int_t* A, int_t* B, int_t N, int_t* A_d, int_t* B_d)
{
    int_t * BlockSum_d;
    int_t * BlockPrefixSum_d;
    // int_t size = (N/(float)128);
    int_t size = (N/(float)128)+1; //For N < 128, the size becomes 0. So, we add 1 to allocate enough memory.
    // printf("At Line: %d\n",__LINE__);
    // fflush(stdout);
    H_ERR(cudaMalloc((void**) &BlockSum_d,sizeof(int_t)*size));
    // printf("At Line: %d size: %d\n",__LINE__, size);
    // fflush(stdout);
    H_ERR(cudaMalloc((void**) &BlockPrefixSum_d,sizeof(int_t)*size));
    // printf("At Line: %d\n",__LINE__);
    // fflush(stdout);
    H_ERR(cudaMemset(B_d, 0, sizeof(int_t)*N));
    // printf("At Line: %d\n",__LINE__);
    // fflush(stdout);

    int_t* BlockSum=(int_t*)malloc((sizeof(int_t))*size);
    int_t* BlockPrefixSum=(int_t*)malloc((sizeof(int_t)*size));

    H_ERR(cudaMemcpy(A_d,A,sizeof(int_t)*N,cudaMemcpyHostToDevice));
    // printf("Calling prefix sum GPU module\n");
    // fflush(stdout);

    PrefixAdd<<<128,128>>>(A_d,B_d,BlockSum_d,N);
    H_ERR(cudaDeviceSynchronize());

    H_ERR(cudaMemcpy(BlockSum,BlockSum_d,sizeof(int_t)*size,cudaMemcpyDeviceToHost));

    int_t sumBlock=0;
    // std::cout<<"CPU prefix block sum:\n";
    for (int_t i=0;i<size;i++)
    {
        sumBlock+=BlockSum[i];
        BlockPrefixSum[i]=sumBlock;
        // cout<<BlockPrefixSum[i]<<" ";
    }
    // printf("At line:%d\n",__LINE__);
    // fflush(stdout);
    cudaMemcpy(BlockPrefixSum_d,BlockPrefixSum,sizeof(int_t)*size,cudaMemcpyHostToDevice);
    PrefixAddIntraBlock<<<128,128>>>(B_d,BlockPrefixSum_d,N);
    cudaMemcpy(B,B_d,sizeof(int_t)*N,cudaMemcpyDeviceToHost);
    // printf("At line:%d\n",__LINE__);
    // fflush(stdout);
#ifdef enable_prefix_test
    int sum=A[0];
    int_t *B_s= (int_t*)malloc((sizeof(int_t))*N);
    B_s[0]=sum;
    for(int i = 1; i < N; i++)
    {
        B_s[i] = sum + A[i];
        sum = B_s[i];
    }
    for(int i=0;i<N;i++)
    {
        if (B[i]!=B_s[i])
        {
            printf("ERR in prefix sum\n");
            getchar();
        }
        // cout<<B[i]<<" ";
    }
#endif
    printf("Finished prefix sum GPU module\n");
    fflush(stdout);
    // return;
}

__device__ __forceinline__ int ThreadLoad(int *ptr)
{
    int retval;           
    asm volatile ("ld.global.cg.s32 %0, [%1];" :    \
            "=r"(retval) :                        \
            "l" (ptr) );                          \
        return retval;  
}

__host__ __device__ __forceinline__ void swap_ptr_index(int_t** a, int_t** b)
{
    int_t* temp = *a;
    *a = *b;
    *b = temp;

}


__device__ __forceinline__ int_t Minimum(int_t a, int_t b)
{
    if (a<b) return a;
    else return b;
}

__host__ __device__ __forceinline__ int_t Maximum(uint_t a, uint_t b)
{
    return (a < b)? b:a;
}

// sync_X_block(0,gridDim.x,lock_d,1);
__device__ __forceinline__ void sync_X_block(int group_id, int N_blocks_source,int* d_lock, int N_groups) 
{

    volatile int *lock = d_lock;    

    // Threadfence and syncthreads ensure global writes 
    // thread-0 reports in with its sync counter
    __threadfence();
    __syncthreads();
    //                int group_bid= blockIdx.x & (N_blocks_source-1);//block id in the group
    int group_bid = blockIdx.x % N_blocks_source;
    int block_offset=group_id*N_blocks_source;
    if (group_bid== 0)//First block in the group
    {
        // Report in ourselves
        if (threadIdx.x == 0)
            lock[group_bid+block_offset] = 1;

        __syncthreads();

        // Wait for everyone else to report in
        //NOTE: change for more than 4 blocks
        int stop_block;
        if(group_id==N_groups-1)
        {
            stop_block=gridDim.x;
        }
        else
        {
            stop_block=block_offset+ N_blocks_source;
        }
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            while (ThreadLoad(d_lock + peer_block) == 0)
                __threadfence_block();

        __syncthreads();

        // Let everyone know it's safe to proceed
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            lock[peer_block] = 0;
    }
    else
    {
        if (threadIdx.x == 0)
        {
            // Report in
            // lock[blockIdx.x] = 1;
            lock[group_bid+block_offset] = 1;


            // Wait for acknowledgment
            //                         while (ThreadLoad (d_lock + blockIdx.x) != 0)
            while (ThreadLoad (d_lock + group_bid+block_offset) != 0)
                __threadfence_block();
            
        }
        __syncthreads();
    }
}


__device__ __forceinline__ int_t ShMem_offset (int_t warp_id, int_t max_supernode_size)
{
    // return ((warp_id*max_supernode_size) +1);
    return (warp_id*(max_supernode_size+1));
}

__global__ void make_exclusive(int_t* l_offset,int_t N,int_t* u_offset)
{
    int thid = blockIdx.x*blockDim.x+threadIdx.x;
    while(thid<N-1)
    {
        u_offset[thid+1]=l_offset[thid];
        thid += (blockDim.x*gridDim.x);
    }
    if (blockIdx.x*blockDim.x+threadIdx.x==0) u_offset[0] =0;
}

__device__ __forceinline__ int_t Compute_r_id_resecheduled(int local_chunk_id, int max_supernode_size, int j, int N_GPU_gSoFa_process, int* owner_gpu, int* index_in_C)
{
    //Note: index_in_C is index of source list of concurrent sources the GPU is working on

    *index_in_C = local_chunk_id * max_supernode_size + j;
    *owner_gpu = *index_in_C%N_GPU_gSoFa_process; //GPU that has information for the source src
    // int pos = (ptr*(local_chunk_id+1))/N_GPU_gSoFa_process;//position of the respective source in the owner GPU
    return (*index_in_C/N_GPU_gSoFa_process);//position of the respective source in the owner GPU

}


void SynchronizeAllDevices(cudaStream_t* stream,int_t N_gpu)
{
    for (int i = N_gpu-1; i >= 0; i--)
    {      
        H_ERR(cudaStreamSynchronize(stream[i]));      
    }
}

void calculate_exactoffsets(int_t* xlsub_begin,int_t *xlsub_end,int_t *xusub_begin,int_t *xusub_end,int_t* xlsub_send_count,int_t* xusub_send_count, int_t vert_count,int_t myrank)
{
    for (int i=0; i<vert_count; i++)
    {
        xlsub_send_count[i] = xlsub_end[i]-xlsub_begin[i];
        xusub_send_count[i] = xusub_end[i]-xusub_begin[i];
        printf("myrank:%d xusub_end[%d]:%d  ,  xusub_begin[%d]:%d, xusub_send_count[%d]:%d\n",myrank,i,xusub_end[i],i,xusub_begin[i],i,xusub_send_count[i]);
    }
}

__global__ void Initialise_cost_array(uint_t* cost_array_d,
        int_t vert_count,uint_t group_MAX_VAL, int_t N_src_group)
{
    int offset=blockDim.x*gridDim.x;
    int total_initialization=vert_count*N_src_group;
    for (int thid=blockDim.x*blockIdx.x+threadIdx.x; thid < total_initialization;thid+=offset)
    {
        cost_array_d[thid]=group_MAX_VAL;
    }
}

__global__ void Initialise_lsub_usub(int_t* lsub, int_t* usub,
    int_t vert_count,int_t* u_offset, int_t* l_offset, int_t* col_st, int_t* col_ed, int_t* csr)
{
    int offset=blockDim.x*gridDim.x;
    int_t offsetL, offsetU;
    // int total_initialization=vert_count*N_src_group;
    for (int vertex=blockDim.x*blockIdx.x+threadIdx.x; vertex < vert_count; vertex+=offset)
    {
        offsetU= atomicAdd(&u_offset[vertex],1);    
        usub[offsetU]=vertex;  // Put source (diagonal nz) on U 
        for (int neighbor_id = col_st[vertex]; neighbor_id < col_ed[vertex]; neighbor_id++)
        {
               
            int_t neighbor = csr[neighbor_id];
            if (neighbor < vertex)
            {
                offsetL= atomicAdd(&l_offset[vertex],1);     
                lsub[offsetL] = neighbor; // Put neighbor on L
            }
            else if (neighbor > vertex)
            {
                offsetU= atomicAdd(&u_offset[vertex],1);     
                usub[offsetU] = neighbor; // Put neighbor on U
            }
        }       
    }
}

__global__ void Compute_fillins_joint_traversal_group_wise_supernode_OptIII_warp_centric (uint_t* cost_array_d,
        int_t* fill_in_d,int_t* frontier,
        int_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        int_t* src_frontier_d,int_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group, int_t group,uint_t max_id_offset, 
        int_t* next_front_d, int_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d,int_t* frontierchecked/*, int_t* offset_next_kernel*/,int_t* swap_GPU_buffers_m,
        int_t* nz_row_U_d, int_t max_supernode_size, int_t N_chunks,int_t* source_flag_d,int_t* my_supernode_d, Super* superObj,
        int_t* validity_supernode_d, int_t* pass_through_d,int_t* fill_count_per_row_d,ull_t* N_edge_checks_per_thread_d,
        int_t* new_col_ed_d, int_t N_GPU_gSoFa_process, int_t local_gpu_id,ull_t* time_LsubUsub, ull_t* time_TotalClockCycles, 
        int_t* fill_sup_merge,int_t* lsub, int_t* usub, int_t* l_offset,int_t* u_offset,int* actual_row_count_U,int* actual_col_count_U,
        int* actual_row_count_L, int* actual_col_count_L,int valid_index, int_t* processed_vertex,int_t* is_OriginalNZ_L, int_t* is_OriginalNZ_U )
{
#ifdef profileLsubUsubClocks
    ull_t start_time_LsubUsub=0;
    ull_t start_time_TotalClockCycles =clock64();
#endif

    int_t level=0; //test for average frontier size
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t original_thid=thid;    
    // if (original_thid==0)  printf("GPU: %d is entering the kernel!\n",local_gpu_id);
    int_t warpId = thid >> 5;
    int_t laneID= threadIdx.x & 0x1f;
    int_t N_warps = (blockDim.x * gridDim.x) >> 5;
    while (thid < N_src_group)
    {
        int_t intranode_offset = thid;
        pass_through_d[intranode_offset] = 0;  
        nz_row_U_d[intranode_offset]=0; // For supernodes
        thid+=(blockDim.x*gridDim.x);
    }
    // if (original_thid==0) printf("GPU: %d is entering the M0 section!\n",local_gpu_id);
    sync_X_block(0,gridDim.x,lock_d,1);
    // if (original_thid==0) printf("GPU: %d is entering the M0_I section!\n",local_gpu_id);
    //Assign cost of the sources as min val
    for (thid=original_thid; thid < N_src_group; thid+=(blockDim.x*gridDim.x))
    {
        int_t cost_array_offset = thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
        int_t fill_array_offset = (thid)*vert_count;// When allocating combined fill array for GPUs in a node
        int source =source_d[thid];

        // if (source==0) printf("GPU:%d working on source:%d\n",gpu_id,source);
        // printf("GPU:%d working on source:%d valid_index:%d thid:%d\n",gpu_id,source,valid_index,thid);
        
        // if (source < vert_count)
        if (thid < valid_index)
        {
            // printf("GPU:%d Accessing cost_array_d at offset:%d for source:%d \n",gpu_id, cost_array_offset+source, source);
            cost_array_d[cost_array_offset+source] = max_id_offset; //max_id_offset=MIN_VAL for the current group
            // printf("GPU:%d Accessed cost_array_d at offset:%d for source:%d \n",gpu_id, cost_array_offset+source, source);
// #ifdef write_LU_struct
            //Writing the source into both the lsub abd usub
#ifdef profileLsubUsubClocks
            start_time_LsubUsub=clock64();
#endif

#ifndef initializeCSR    
            int offset;
            // printf("GPU:%d Accessing u_offset at position:%d for source:%d \n",gpu_id, source, source);    
            offset= atomicAdd(&u_offset[source],1);                
            usub[offset]=source;
            #ifdef enable_original_nz_flag
            is_OriginalNZ_U[offset]=1;
            #endif
#endif 

#ifdef profileLsubUsubClocks
            time_LsubUsub[original_thid] += (clock64()-start_time_LsubUsub);
#endif
// #endif
            //Done when detecting supernodes
            fill_in_d[fill_array_offset+ source] = source;//Done when detecting supernodes
        }
    }
    for (int_t src_id=blockIdx.x ; src_id < N_src_group; src_id +=gridDim.x)
    {
        int_t source=source_d[src_id]; 
        if (src_id < valid_index)
        {
            // processed_vertex[source] = 1;
            for (int_t b_tid=threadIdx.x+col_st[source]; b_tid < col_ed[source]; b_tid+=blockDim.x)
            {
                int_t neighbor=csr[b_tid];
  #ifdef profile_edge_checks
                N_edge_checks_per_thread_d[original_thid]++;
#endif
                int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;                
                int_t fill_array_offset = (src_id)*vert_count;// When allocating combined fill array for GPUs in a node
                cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group
                fill_in_d[fill_array_offset+ neighbor]=source;           
                if (neighbor < source) 
                {
                    //Note L structures need to be expanded because the non-zero in column wise will be increased on moving from lower rows to the upper row indices
                    //Putting source into the L structure for neighbor
// #ifdef write_LU_struct
#ifdef profileLsubUsubClocks
                    start_time_LsubUsub=clock64();
#endif
#ifndef initializeCSR    
                    int offset = atomicAdd(&l_offset[source],1);
                    lsub[offset]=neighbor;
                    #ifdef enable_original_nz_flag
                    is_OriginalNZ_L[offset]=1;
                    #endif
#endif
#ifdef profileLsubUsubClocks
                    time_LsubUsub[original_thid] += (clock64()-start_time_LsubUsub);
#endif
                  

// #endif

#ifdef enable_actual_UCount
                    atomicAdd(&actual_row_count_L[source],1);
                    atomicAdd(&actual_col_count_L[neighbor],1);

#endif

                    int_t front_position=atomicAdd(frontier_size,1);            
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                } 
#ifdef detect_supernodes
                else if (neighbor > source)
                    // else
                {

#ifdef profileLsubUsubClocks
                    start_time_LsubUsub=clock64();
#endif
#ifndef initializeCSR    
                    int offset= atomicAdd(&u_offset[source],1);
                    usub[offset]=neighbor;
                    #ifdef enable_original_nz_flag
                    is_OriginalNZ_U[offset]=1;
                    #endif
#endif
#ifdef profileLsubUsubClocks
                    time_LsubUsub[original_thid] += (clock64()-start_time_LsubUsub);
#endif
// #endif
// #endif
                    atomicAdd(&nz_row_U_d[src_id],1);
#ifdef enable_actual_UCount
                    atomicAdd(&actual_row_count_U[source],1);                    
                    atomicAdd(&actual_col_count_U[neighbor],1);

#endif
                   
                }
#ifdef enable_actual_UCount
                else
                {
                    atomicAdd(&actual_row_count_U[source],1);
                    atomicAdd(&actual_col_count_U[neighbor],1);               

                } 
#endif         
#endif
            }
        }
    }
    // if (original_thid==0) printf("CUDA Kernel: Line: %d: Done with the first pass\n",__LINE__);
    sync_X_block(0,gridDim.x,lock_d,1);
    while(frontier_size[0]!=0)
    {
        // my_current_frontier_d[original_thid]=INT_MAX;
        for(int_t thGroupId = warpId; thGroupId < frontier_size[0];)
        {
            int_t front=frontier[thGroupId];
            int_t source_id = src_frontier_d[thGroupId];
            int_t cost_array_offset=source_id * vert_count;
            int_t fill_array_offset = (source_id)*vert_count;// When allocating combined fill array for GPUs in a node
            int_t src=source_d[source_id];
            uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);//Cost1 for the neighbors of the representative that are less than the original front
            if (src < vert_count)
            {  
#ifdef enable_debug
#ifdef all_frontier_checked
                frontierchecked[thGroupId]=1;
#endif  
#endif
                int_t end_position;
                end_position = col_ed[front];
                for (int k=col_st[front] + laneID; k < end_position; k+=32)
                {
                    int_t m = csr[k];
#ifdef profile_TEPS
                    // time_TotalClockCycles[original_thid]++;
#endif
#ifdef profile_edge_checks
                    N_edge_checks_per_thread_d[original_thid]++;
#endif
                    if (cost_array_d[cost_array_offset+m] > cost)
                    {
                        if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                        {
                            if (m < src)
                            {
                                int_t front_position=atomicAdd(next_frontier_size,1);
                                next_frontier[front_position]=m;
                                next_src_frontier_d[front_position]=source_id;
                            }
                            if ((m + max_id_offset) > cost)
                            {
                                if (atomicMax(&fill_in_d[fill_array_offset+m],src) < src)
                                {
                                    atomicAdd(fill_count,1);
#ifdef print_fillins
                                    printf("Fill at: (%d, %d)\n",src,m);
#endif
#ifdef profile_fill_count_row
                                    atomicAdd(&fill_count_per_row_d[src],1);
#endif

#ifdef detect_supernodes
                                    if (m > src)
                                    {
 
#ifdef profileLsubUsubClocks
                                        start_time_LsubUsub=clock64();
#endif
                                        int offset= atomicAdd(&u_offset[src],1);                               
                                        usub[offset]=m;
                                        #ifdef enable_original_nz_flag
                                        is_OriginalNZ_U[offset]=1;
                                        #endif
#ifdef profileLsubUsubClocks
                                        time_LsubUsub[original_thid] += (clock64()-start_time_LsubUsub);
#endif
                                        atomicAdd(&nz_row_U_d[source_id],1);
#ifdef enable_actual_UCount
                                        atomicAdd(&actual_row_count_U[src],1);
                                        // if (src ==0) printf("Fill-in at: (0,%d)\n",m);
                                        atomicAdd(&actual_col_count_U[m],1);
#endif
                                        // atomicAdd(&nz_row_U_d[source_id*N_GPU_gSoFa_process+local_gpu_id],1);
                                    }

                                    else
                                    {

#ifdef profileLsubUsubClocks
                                        start_time_LsubUsub=clock64();
#endif
                                        int offset = atomicAdd(&l_offset[src],1);                                        
                                        lsub[offset]= m;
                                        #ifdef enable_original_nz_flag
                                        is_OriginalNZ_L[offset]=1;
                                        #endif
#ifdef profileLsubUsubClocks
                                        time_LsubUsub[original_thid] += (clock64()-start_time_LsubUsub);
#endif

#ifdef enable_actual_UCount
                                        atomicAdd(&actual_row_count_L[src],1);
                                        atomicAdd(&actual_col_count_L[m],1);

#endif
                                          
                                    }
#endif

                                }
                            }
                        }
                    }

                }

            }
            if (laneID==0) thGroupId = atomicAdd(next_front_d,1);

            thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
        }

        sync_X_block (0,gridDim.x,lock_d,1);
        // if (original_thid==0)  printf("GPU: %d is entering the M3 section!\n",local_gpu_id);
        swap_ptr_index(&frontier,&next_frontier);
        swap_ptr_index(&src_frontier_d,&next_src_frontier_d);
        if (original_thid==0)
        {    
#ifdef enable_debug
            printf("frontier size in the last loop: %d\n",frontier_size[0]);
            printf("Number of fill-ins detected till now: %d\n",fill_count[0]);
#endif
            // swap_GPU_buffers_m[0] ^=1;
            frontier_size[0]=next_frontier_size[0];
            next_frontier_size[0]=0;
            next_front_d[0] = N_warps;

        }
        level++;
        sync_X_block(0,gridDim.x,lock_d,1); 
    }

#ifdef profileLsubUsubClocks
    time_TotalClockCycles[original_thid] += (clock64()-start_time_TotalClockCycles);
#endif
}


__global__ void  warpcentric_local_detection_test  (int_t* nz_row_U_d_g0,int_t* nz_row_U_d_g1,int_t* nz_row_U_d_g2,int_t* nz_row_U_d_g3, int_t* nz_row_U_d_g4, int_t* nz_row_U_d_g5, 
        int_t N_src_group, int_t chunk_size,
        int_t N_gpu, int_t global_gpu_id, Super* superObj_d,int_t vert_count, int_t group, int_t N_chunks, int_t* Nsup_per_chunk_d,
        int_t* Nsup_d,
        int_t* fill_in_d_g0, int_t* fill_in_d_g1, int_t* fill_in_d_g2, int_t* fill_in_d_g3,int_t* fill_in_d_g4,int_t* fill_in_d_g5,int_t* source_d, 
        int_t* my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d,int_t* col_st_d,int_t local_gpu_id,
        int_t N_GPU_gSoFa_process,int_t* count,int_t* my_pass_through_d,int_t* my_chunks, int num_curr_chunks_per_node, int* my_representative,
        int_t* sup_begin,int_t* sup_end, int_t* relaxed_col_d,  int_t* relax_end_d, int_t* relaxed_supernode_d)
{
    extern __shared__ int_t ShMem[];
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t original_thid = thid;

    int_t chunk_id;//global chunk_id
    // int_t offset_warp = ShMem_offset(warp_id_block,max_supernode_size); // Max_supernode_size is allocated per warp in the shared memory. Need to change the definition. Only work for 1 block 
    int_t local_chunk_id_to_GPU = 0;
    for (int i=local_gpu_id; i<num_curr_chunks_per_node; i+=N_GPU_gSoFa_process)
        // for (int i=local_gpu_id; i<num_curr_chunks_per_node; i+=N_GPU_gSoFa_process)
    {
        if (original_thid==0) ShMem[0]=0;
        // __syncwarp();
        __syncthreads();// Only 1 thread block is used",i);
        chunk_id = my_chunks[i];
        int local_chunk_id = i;
        //Each GPU in a node processes 1 Chunk of size (max_supenode_size) to process 
        int starting_source_index_C = local_chunk_id_to_GPU * chunk_size;
        //Detect supernodes in chunk_id
        if (chunk_id < N_chunks)
        {
            //The chunk is valid. Detect supernode in the chunk
            //Let 1 GPU detect supernode in 1 chunk unlike warp centric.
            int starting_source_chunk = chunk_id * chunk_size;
            // printf("GPU: %d chunk_id:%d starting_source_chunk:%d\n",global_gpu_id, chunk_id,starting_source_chunk);
            for (int source_offset = original_thid; source_offset< chunk_size; source_offset+=(blockDim.x*gridDim.x))
            {
                int src = chunk_id * chunk_size + source_offset;//Each thread grabs a source or row in the chunk of max_supernode_size
                // if (src == 12) 
                // {
                //     printf("GPU: %d src:%d\n",global_gpu_id, src);
                // }
                if (src < vert_count)
                {
                #ifdef Enable_supernodal_relaxation
                if ((relaxed_col_d[src] != src) && (relaxed_col_d[src] != EMPTY))//non-singleton relaxed supernode
                {
                    #ifdef debug_gsofa_supernode
                    printf("original_thid:%d Continue from first condition for column: %d\n", original_thid, src);
                    #endif
                    // fflush(stdout);
                    //For multi-node continue only if the relaxed_col_d[src] is in the same node else process this src
                    // if ((relaxed_col_d[src] % max_supernode_size) == (src % max_supernode_size)) //i.e. if src and relaxed_col_d[src] are in the same chunk. i.e. same compute node
                    {
                        continue; // the supernode for src will be relaxed in below logic. So, skip this source. 
                    }
                }
                if (relax_end_d[src] > src)//non-singleton relaxed supernode
                {
                    int_t supernode_index = atomicAdd(Nsup_d,1);  
                    superObj_d[supernode_index].start = src;
                    superObj_d[supernode_index].end = relax_end_d[src];//Note: If end is in different node. Assign end as end to the last chunk in current node                
                    relaxed_supernode_d[supernode_index]++;
                    #ifdef debug_gsofa_supernode
                    
                    printf("original_thid:%d Continue from second condition for column: %d\n",original_thid, src);
                    #endif
                    
                    continue;//continue to the next source
                }
                #endif
                int j = src - starting_source_chunk;
                int_t owner_gpu; 
                int_t index_in_C; //Note: index_in_C is index of source list of concurrent sources the GPU is working on
                int r_id = Compute_r_id_resecheduled(local_chunk_id, chunk_size, j, N_GPU_gSoFa_process, &owner_gpu, &index_in_C);
                int_t* nz_row_U_d;
                //First condition checking for T3 detection
                switch(owner_gpu)
                {
                    case 0:
                        nz_row_U_d= nz_row_U_d_g0;

                        break;
                    case 1:
                        nz_row_U_d= nz_row_U_d_g1;

                        break;
                    case 2:
                        nz_row_U_d= nz_row_U_d_g2;

                        break;
                    case 3:
                        nz_row_U_d= nz_row_U_d_g3;

                        break;
                    case 4:
                        nz_row_U_d= nz_row_U_d_g4;

                        break;
                    case 5:
                        nz_row_U_d= nz_row_U_d_g5;

                        break;
                    default:
                        nz_row_U_d= nz_row_U_d_g0; 
                        // pass_through_d=pass_through_d_g0;       
                }
               
                    // if (src == 12) 
                    // {
                    //     printf("Line:%d GPU: %d src:%d\n",__LINE__, global_gpu_id, src);
                    // }
                 
                    #ifdef Enable_supernodal_relaxation
                    // if ((original_thid==0) || ((relaxed_col_d[src-1] != EMPTY)))                    
                    int flag = (src==0) ? 1: (relaxed_col_d[src-1] != EMPTY);
                    // if ((original_thid==0) || flag)
                    if ((src == starting_source_chunk) || flag)
                    #else
                    if ((original_thid==0))//remove abs later after confirming the logic//No relaxation is allowed when difference compared to 1
                    #endif
                    {
                        //Begin of new supernode for each smaller chunk at the beginning of small chunk
                        //First position allocated for a warp is the #entries in the array whose each entry is the start position of a single thread to detect all the 
                        //items in the specific supernode.
                        #ifdef debug_gsofa_supernode
                        printf("starting new supernode at src:%d\n", src);
                        #endif
                        int_t index_shared_mem = atomicAdd(&ShMem[0],1);//Atomically add the #supernodes detected into the shared memory
                        // ShMem[offset_warp+index_shared_mem+1] = pos; //pos is the source_index of the source of the beginning of the supernode
                        ShMem[index_shared_mem+1] = src; //pos is the source_index of the source of the beginning of the supernode

                    }
                    // else if ((nz_row_U_d[pos-1] - nz_row_U_d[pos]) != 1)
                    else 
                    {
                        int_t previous_src = src-1;
                        // int_t last_src_r_id= Compute_r_id_local( previous_src, N_src_group, N_GPU_gSoFa_process,g_id);
                        j = previous_src - starting_source_chunk;
                        int_t previous_src_index_in_C;

                        int_t last_src_r_id = Compute_r_id_resecheduled(local_chunk_id, chunk_size, j, N_GPU_gSoFa_process, &owner_gpu, &previous_src_index_in_C);
                        // printf("last_src_r_id: %d\n",last_src_r_id);
                        int_t* last_nz_row_U_d;
                        switch(owner_gpu)
                        {
                            case 0:
                                last_nz_row_U_d= nz_row_U_d_g0;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 1:
                                last_nz_row_U_d= nz_row_U_d_g1;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 2:
                                last_nz_row_U_d= nz_row_U_d_g2;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 3:
                                last_nz_row_U_d= nz_row_U_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 4:
                                last_nz_row_U_d= nz_row_U_d_g4;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 5:
                                last_nz_row_U_d= nz_row_U_d_g5;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            default:
                                last_nz_row_U_d= nz_row_U_d_g0;        
                        }

// #ifndef enable_supernode_relaxation
                        if ((last_nz_row_U_d[last_src_r_id] - nz_row_U_d[r_id]) != 1)
// #else
//                         if (((last_nz_row_U_d[last_src_r_id] - nz_row_U_d[r_id]) > supernode_relaxation) && ((last_nz_row_U_d[last_src_r_id] - nz_row_U_d[r_id]) < 1))
// #endif
                        {
                            //First condition not satisfied
                            int_t index_shared_mem = atomicAdd(&ShMem[0],1);//Atomically add the #supernodes detected into the shared memory
                            // ShMem[offset_warp+index_shared_mem+1] = pos; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
                            ShMem[index_shared_mem+1] = src; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
                            #ifdef debug_gsofa_supernode
                            printf("thread:%d Adding new supernode at source: %d\n",original_thid, src);
                            #endif
                            // printf("index_shared_mem:%d  offset_warp:%d   pos:%d\n",index_shared_mem,offset_warp,pos);
                        }
                        else
                        {
                            //Satisfies the first condition of non-zero count
                            #ifdef debug_gsofa_supernode
                            printf("Thread:%d Pass through detected for src: %d!\n", original_thid, src);
                            #endif
                            int diff = src-starting_source_chunk;
                            int pos_in_C = starting_source_index_C + diff;
                            my_pass_through_d[pos_in_C%N_src_group] = 1;
                            // my_pass_through_d[r_id%N_src_group] = 1;
                            // printf("Passthrough count:%d\n",atomicAdd(count,1)+1);
                        }
                    }
                }

            }
            #ifdef debug_gsofa_supernode
            printf("Finished a chunk!\n\n");
            #endif
            // for (int source = thid + chunk_id*max_supernode_size; thid <)
        }
        //Synchronize all working threads
        // sync_X_block(0,gridDim.x,lock_d,1);
        __syncthreads();// Only 1 thread block is used
        //Entering the second phase (second condition) for supernode detection
        int_t Nsup_small_chunk = ShMem[0];
        if (chunk_id < N_chunks)
        {
            for (int thid = original_thid; thid < Nsup_small_chunk;  thid+=(blockDim.x*gridDim.x))
            {
                // 1 thread assign the start of supernode to all the rows in the supernode.

                //Each thread starts a new supernode at this instant
                int_t supernode_index = atomicAdd(Nsup_d,1);                
                int_t first_row_supernode = ShMem[thid+1];
#ifdef superLU_structure
                sup_begin[supernode_index] = first_row_supernode;
                sup_end[supernode_index] = first_row_supernode;
#endif
                // #ifdef supernode_merging
                //                 my_representative[first_row_supernode] = first_row_supernode;
                // #endif 

                superObj_d[supernode_index].start = first_row_supernode;
                superObj_d[supernode_index].end = first_row_supernode;
                int_t next_row= first_row_supernode+1;
                int starting_source_chunk = chunk_id * chunk_size;
                int j = next_row - starting_source_chunk;


                int_t owner_gpu;
                int_t next_row_index_in_C;
                int_t next_row_id = Compute_r_id_resecheduled(local_chunk_id, chunk_size, j, N_GPU_gSoFa_process, &owner_gpu, &next_row_index_in_C);

                int_t* fill_in_d;
                // if (owner_gpu != 0)
                // {
                // 	printf("Owner GPU out of range: %d\n",owner_gpu);
                // }
                switch(owner_gpu)
                {
                    case 0:
                        fill_in_d= fill_in_d_g0;
                        // pass_through_d= pass_through_d_g0;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 1:
                        fill_in_d= fill_in_d_g1;
                        // pass_through_d= pass_through_d_g1;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 2:
                        fill_in_d= fill_in_d_g2;
                        // pass_through_d= pass_through_d_g2;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 3:
                        fill_in_d= fill_in_d_g3;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 4:
                        fill_in_d= fill_in_d_g4;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 5:
                        fill_in_d= fill_in_d_g5;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    default:
                        fill_in_d= fill_in_d_g0;  
                        // pass_through_d= pass_through_d_g0;    
                }
                // int_t pass_through_local_index = next_row % N_src_group;


                int diff = next_row-starting_source_chunk;
                int pos_in_C = starting_source_index_C + diff;
                int_t pass_through_local_index = pos_in_C % N_src_group;
                // int_t pass_through_local_index = next_row_index_in_C % N_src_group;


                // int_t pass_through_local_index = next_row_id % N_src_group;
                #ifdef debug_gsofa_supernode
                printf("Thread:%d Checking pass_through for next_row:%d pass_through_local_index:%d \n",original_thid,next_row,pass_through_local_index);
                #endif
                while (my_pass_through_d[pass_through_local_index] == 1)
                {
                    int_t current_row_id = next_row_id; //The GPU has its own dedicated source_d so no mapping is done to current_row_id
                    int_t current_row =   next_row; 
                    int_t offset_current_row_id = current_row_id*vert_count;
                    if (fill_in_d[offset_current_row_id + first_row_supernode] == current_row)
                    {
                        //Inside the second filter that tests if Urj is non-zero for supernode starting at r and current rwo j
                        //Extend the supernode
                        // printf("Supernode extended! \n");
                        superObj_d[supernode_index].end = current_row;  

#ifdef superLU_structure
                        // sup_begin[supernode_index] = first_row_supernode;
                        sup_end[supernode_index] = first_row_supernode;
#endif
                        // #ifdef supernode_merging
                        //                         my_representative[current_row] =  first_row_supernode;       
                        // #endif          
                        // col_ed_d[current_row] = 1 + col_st_d[current_row]+ binary_search(first_row_supernode,&csr_d[col_st_d[current_row]],0,col_ed_d[current_row]-col_st_d[current_row]-1);

                    }
                    else
                    {
                        //Introduce a new supernode
                        // printf("Supernode Added! \n");
                        supernode_index = atomicAdd(Nsup_d,1);

                        // #ifdef supernode_merging
                        //                         my_representative[current_row] =  current_row; 
                        // #endif
#ifdef superLU_structure
                        sup_begin[supernode_index] = current_row;
                        sup_end[supernode_index] = current_row;
#endif
                        superObj_d[supernode_index].start = current_row;
                        superObj_d[supernode_index].end = current_row;
                    }
                    next_row++;
                    if (next_row <vert_count)
                    {
                        j = next_row - starting_source_chunk;

                        next_row_id = Compute_r_id_resecheduled(local_chunk_id, chunk_size, j, N_GPU_gSoFa_process, &owner_gpu, &next_row_index_in_C);

                        // next_row_id= Compute_r_id_local(next_row,N_src_group,N_GPU_gSoFa_process,g_id);
                        // pass_through_local_index = next_row%N_src_group;
                        diff = next_row-starting_source_chunk;
                        pos_in_C = starting_source_index_C + diff;

                        // pass_through_local_index = next_row_index_in_C%N_src_group;
                        pass_through_local_index = pos_in_C%N_src_group;
                        // pass_through_local_index = next_row_id%N_src_group;

                        switch(owner_gpu)
                        {
                            case 0:
                                fill_in_d= fill_in_d_g0;
                                // pass_through_d= pass_through_d_g0;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 1:
                                fill_in_d= fill_in_d_g1;
                                // pass_through_d= pass_through_d_g1;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 2:
                                fill_in_d= fill_in_d_g2;
                                // pass_through_d= pass_through_d_g2;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 3:
                                fill_in_d= fill_in_d_g3;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 4:
                                fill_in_d= fill_in_d_g4;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 5:
                                fill_in_d= fill_in_d_g5;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            default:
                                fill_in_d= fill_in_d_g0;  
                                // pass_through_d= pass_through_d_g0;    
                        }
                    } 
                    else
                    {
                        break;
                    }
                }

            }
        }
        __syncthreads();
        if (original_thid ==0)
        {
            ShMem[0]=0;
            // Nsup_per_chunk_d[chunk_id] =  Nsup_small_chunk; //Writing the number for the prefix sum in the next kernel
        }
        __syncthreads();
        local_chunk_id_to_GPU++;
    }

}

void DetectSuperNodes_parallel_test( aux_device *dev_mem, int_t N_src_group,int_t chunk_size, int_t N_gpu, int_t gpu_id, 
        Super* &superObj_d,int_t vert_count, int_t group,int_t N_chunks,int_t* Nsup_per_chunk_d,
        int_t* Nsup_d,int_t* source_d,int_t* &my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d, int_t* col_st_d, 
        cudaStream_t streamObj,int_t local_gpu_id,int_t N_GPU_gSoFa_process,int_t* count,int_t* my_pass_through_d,
        int_t* my_chunks, int num_curr_chunks_per_node,int_t* my_representative,int_t* sup_begin,int_t* sup_end, int_t* relaxed_supernode_d)
{
    //***************Warp centric Supernode detection************//////////////
    //***************1 warp assignined for 1 small chunk to detect the local supernodes************//////////////
    int_t NthreadsBlock = 32;
    // int_t NthreadsBlock = 1;
    int_t N_entries_ShMem = (chunk_size+1); // +1 is for maintaining the count of supernodes per warp
    // printf("Before supernode detection kernel of GPU:%d!\n",gpu_id);
    // fflush(stdout);
    warpcentric_local_detection_test<<<1,NthreadsBlock, N_entries_ShMem * sizeof(int_t),streamObj>>> (dev_mem[0].nz_row_U_d,dev_mem[1].nz_row_U_d,dev_mem[2].nz_row_U_d,dev_mem[3].nz_row_U_d,dev_mem[4].nz_row_U_d,dev_mem[5].nz_row_U_d,
            N_src_group,chunk_size, N_gpu,
            gpu_id, superObj_d,vert_count,group,N_chunks,Nsup_per_chunk_d,
            Nsup_d,
            dev_mem[0].fill_in_d, dev_mem[1].fill_in_d, dev_mem[2].fill_in_d, dev_mem[3].fill_in_d, dev_mem[4].fill_in_d,dev_mem[5].fill_in_d,source_d,my_supernode_d,new_col_ed_d,
            col_ed_d,csr_d,col_st_d,local_gpu_id, N_GPU_gSoFa_process,count,my_pass_through_d,my_chunks, 
            num_curr_chunks_per_node,my_representative,sup_begin,sup_end, dev_mem[local_gpu_id].relaxed_col_d, dev_mem[local_gpu_id].relax_end_d, relaxed_supernode_d);
    H_ERR(cudaDeviceSynchronize()); //Added for debugging
    // printf("After supernode detection kernel of GPU:%d!\n",gpu_id);
    // fflush(stdout);
}


void Transpose(int_t n,int_t nnz, int_t* &colptr_begin, int_t* &colptr_end, int_t* &rowind)
{
    register int_t i, j, col;
    int_t *t_colptr_end,*t_colptr_begin, *t_rowind; /* a column oriented form of T = A' */
    int_t *marker;
    if ( !(marker = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for marker[]");
    if ( !(t_colptr_end = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for t_colptr_end[]");
    if ( !(t_colptr_begin = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for t_colptr_begin[]");
    if ( !(t_rowind = (int_t*) SUPERLU_MALLOC( nnz * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    // int* t_rowind = new int[nnz];
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
        for (i = colptr_begin[j]; i < colptr_end[j]; ++i)
            ++marker[rowind[i]];
    }
    t_colptr_begin[0] = 0;
    for (i = 0; i < n; ++i) {
        t_colptr_end[i] = t_colptr_begin[i] + marker[i];
        t_colptr_begin[i+1] = t_colptr_end[i]; //anil added 
        marker[i] = t_colptr_begin[i];
    }

    /* Transpose the matrix from A to T */
    for (j = 0; j < n; ++j) {
        for (i = colptr_begin[j]; i < colptr_end[j]; ++i) {
            col = rowind[i];
            t_rowind[marker[col]] = j;
            ++marker[col];
        }
    }

    swap_ptr_index(&rowind,&t_rowind);
    swap_ptr_index(&colptr_end,&t_colptr_end);
    swap_ptr_index(&colptr_begin,&t_colptr_begin);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr_end);
    SUPERLU_FREE(t_colptr_begin);
}

#if 0
/*   relax_supnode() identifies the initial relaxed supernodes, assuming that 
 *   the matrix has been reordered according to an postorder of the etree.
 * Note: gSoFa: Some of the leaf node may not have any relax_end[] value for separating the relaxed supernodes from gSoFa supernodes.
 * </pre>
 */ 
 static void relax_supnode
 /************************************************************************/
 (
  const int_t n, /* number of columns in the matrix (input) */
  int_t       *et,   /* column elimination tree (input) */
  const int_t relax, /* max no of columns allowed in a relaxed snode (input) */
  int_t       *desc, /* number of descendants of each etree node. */
  int_t       *relax_end, /* last column in a supernode (output) */
  int_t	   *representative_col, /* first column in a supernode (output) */
  int_t      *included_non_singleton /* flag if the column is already included in non-singleton supernode */
  )
  
 {
 
     register int_t j, parent, nsuper;
     register int_t fsupc; /* beginning of a snode */
 
     ifill_dist(relax_end, n, EMPTY);
     ifill_dist(representative_col, n, EMPTY);//for gSoFa
     ifill_dist(desc, n+1, 0);
     nsuper = 0;
 
     /* Compute the number of descendants of each node in the etree. */
     for (j = 0; j < n; j++) {
         parent = et[j];
         if ( parent != n )  /* not the dummy root */
             {
                 if (included_non_singleton[parent] != EMPTY) // Already included in the gSOFa non-singleton supernodes
                 {
                     desc[parent] = relax+1; // Don't relax the parent and iyts descendants
                 }
                 else
                 {
                     desc[parent] += desc[j] + 1;
                 }
             }
     }
 
     /* Identify the relaxed supernodes by postorder traversal of the etree. */
     for (j = 0; j < n; ) { 
         parent = et[j];
         fsupc = j;
         while ( parent != n && desc[parent] < relax ) {
             j = parent;
             parent = et[j];
             // representative_col[parent] = fsupc;//for gSoFa
         }
         /* Found a supernode with j being the last column. */
         relax_end[fsupc] = j; /* Last column is recorded. */		
         // representative_col[j] = fsupc;//for gSoFa
         ifill_dist(&representative_col[fsupc], j-fsupc+1, fsupc);//for gSoFa
         ++nsuper;
         ++j;
         /* Search for a new leaf. */
         while ( desc[j] != 0 && j < n ) ++j;
     }
 
 #if ( DEBUGlevel>=1 )
     printf(".. No of relaxed snodes: " IFMT "\trelax: " IFMT "\n", nsuper, relax);
 #endif
 } /* relax_supnode */
#endif
 
void compute_blocksizes(int* BLKS_NUM, int* blockSize)
  {
      cudaOccupancyMaxPotentialBlockSize( BLKS_NUM, blockSize, Compute_fillins_joint_traversal_group_wise_supernode_OptIII_warp_centric, 0, 0);
  }
  

void symbfact_min_id(int num_process,int_t max_supernode_size, int N_GPU_gSoFa_process, int vert_count, int edge_count, int myrank,ull_t* fill_count,
        int* Nsupernode_process, int_t** lsub1, int_t** usub1, int_t* col_cnt_chol,int_t* row_cnt_chol, Glu_freeable_t *Glu_freeable, Glu_persist_t* Glu_persist,
        struct Super* Supernode_per_process, int_t* NNZ_L1, int_t* NNZ_U1, int_t** usubFinal, int_t** processed_vertex1,
        int_t** is_OriginalNZ_L1,int_t** is_OriginalNZ_U1,cudaStream_t* stream,struct aux_device * dev_mem,int N_src_group,struct gSoFa_para_t* gSoFa_para, int_t mygSoFaOffset) 

{
    /*****************************************************************************************
    Note here N_GPU_gSoFa_process = N_GPU_gSoFa_process number of GPU per gSoFa process. 
    N_GPU_gSoFa_process is actually not the total number of GPUs in a node. 
    Changes made to allow multiple gSoFa process in a compute node 
    *****************************************************************************************/
   
    // printf("Process: %d entered gSoFa() vert_count:%d \n",myrank, vert_count);
    // fflush(stdout);
    double gSoFa_launch_time_start = SuperLU_timer_();

	//-------Variables for fill-in prediction. Used only if fill_in_prediction (enable_actual_UCount) is enabled
	int* actual_row_count_U;
	int* actual_col_count_U;
	int* actual_row_count_L;
	int* actual_col_count_L;
	//--------Variables for fill-in prediction. Used only if fill_in_prediction (enable_actual_UCount) is enabled
    int N_gpu = num_process*N_GPU_gSoFa_process;
    H_ERR(cudaSetDevice(mygSoFaOffset));  // offset of rank within the node.
    int_t N_chunks= gSoFa_para->N_chunks;   
    int BLKS_NUM = gSoFa_para->BLKS_NUM;
    int blockSize = gSoFa_para->blockSize;
    int_t N_groups=(ceil) (vert_count/(float)N_src_group);
    int num_nodes = num_process;   
    int total_num_chunks_per_node = gSoFa_para->total_num_chunks_per_node;
    int num_curr_chunks_per_node = gSoFa_para->num_curr_chunks_per_node;
    int_t* mysrc = gSoFa_para->mysrc;
    int* counter = gSoFa_para->counter;
    int iteration = 0;
    int_t* my_chunks=gSoFa_para->my_chunks;
    double streamcreate_time=0;	   
    int* my_supernode;//not used
    int* frontierchecked;
    int_t * new_col_ed_d;
    int_t* relaxed_supernode_d;
    uint_t group_loops=MAX_VAL/vert_count;  
    int_t temp_allocated_frontier_size; //not used //=allocated_frontier_size;
    int_t N_small_chunks;//not used =  (int) ceil (vert_count/(float)chunk_size);    
    double T3_parallel_detection =0;
    double T3_parallel_detection_temp=0;
    double initialization_overhead =0;
    int size_big_chunk = N_src_group*N_GPU_gSoFa_process;    
    int N_big_chunks = (ceil) (vert_count/(float)(size_big_chunk));
    int N_src_process = N_GPU_gSoFa_process * N_src_group;
    int N_src_loop = num_process * N_src_process;
    int total_big_chunks_per_node = (ceil)(vert_count/(float)N_src_loop);   
    int_t* my_representative;
    bool stop_computing = false;
    int_t group=0;
    int number_iterations = (ceil)((total_num_chunks_per_node)/(float)num_curr_chunks_per_node);
    int_t* lsub;int_t* usub;
    int_t* is_OriginalNZ_L; int_t* is_OriginalNZ_U;
    int_t* Lcount, *Ucount;
    int_t NNZ_L, NNZ_U;
    int_t* l_offset; int_t* u_offset;
    H_ERR(cudaMalloc((void**) &l_offset,sizeof(int_t)*(vert_count))); //Output of prefix sum
    H_ERR(cudaMalloc((void**) &u_offset,sizeof(int_t)*(vert_count))); //Every GPU share same l_offset and u_offset  
#ifdef parallel_prefix_sum
   
    int_t* l_offset_h = (int*) SUPERLU_MALLOC((sizeof(int)*vert_count));
    int_t* u_offset_h= (int*) SUPERLU_MALLOC((sizeof(int)*vert_count));
    H_ERR(cudaMalloc((void**) &Lcount,sizeof(int_t)*(vert_count)));  //Input
    H_ERR(cudaMalloc((void**) &Ucount,sizeof(int_t)*(vert_count))); 
    // prefix_sum(row_cnt_chol, l_offset_h, vert_count,Lcount,l_offset); 
    // prefix_sum(col_cnt_chol, u_offset_h, vert_count,Ucount,u_offset); 
    prefix_sum(row_cnt_chol, l_offset_h, vert_count,l_offset, Lcount); 
    prefix_sum(col_cnt_chol, u_offset_h, vert_count,u_offset, Ucount); 
    NNZ_L = l_offset_h[vert_count-1];
    NNZ_U = u_offset_h[vert_count-1];
    make_exclusive<<<128,128>>>(Ucount,vert_count,u_offset);    
    H_ERR(cudaDeviceSynchronize());
    make_exclusive<<<128,128>>>(Lcount,vert_count,l_offset);    
    H_ERR(cudaDeviceSynchronize());
    int_t* xlsub_begin = Glu_freeable->xlsub_begin;
    int_t* xlsub_end = Glu_freeable->xlsub_end;
    int_t* xusub_begin = Glu_freeable->xusub_begin;
    int_t* xusub_end = Glu_freeable->xusub_end;
    
    H_ERR(cudaMemcpy(xlsub_begin, l_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective sources from the CPU to GPU. New source scheduling
    H_ERR(cudaMemcpy(xusub_begin, u_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective sources from the CPU to GPU. New source scheduling
#else
    compute_csr_offsets(&NNZ_L, &NNZ_U, Glu_freeable, col_cnt_chol, row_cnt_chol, vert_count);
    *NNZ_L1=NNZ_L;
    *NNZ_U1 = NNZ_U;
    int_t* xlsub_begin = Glu_freeable->xlsub_begin;
    int_t* xlsub_end = Glu_freeable->xlsub_end;
    int_t* xusub_begin = Glu_freeable->xusub_begin;
    int_t* xusub_end = Glu_freeable->xusub_end;
    H_ERR(cudaMemcpy( l_offset ,xlsub_begin, vert_count*sizeof(int_t),cudaMemcpyHostToDevice));//Copying the respective sources from the CPU to GPU. New source scheduling
    H_ERR(cudaMemcpy( u_offset ,xusub_begin, vert_count*sizeof(int_t),cudaMemcpyHostToDevice));//Copying the respective sources from the CPU to GPU. New source scheduling
#endif

#ifdef initializeCSR
int_t* xlsub_original_nz_offset = Glu_freeable->xlsub_original_nz_offset;
int_t* xusub_original_nz_offset = Glu_freeable->xusub_original_nz_offset;
#endif

    int_t* processed_vertex;
    //Every GPU in a compute node share same lsub and usub
    H_ERR(cudaMallocManaged((void**) &relaxed_supernode_d,sizeof(int_t)*vert_count)); //The prefix sum is inclusive
    H_ERR(cudaMemset(relaxed_supernode_d, 0, sizeof(int_t)*vert_count));
    H_ERR(cudaMallocManaged((void**) &lsub,sizeof(int_t)*NNZ_L)); //The prefix sum is inclusive
    H_ERR(cudaMallocManaged((void**) &usub,sizeof(int_t)*NNZ_U)); //The prefix sum is inclusive
    #ifdef enable_original_nz_flag
    H_ERR(cudaMallocManaged((void**) &is_OriginalNZ_L,sizeof(int_t)*NNZ_L)); //The prefix sum is inclusive
    H_ERR(cudaMemset(is_OriginalNZ_L, 0, sizeof(int_t)*NNZ_L));
    H_ERR(cudaMallocManaged((void**) &is_OriginalNZ_U,sizeof(int_t)*NNZ_U)); //The prefix sum is inclusive
    H_ERR(cudaMemset(is_OriginalNZ_U, 0, sizeof(int_t)*NNZ_U));
    *is_OriginalNZ_L1 = is_OriginalNZ_L;
    *is_OriginalNZ_U1 = is_OriginalNZ_U; 
    #endif
    *lsub1 = lsub;
    *usub1 = usub; 

    *Nsupernode_process =0;
    double gSoFa_launch_time = SuperLU_timer_() - gSoFa_launch_time_start;

    printf("IAM:%d gSoFa_launch_time: %f ms\n",myrank, gSoFa_launch_time*1000); 
    fflush(stdout);
    // start_time=SuperLU_timer_();
    // cout<<"IAM: "<<myrank<<" Running the merge kernel"<<endl;
    // fflush(stdout);   

    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
#ifdef lambda
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process + offset)); 
#else
        // H_ERR(cudaSetDevice(i)); // offset of rank within the node.
        H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
        Initialise_cost_array<<<128,128,0,stream[i]>>>( dev_mem[i].cost_array_d,vert_count,dev_mem[i].group_MAX_VAL,N_src_group);
        #ifdef initializeCSR
        printf("IAM: %d initialiseCSR\n",myrank);
        fflush(stdout);
        Initialise_lsub_usub<<<128,128,0,stream[i]>>>(lsub,usub,vert_count,u_offset, l_offset,dev_mem[i].col_st_d, dev_mem[i].col_ed_d, dev_mem[i].csr_d);        
        #endif
        H_ERR(cudaDeviceSynchronize());
    } 
    #ifdef initializeCSR
    H_ERR(cudaMemcpy(Glu_freeable->xlsub_original_nz_offset, l_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective offsets
    H_ERR(cudaMemcpy(Glu_freeable->xusub_original_nz_offset, u_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective offsets
    #endif
    printf("IAM: %d copied initialiseCSR\n",myrank);
        fflush(stdout);
    // printf("IAM:%d At Line:%d\n",myrank, __LINE__);
    // fflush(stdout);
    for (int local_chunk_count=0; local_chunk_count < (total_num_chunks_per_node ); local_chunk_count+=num_curr_chunks_per_node)//To ensure all the chunks are processsed

    {	       
        int starting_chunk = iteration * num_nodes * num_curr_chunks_per_node + myrank;
        my_chunks[0] = starting_chunk;
        for (int j=1; j<num_curr_chunks_per_node; j++)
        {
            my_chunks[j] = my_chunks[j-1] + num_nodes;

        }
        memset(counter, 0, N_GPU_gSoFa_process*sizeof(int));
        int ptr = 0;
        int source_begin;
        for (int iter=0; iter < num_curr_chunks_per_node; iter++)
        {
            //Assigning 1 chunk in 1 iter
            source_begin = my_chunks[iter] * gSoFa_para->chunk_size;
            if (my_chunks[0] >= vert_count)
                // if (source_begin >= vert_count) 
            {
                stop_computing = true;
                break;
            }
            for (int j = 0; j < gSoFa_para->chunk_size; j++)
            {
                int source = source_begin+j;
                int local_gpu_id = ptr % N_GPU_gSoFa_process;
                mysrc[local_gpu_id*N_src_group+counter[local_gpu_id]] = source;

#ifndef create_supernodal_LU
                //For debugging when Nsup is set to 1
                Supernode_per_process[*Nsupernode_process].start =  source;
                Supernode_per_process[*Nsupernode_process].end =  source;
                (*Nsupernode_process)++;
#endif
                counter[local_gpu_id]++;
                ptr ++;
            }
        } 
        if (stop_computing)
        {
            break;
        }
        for (int i=0;i<N_GPU_gSoFa_process;i++)
        // for (int i=1;i<N_GPU_gSoFa_process;i++)//debugging for multi-GPU Supernode relaxation
        {
#ifdef lambda
            int offset = GPUOffset;
            H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process+offset)); 
#else
            // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
            H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
            int myGlobalrank = myrank * N_GPU_gSoFa_process + i;
            H_ERR(cudaMemcpyAsync(dev_mem[i].source_d, &mysrc[i*N_src_group] , N_src_group*sizeof(int_t),cudaMemcpyHostToDevice,stream[i]));//Copying the respective sources from the CPU to GPU. New source scheduling
            int last_heighest_source=0;
            int valid_index =0;
            for (int k=0;k<N_src_group;k++)
            {
                if (last_heighest_source <  mysrc[i*N_src_group+k])
                {
                    last_heighest_source=mysrc[i*N_src_group+k];
                    if (mysrc[i*N_src_group+k] < vert_count)
                    {
                        valid_index = k+1;
                        // printf("GPU: %d will work on source:%d\n",i,mysrc[i*N_src_group+k]);
                    }
                }
                // if (last_heighest_source > mysrc[i*N_src_group+k])
                // {
                //     printf("Found decreasing source at GPU: %d\n",i);
                //     printf("valid index of GPU:%d is %d\n",i,valid_index);
                // }
 
            }
            Compute_fillins_joint_traversal_group_wise_supernode_OptIII_warp_centric<<<BLKS_NUM,blockSize,0, stream[i]>>>  (dev_mem[i].cost_array_d,dev_mem[i].fill_in_d,dev_mem[i].frontier_d,
                    dev_mem[i].next_frontier_d,vert_count,dev_mem[i].csr_d,dev_mem[i].col_st_d,dev_mem[i].col_ed_d,dev_mem[i].fill_count_d,myGlobalrank,N_gpu,
                    dev_mem[i].src_frontier_d,dev_mem[i].next_src_frontier_d,dev_mem[i].source_d, dev_mem[i].frontier_size_d, dev_mem[i].next_frontier_size_d,
                    dev_mem[i].lock_d,N_groups,
                    dev_mem[i].dump_m,dev_mem[i].load_m,
                    N_src_group,group,dev_mem[i].max_id_offset,dev_mem[i].next_front_d,temp_allocated_frontier_size,
                    dev_mem[i].my_current_frontier_d,frontierchecked,dev_mem[i].swap_GPU_buffers_m,
                    dev_mem[i].nz_row_U_d, max_supernode_size,  N_small_chunks,dev_mem[i].source_flag_d, dev_mem[i].my_supernode_d, dev_mem[i].superObj_d,dev_mem[i].validity_supernode_d,
                    dev_mem[i].pass_through_d,dev_mem[i].fill_count_per_row_d, dev_mem[i].N_edge_checks_per_thread_d,new_col_ed_d,N_GPU_gSoFa_process,i,dev_mem[i].time_LsubUsub, 
                    dev_mem[i].time_TotalClockCycles, dev_mem[i].fill_sup_merge,lsub, usub, 
                    l_offset,u_offset,actual_row_count_U,actual_col_count_U,actual_row_count_L,actual_col_count_L,valid_index,processed_vertex, is_OriginalNZ_L, is_OriginalNZ_U);      
        }
        SynchronizeAllDevices(stream,N_GPU_gSoFa_process);
        for (int i=0;i<N_GPU_gSoFa_process;i++)
        {          
            dev_mem[i].max_id_offset-=vert_count;
            dev_mem[i].count_group_loop++;
#ifdef disable_maxID_optimization
            if (true)
#else
                if (dev_mem[i].count_group_loop >= group_loops)
#endif
                {
                    dev_mem[i].max_id_offset=MAX_VAL-vert_count;           
                    dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;

#ifdef lambda
                    int offset = GPUOffset;
                    H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process + offset)); 
#else
                    // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
                    H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
                    Initialise_cost_array<<<128,128,0,stream[i]>>>(dev_mem[i].cost_array_d,vert_count,dev_mem[i].group_MAX_VAL,N_src_group);
                    dev_mem[i].count_group_loop=0;
                }
                else
                {
                    dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;
                }
        }
        SynchronizeAllDevices(stream,N_GPU_gSoFa_process);   
#ifdef debug_fill_count
        ull_t fill_count_group = 0;
        for (int i=0;i<N_GPU_gSoFa_process;i++)
        {
            ull_t indiv_gpu_fillcount =0 ;
#ifdef lambda
            int offset = GPUOffset;
            H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process+offset)); 
#else
            // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
            H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
            H_ERR(cudaMemcpy(&indiv_gpu_fillcount,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
            fill_count_group += indiv_gpu_fillcount;
        }

#endif
          //******Check the supernodal expansion***********//
        //   printf("At Line:%d\n",__LINE__);
        //   fflush(stdout);
#ifdef detect_supernodes
        T3_parallel_detection_temp = SuperLU_timer_();
        for (int i=0;i<N_GPU_gSoFa_process;i++)
        {
#ifdef lambda
            int offset = GPUOffset;
            H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process+offset)); 
#else
            // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
            H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
            int myGlobalrank = myrank*N_GPU_gSoFa_process + i;
            // printf("At Line:%d for GPU:%d \n",__LINE__, i);
            // fflush(stdout);
#ifdef create_supernodal_LU

            DetectSuperNodes_parallel_test (dev_mem,N_src_group, gSoFa_para->chunk_size, N_gpu,
                    myGlobalrank, dev_mem[i].superObj_d, vert_count,group, N_chunks, dev_mem[i].Nsup_per_chunk_d,dev_mem[i].Nsup_d,
                    dev_mem[i].source_d, my_supernode, new_col_ed_d,
                    dev_mem[i].col_ed_d, dev_mem[i].csr_d, dev_mem[i].col_st_d,stream[i],i, N_GPU_gSoFa_process,dev_mem[i].count, 
                    dev_mem[i].pass_through_d,my_chunks,  num_curr_chunks_per_node,my_representative,dev_mem[i].sup_begin,dev_mem[i].sup_end, relaxed_supernode_d);
#endif

        }
        // printf("IAM:%d At Line:%d\n",myrank, __LINE__);
        // fflush(stdout);
        SynchronizeAllDevices(stream,N_GPU_gSoFa_process);
        // printf("At Line:%d \n",__LINE__);
        // fflush(stdout);
        T3_parallel_detection += (SuperLU_timer_()-T3_parallel_detection_temp);
#endif
        iteration++;
    }
    SynchronizeAllDevices(stream,N_GPU_gSoFa_process);
    H_ERR(cudaMemcpy(xlsub_end, l_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective sources from the CPU to GPU. New source scheduling
    H_ERR(cudaMemcpy(xusub_end, u_offset , vert_count*sizeof(int_t),cudaMemcpyDeviceToHost));//Copying the respective sources from the CPU to GPU. New source scheduling
    // printf("IAM:%d At Line:%d\n",myrank, __LINE__);
    // fflush(stdout);
    double sorting_start_time = SuperLU_timer_();
#ifdef SortLsubUsubIndicies 
    for (int iter=0; iter<vert_count; iter++)
    {
        // sort(&lsub[xlsub_begin[iter]],&lsub[xlsub_end[iter]]);
        sort(&usub[xusub_begin[iter]],&usub[xusub_end[iter]]);
    }
#endif 
    double sort_time =  SuperLU_timer_()- sorting_start_time;
    // printf("Sorting time: %lf ms\n",sort_time*1000);
    // fflush(stdout);

#ifdef create_supernodal_LU
    // This section generates the lsub/xlsub, usub/xusub by removing the bubbles (no removal of bubbles)
    // + removes the  L and U structure of following nodes in L and U
    // Confusion: T3 type supernode only ensures (rows in U in my case to be matched). Can we still use supernode info to merge row in L?
    // Super* Supernode_per_process = (Super *)malloc(vert_count);
    *Nsupernode_process =0;
    int_t* included_non_singleton = intMalloc_dist(vert_count);
    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
#ifdef lambda
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(i+myrank*N_GPU_gSoFa_process+offset)); 
#else
        // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
        H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
        //Writing the supernode into supernode data structure per process
        
        // H_ERR(cudaMemcpy(dev_mem[i].superObj,dev_mem[i].superObj_d,sizeof(Super)* vert_count,cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(dev_mem[i].superObj,dev_mem[i].superObj_d,sizeof(Super)* gSoFa_para->sup_per_gpu,cudaMemcpyDeviceToHost));
        int_t GpuSupCnt=0;
        H_ERR(cudaMemcpy(&GpuSupCnt,dev_mem[i].Nsup_d,sizeof(int_t),cudaMemcpyDeviceToHost));
        // printf("In GPU: %d Finished copying the GpuSupCnt=%d device to host!\n",i,GpuSupCnt);
        // fflush(stdout); 
        H_ERR(cudaMemcpy( &Supernode_per_process[Nsupernode_process[0]],dev_mem[i].superObj_d,sizeof(Super)* GpuSupCnt,cudaMemcpyDeviceToHost));
        (*Nsupernode_process) = (*Nsupernode_process) + GpuSupCnt;
        
        printf("In GPU: %d Finished copying the GpuSupCnt=%d device to host!\n",i,GpuSupCnt);
        fflush(stdout); 
#if 0
        #if 0//not running for now
        // printf("At Line: %d !\n",__LINE__);
        // fflush(stdout);         
        int_t* processed;
		if ( !(processed = (int_t*) SUPERLU_MALLOC( vert_count * sizeof(int_t)) ) )
		ABORT("SUPERLU_MALLOC fails for processed[]");		
		ifill_dist(processed, vert_count, EMPTY);//processed stores the representative column of the supernode.
        // printf("At Line: %d !\n",__LINE__);
        // fflush(stdout); 
        
        ifill_dist(included_non_singleton, vert_count, EMPTY);
        for (int j=0; j < GpuSupCnt; j++)
		{ 
            int start = dev_mem[i].superObj[j].start;
            int end = dev_mem[i].superObj[j].end;
            if (start != end)
            {
                // printf("Adding gSoFa non-singleton supernode [%d, %d]!\n",start,end);
                // fflush(stdout);
                Supernode_per_process[Nsupernode_process[0]].start = start;
			Supernode_per_process[Nsupernode_process[0]].end = end;//end is not used
            Nsupernode_process[0]++;
            int_t representative = start;
            while (start <= end)
            {
                included_non_singleton[start] = representative;
                start++;
            }
            }            
        }
        #if 0
        for (int j=0; j < GpuSupCnt; j++)
		{
            // printf("At Line: %d !\n",__LINE__);
            // fflush(stdout); 
			int start = dev_mem[i].superObj[j].start;
            // printf("At Line: %d !\n",__LINE__);
            // fflush(stdout); 
			int end = dev_mem[i].superObj[j].end;
            printf("Working on supernode [%d, %d]!\n",start,end);
            fflush(stdout);
			int representative=-1; 
			printf("Begin of relaxation of the supernodes!\n");
            fflush(stdout); 
			// if ((processed[start]!= EMPTY) && (processed[end] == EMPTY))
			if ((processed[start]== EMPTY) && (processed[end] == EMPTY))
			{
                printf("At Line: %d !\n",__LINE__);
                fflush(stdout);         
				printf("[%d, %d ] supernode is not yet processed!\n",start,end);
				//Both the start and end of the supernode are not covered by any other supernode. So, we can create a new supernode
				int last_col = gSoFa_para->relax_end[start];
				// if ((last_col != EMPTY) && (last_col != start) && (last_col > end))
				if ((last_col != EMPTY) && (last_col > end))
				{
					printf("[%d, %d ] supernode can be expanded to [%d, %d] if in same process (chunk_id)!\n",start,end, start, last_col);
					//expand the supernode
					if (last_col/max_supernode_size == end/max_supernode_size)
					{
						//only expand if end and last_col in the same chunk_id (size of max_supernode_size)
						//don't relax if the last_col and end belong to different chunk_id
						printf("Relaxing the supernode from [%d, %d] to [%d, %d]!\n",start,end,start,last_col);
						end = last_col;
					}										
				}

			}	

			//update the processed array			
			// int skip = 0;
			if (start == end)
			{
				//skip if singleton supernode (start == end) and start column and representative column from relaxation array is different 
				if ((gSoFa_para->representative_col[start] != start) && (gSoFa_para->representative_col[start] != EMPTY))
				{
					// skip = 1;
					printf("Skipped the supernode [%d, %d] to supernode data structure because of better relaxation option!\n",start,end);
                    fflush(stdout);
					continue;
				}				
			}
			printf("Adding the supernode [%d, %d] to supernode data structure!\n",start,end);
			Supernode_per_process[Nsupernode_process[0]].start = start;
			Supernode_per_process[Nsupernode_process[0]].end = end;//end is not used
			representative = start; 
			while (start <=end)
			{
				processed[start] = representative;
				start++;
			}
			(*Nsupernode_process)++;
		}	
        #endif
    #else
        int selected_col_cnt=0;
        int * column_selected =  intMalloc_dist(vert_count);
        memset(column_selected, 0, vert_count*sizeof(vert_count));
        for (int j=0; j < GpuSupCnt; j++)
        {
            Supernode_per_process[Nsupernode_process[0]].start = dev_mem[i].superObj[j].start;
            Supernode_per_process[Nsupernode_process[0]].end = dev_mem[i].superObj[j].end;//end is not used;
        
            int start = dev_mem[i].superObj[j].start;
            int stop = dev_mem[i].superObj[j].end;
            #ifdef DEBUG_relaxation
            if ((start == 49) || (start == 50))
            {
                printf ("Supernode [%d, %d] is added!\n",start,stop);
                fflush(stdout);
            }
            if (relaxed_supernode_d[Nsupernode_process[0]] !=0)
            {
                printf("Relaxed supernode [%d, %d]. Supernode Index:%d Added %d times!\n",start, stop,*Nsupernode_process, relaxed_supernode_d[Nsupernode_process[0]]);
                fflush(stdout);
            }
            else
            {
                // if ((stop <= 50) && ((stop-start) >= 1))
                // if (stop <= 50)
                // if (stop <= 590)
                if (stop <= 800)
                {
                    printf("Exact supernode. [%d, %d] Supernode Index: %d\n",start,stop,*Nsupernode_process);
                    fflush(stdout);
                }
            }
            #endif
            while (start <= stop)
            {
                column_selected[start]++;// = 1;
                selected_col_cnt++;
                start++;
            }
            // printf("Finalizing the supernode [%d, %d]!\t",Supernode_per_process[Nsupernode_process[0]].start,Supernode_per_process[Nsupernode_process[0]].end);

            fflush(stdout);
            (*Nsupernode_process)++;
        }
        printf ("Number of processed columns:%d in the supernodes Unprocessed supernodes:%d \n", selected_col_cnt, vert_count-selected_col_cnt);
        fflush(stdout);
        for (int i=0; i<vert_count; i++)
        {
            if (column_selected[i]==0)
            {
                printf("Column: %d not included in any supernode!\n",i);
            }
            else if  (column_selected[i] > 1)
            {
                printf("Column: %d included for:%d times\n ",i,column_selected[i]);
            }
        }
    #endif
    #endif
    }
    // printf("IAM:%d At Line:%d\n",myrank, __LINE__);
    // fflush(stdout);
    #if 0    
    double time_relax_supnode_start = SuperLU_timer_();
        int relax = sp_ienv_dist(2);
         printf("Relaxation parameter: %d\n",relax);
         int* desc_temp;
         int* relax_end = intMalloc_dist(vert_count);
         int* representative_col = intMalloc_dist(vert_count);

         if ( !(desc_temp = intMalloc_dist(vert_count+1)) )
             ABORT("Malloc fails for desc_temp[]");
         relax_supnode(vert_count, gSoFa_para->etree, relax, desc_temp, relax_end,representative_col,included_non_singleton);
         gSoFa_para->relax_end = relax_end;
         gSoFa_para->representative_col = representative_col;
         printf("Relax_end: ");
         for (int i=0;i<vert_count;i++)
         {
             printf("%d ",gSoFa_para->relax_end[i]);
         } 

             printf("\n representative_col: ");
         for (int i=0;i<vert_count;i++)
         {
             printf("%d ",gSoFa_para->representative_col[i]);
         } 
         SUPERLU_FREE(desc_temp);
     //~Computing relaxed supernodes
     double time_relax_supnode = SuperLU_timer_() - time_relax_supnode_start;
    //  printf("Time for computing relaxed supernodes: %f ms\n", time_relax_supnode*1000);
    // fflush(stdout);
     for (int j=0; j < vert_count;) // Note: Compute mysource[] again when doing multi-process version instead of all the sources.
     {
        //Relax the remaining singleton supernodes when possible 
        if (included_non_singleton[j] == EMPTY)
        {
            //Singleton supernode from gSoFa
            //checking the relaxation of the suppernode
            // printf("At Line %d, \n", __LINE__);
            // fflush(stdout);
            if (gSoFa_para->relax_end[j] > j)
            {
                // printf("Relaxing the singleton supernode from [%d, %d] to [%d, %d]!\n",j,j,j,gSoFa_para->relax_end[j]);
                // fflush(stdout);
                //Relax the singleton supernode
                Supernode_per_process[Nsupernode_process[0]].start = j;
                Supernode_per_process[Nsupernode_process[0]].end = gSoFa_para->relax_end[j];
                (*Nsupernode_process)++;
                j = gSoFa_para->relax_end[j]+1;
                continue;
            }
            else
            {
                // printf("Adding the singleton supernode [%d, %d] to supernode data structure!\n",j,j);
                // fflush(stdout);
                Supernode_per_process[Nsupernode_process[0]].start = j;
                Supernode_per_process[Nsupernode_process[0]].end = j;
                (*Nsupernode_process)++;
                j++;
                continue;
            }
        }
        j++;
     }
     #endif

    

#else

#endif
    // *time +=((SuperLU_timer_()-start_time)*1000);     

    double time_post_supernode_selection_start = SuperLU_timer_();
#ifdef debugMemcpy
    int_t pass_through_detected=0;
    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
#ifdef lambda
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(offset+ i+myrank*N_GPU_gSoFa_process)); 
#else
        // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
        H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
        ull_t fill_count_temp =0 ;
        int_t temp_supernode = 0;
        int_t temp_pass_through_count=0;
        H_ERR(cudaMemcpy(&fill_count_temp,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(&temp_supernode,dev_mem[i].Nsup_d,sizeof(int_t),cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(&temp_pass_through_count,dev_mem[i].count,sizeof(int_t),cudaMemcpyDeviceToHost));
        pass_through_detected += temp_pass_through_count;      
        *fill_count += fill_count_temp;
    }
#endif

    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
        ull_t fill_count_temp =0 ;        
#ifdef lambda
        gpu = i+myrank*N_GPU_gSoFa_process;
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(gpu+offset)); 

#else
        // gpu = i;
        // H_ERR(cudaSetDevice(gpu)); 
        H_ERR(cudaSetDevice(i + mygSoFaOffset));

#endif
        H_ERR(cudaMemcpy(&fill_count_temp,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
        *fill_count += fill_count_temp;
    }
    // printf("Inside cuda code at Line:%d\n",__LINE__);
    // fflush(stdout);
#ifdef enable_actual_UCount
    int sum_actual =0;
    int num_mismatches =0;
    int num_mismatches_L=0;
    for (int i=0;i<vert_count;i++)
    {
        // assert(col_cnt_chol[i]==actual_row_count_U[i]);
// #ifndef switchLU_write
//         if (row_cnt_chol[i] < actual_row_count_U[i])
// #else
            if (col_cnt_chol[i] < actual_row_count_U[i])
// #endif
            {                
                num_mismatches++;
                // printf("Allocation could have invalid writes!\n");
            }            
// #ifndef switchLU_write
//         if (row_cnt_chol[i] < actual_col_count_L[i])
// #else
            if (col_cnt_chol[i] < actual_col_count_L[i])
// #endif
            {
                num_mismatches_L++;
            }
    }
    printf("Actual_Ucount Total: %d\n",sum_actual);
    printf("Number of mismatchesU_row: %d\t #vertex:%d\n",num_mismatches,vert_count);
    printf("Number of mismatches L_col: %d\t #vertex:%d\n",num_mismatches_L,vert_count);

    fflush(stdout);
#endif
#ifdef debuggSoFa
    for (int_t i=0; i<vert_count; i++)
    {
        *N_processed_source += processed_vertex[i];
    }
    cout<<"Process: "<<myrank<<" Number of fill-ins detected: "<<*fill_count<<endl;
    cout<<"Process: "<<myrank<<" Number of source processed: "<<*N_processed_source<<endl;
#endif
    // cout<<"Process: "<<myrank<<" FINAL Number of non-zeros: "<<(*fill_count + edge_count)<<endl;
    // cout<<"Process: "<<myrank<<" PREDICTED Number of non-zeros: "<<(sum+sum_row)-vert_count<<endl;
    // cout<<"Process: "<<myrank<<" time for fill-in detection: "<<time[0] <<endl;

    // fflush(stdout);
    cout<<"Process: "<<myrank<<" Time for only stream creation: "<< streamcreate_time * 1000 <<" ms"<<endl;
    cout<<"Process: "<<myrank<<" Time for only supernodes detection: "<<T3_parallel_detection*1000<<" ms"<<endl;
    cout<<"Process: "<<myrank<<" initialization_overhead: "<<initialization_overhead*1000 <<" ms"<< endl;
#ifdef profileLsubUsubClocks
    ull_t TotalSymbfactCycles=0;
    ull_t TotallsubUsubCycles=0;
    ull_t* SymbfactCycles= (ull_t*) malloc (BLKS_NUM*blockSize*sizeof(ull_t));
    ull_t* lsubUsubCycles= (ull_t*) malloc (BLKS_NUM*blockSize*sizeof(ull_t));
    memset(SymbfactCycles, 0, BLKS_NUM*blockSize*sizeof(ull_t));
    memset(lsubUsubCycles, 0, BLKS_NUM*blockSize*sizeof(ull_t));

    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
#ifdef lambda
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(offset+ i+myrank*N_GPU_gSoFa_process)); 
#else
        // printf("setting up GPU: %d\n",i);
        // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
        H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif

        H_ERR(cudaMemcpy(SymbfactCycles,dev_mem[i].time_TotalClockCycles,sizeof(ull_t)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(lsubUsubCycles,dev_mem[i].time_LsubUsub,sizeof(ull_t)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));

        for (int i=0; i<BLKS_NUM*blockSize;i++ )
        {
            TotalSymbfactCycles += SymbfactCycles[i];
            TotallsubUsubCycles += lsubUsubCycles[i];
        }

    }

    printf("Process:%d Writing in lsub/usub is %f percent of the symbolic factorization time\n",myrank, (TotallsubUsubCycles/(float)TotalSymbfactCycles)*100);
#endif

#ifdef destroy_streams
    for (int i=0;i<N_GPU_gSoFa_process;i++)
    {
#ifdef lambda
        int offset = GPUOffset;
        H_ERR(cudaSetDevice(offset+ i+myrank*N_GPU_gSoFa_process)); 
#else
        // H_ERR(cudaSetDevice(i));  // offset of rank within the node.
        H_ERR(cudaSetDevice(i + mygSoFaOffset));
#endif
        H_ERR( cudaStreamDestroy(stream[i]));	
    }
#endif
    //~destroy the created cuda streams
    cout<<"Exiting gSoFa!"<<endl;
    fflush(stdout);
 double time_post_supernode_selection = SuperLU_timer_() - time_post_supernode_selection_start;
    cout<<"Process: "<<myrank<<" Time for post supernode selection: "<<time_post_supernode_selection*1000<<" ms"<<endl;
    // PrintInt10("xlsub_begin",Glu_freeable->xlsub_begin, 10);
    // PrintInt10("xlsub_begin",10, Glu_freeable->xlsub_begin);
    // fflush(stdout);       
    //  PrintInt10("xlsub_end", 10, Glu_freeable->xlsub_end);
    //  fflush(stdout);
    //  PrintInt10("xlsub_original_nz_offset",10, Glu_freeable->xlsub_original_nz_offset); 
    //  fflush(stdout);
    return;
}
#endif

