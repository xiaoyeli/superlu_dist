#pragma once

// #include "pdgstrf.h"


typedef struct mdwin_t
{
	double cpu_bandwidth;
	int communication_overlap;
	double acc_async_cost;


	int_t fixed_partition;
	double frac;

	double CpuDgemmLookUp[8][8][9];
	double PhiDgemmLookUp[8][8][9];
	double PhiBWLookUp[8];
	double MicPciBandwidth[18];
	double MicScatterBW[24][24];

#ifdef OFFLOAD_PROFILE
	double MicScatterTLI[MAX_BLOCK_SIZE / STEPPING][MAX_BLOCK_SIZE / STEPPING];
	double host_scu_flop_rate[CBLOCK / CSTEPPING][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING];
#endif
} mdwin_t;

int_t
get_max_buffer_size ();

double get_acc_async_cost();

double estimate_acc_time(int m, int n , int k);

double estimate_acc_gemm_time(int m, int n , int k);

double estimate_acc_scatter_time(int m, int n , int k);

double estimate_cpu_time(int m, int n , int k);

double acc_data_send_time(size_t sz);

void LookUpTableInit(int my_rank);


int_t fixed_cpu_acc_partition (Ublock_info_t *Ublock_info_Phi, int_t num_u_blks_Phi , int_t Rnbrow, int_t ldu_Phi);
int_t tuned_partition(int_t num_u_blks_Phi, Ublock_info_t *Ublock_info_Phi, Remain_info_t* Remain_info,
                      int_t RemainBlk, double cpu_time_0, int_t Rnbrow, int_t ldu_Phi );