#include "acc_aux.h"

#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

// int
// get_thread_per_process ()
// {
//     char *ttemp;
//     ttemp = getenv ("THREAD_PER_PROCESS");

//     if (ttemp)
//         return atoi (ttemp);
//     else
//         return 1;
// }


static inline double
load_imb (double *A, int nthreads)
{
    int i;
    double _mx, _avg;
    _mx = 0;
    _avg = 0;
    for (i = 0; i < nthreads; i++)
    {
        _mx = (((_mx) > (A[i])) ? (_mx) : (A[i]));
        _avg += A[i];
    }
    _avg = _avg / (double) nthreads;
    return _mx - _avg;
}

// #define ACC_ASYNC_COST 3.79e-3

#define  MAX_DIM 12800
#define  MAX_IN_DIM 256
#define  LOG_2_MAX_IN_DIM 8
#define  LOG_2_MAX_DIM 7


double get_acc_async_cost()
{
    char *ttemp;
    ttemp = getenv ("ACC_ASYNC_COST");
    if (ttemp)
        return (double) atof (ttemp);
    else
        return 4e-3;
}

// #define  CPU_BANDWIDTH 35.0

double cpu_bandwidth;
int communication_overlap;
double acc_async_cost;


int_t fixed_partition;
double frac;

/* Sherry: these lookup tables are not needed on Titan, nor Summit */
double CpuDgemmLookUp[8][8][9];
double PhiDgemmLookUp[8][8][9];
double PhiBWLookUp[8];       // no used?
double MicPciBandwidth[18];  // no used?
double MicScatterBW[24][24];

#ifdef OFFLOAD_PROFILE
double MicScatterTLI[MAX_BLOCK_SIZE / STEPPING][MAX_BLOCK_SIZE / STEPPING];
double host_scu_flop_rate[CBLOCK / CSTEPPING][CBLOCK / CSTEPPING][CBLOCK / CSTEPPING];
#endif

static inline unsigned int next_power_2(unsigned int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


static inline unsigned int previous_power_2(unsigned int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v / 2;
}


#include <stdint.h>
// static inline uint32_t my_log2(const uint32_t x) {
//     uint32_t y;
//     asm ( "\tbsr %1, %0\n"
//           : "=r"(y)
//           : "r" (x)
//         );
//     return y;
// }

static inline uint32_t my_log2(const uint32_t x)
{
    return (uint32_t) log2((double) x);
}

static inline unsigned int nearst_2_100(unsigned int v)
{
    v = (v + 99) / 100;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return SUPERLU_MIN(my_log2(v), LOG_2_MAX_DIM) ;
}

static inline unsigned int nearst_k(unsigned int v)
{

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return SUPERLU_MIN(my_log2(v), LOG_2_MAX_IN_DIM) ;
}



double estimate_acc_time(int m, int n , int k)
{
    double flop_rate = PhiDgemmLookUp[nearst_2_100(m)][nearst_2_100(m)][nearst_k(k)];
    double gemm_time = 2e-9 * (double) m * (double)n * (double)k / (flop_rate);

    double mop_rate = PhiBWLookUp[nearst_2_100( sqrt((double) m * (double)n))];

    double scatter_time = 24e-9 * (double) m * (double)n / mop_rate ;
    // printf("gemm_time %.2e scatter_time %.2e, flop_rate %lf mop_rate %lf ",gemm_time, scatter_time, flop_rate,mop_rate);
    if (gemm_time < 0)
    {
        /* code */
        printf(" m %d n %d k %d \n", m, n, k);
        exit(0);
    }

    double eta = 1;       /*to allow more computations on ACC only applicable for MPI cases*/
    // if(m>1024 && k>32) eta=1.5;
    if (communication_overlap)
    {
        if (m > 2048 && k > 32) eta = 5.0;
        if (m > 4096 && k > 32) eta = 6.0;
        if (m > 4096 && k > 64) eta = 8.0;
    }


    return (gemm_time + scatter_time) / eta;
}



double estimate_acc_gemm_time(int m, int n , int k)
{
    double flop_rate = PhiDgemmLookUp[nearst_2_100(m)][nearst_2_100(m)][nearst_k(k)];
    double gemm_time = 2e-9 * (double) m * (double)n * (double)k / (flop_rate);


    double eta = 1;       /*to allow more computations on ACC only applicable for MPI cases*/
    // if(m>1024 && k>32) eta=1.5;
    if (communication_overlap)
    {
        if (m > 2048 && k > 32) eta = 5.0;
        if (m > 4096 && k > 32) eta = 6.0;
        if (m > 4096 && k > 64) eta = 8.0;
    }


    return (gemm_time) / eta;
}


double estimate_acc_scatter_time(int m, int n , int k)
{

    double mop_rate = PhiBWLookUp[nearst_2_100( sqrt((double) m * (double)n))];

    double scatter_time = 24e-9 * (double) m * (double)n / mop_rate ;

    double eta = 1;       /*to allow more computations on ACC only applicable for MPI cases*/
    // if(m>1024 && k>32) eta=1.5;
    if (communication_overlap)
    {
        if (m > 2048 && k > 32) eta = 5.0;
        if (m > 4096 && k > 32) eta = 6.0;
        if (m > 4096 && k > 64) eta = 8.0;
    }


    return (scatter_time) / eta;
}

double estimate_cpu_time(int m, int n , int k)
{
    if (m == 0 || n == 0 || k == 0)
    {
        return 0;
    }
    double flop_rate = CpuDgemmLookUp[nearst_2_100(m)][nearst_2_100(m)][nearst_k(k)];
    double gemm_time = 2e-9 * (double) m * (double)n * (double)k / (flop_rate);
    double scatter_time = 24e-9 * (double) m * (double)n / cpu_bandwidth ;
    return gemm_time + scatter_time;
}


double acc_data_send_time(size_t sz)
{
    if (my_log2((sz + 999) / 1000) > 17 ) return 1e-9 * (double) sz / MicPciBandwidth[17];
    return 1e-9 * (double) sz / MicPciBandwidth[my_log2((sz + 999) / 1000)];
}


void LookUpTableInit(int my_rank)
{
    char *ttemp;
    char buffer[1024];
    char *line;
    FILE *fp;

    ttemp = getenv("CPU_BANDWIDTH");
    if (ttemp)
    {
        cpu_bandwidth = atof(ttemp);
#ifdef GPU_DEBUG
        if (!my_rank) printf("Bandwidth of CP is %lf \n", cpu_bandwidth );
#endif
    }
    else
    {
        printf("Please set CPU_BANDWIDTH : bbye\n");
        exit(0);

    }

    // ttemp = getenv("SLU_ACC_COMM_OVERLAP");
    // if (ttemp)
    // {
    //     communication_overlap = atoi(ttemp);
    //     if (!my_rank && communication_overlap ) printf("Using communication computation overlap version\n");
    // }
    // else
    // {
    //     printf("Please set SLU_ACC_COMM_OVERLAP : bbye\n");
    //     exit(0);
    // }


    /*Reading CPU performance table */
    ttemp = getenv("CPU_DGEMM_PERF_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");
        double max_flop_rate = 0;
        if (!fp)
        {
            if (!my_rank) printf("can not open %s: exiting  \n", ttemp);
            exit(0);
        }

        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)
        {

            int m, n, k;
            double flop_rate;
            sscanf(line, "%d, %d, %d, %lf ", &m, &n, &k, &flop_rate);
            CpuDgemmLookUp[nearst_2_100(m)][nearst_2_100(m)][nearst_k(k)] = flop_rate;
            max_flop_rate = SUPERLU_MAX(flop_rate, max_flop_rate);
        }
        fclose(fp);
        // printf("CPU: MAX FLOP Rate %lf GFLOP/Sec\n",max_flop_rate );
    }
    else
    {
        printf("Assign CPU performance table \n");
        exit(0);
    }

    ttemp = getenv("ACC_DGEMM_PERF_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");
        if (!fp)
        {
            printf("can not open %s: exiting  \n", ttemp);
            exit(0);
        }
        double max_flop_rate = 0;
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)
        {

            int m, n, k;
            double flop_rate;
            sscanf(line, "%d, %d, %d, %lf ", &m, &n, &k, &flop_rate);
            PhiDgemmLookUp[nearst_2_100(m)][nearst_2_100(m)][nearst_k(k)] = flop_rate;
            max_flop_rate = SUPERLU_MAX(flop_rate, max_flop_rate);
        }
        fclose(fp);
#ifdef GPU_DEBUG
        if (!my_rank) printf("ACC: MAX FLOP Rate %lf GFLOP/Sec\n", max_flop_rate );
#endif
    }
    else
    {
        printf("Assign ACC DGEMM performance table \n");
        exit(0);
    }

    ttemp = getenv("ACC_SCATTER_PERF_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");
        double max_mop_rate = 0;
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)
        {

            int m;
            double mop_rate, sd;
            sscanf(line, "%d, %lf, %lf", &m, &mop_rate, &sd);
            PhiBWLookUp[nearst_2_100(m)] = mop_rate;
            max_mop_rate = SUPERLU_MAX(mop_rate, max_mop_rate);
        }
        fclose(fp);
#ifdef GPU_DEBUG
        if (!my_rank) printf("ACC: MAX MOP Rate %lf GFLOP/Sec\n", max_mop_rate );
#endif
    }
    else
    {
        printf("Assign ACC DGEMM performance table \n");
        exit(0);
    }


    ttemp = getenv("ACC_PCI_BW_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");

        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)
        {

            int m;
            double bw;
            sscanf(line, "%d,%lf", &m, &bw);
            MicPciBandwidth[my_log2(m / 1000)] = bw;

        }
        fclose(fp);
    }
    else
    {
        printf("Assign ACC_PCI_BW_TABLE \n");
        exit(0);
    }

    ttemp = getenv("ACC_SCATTER_BW_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");


        for (int i = 0; i < 24; ++i)
        {
            for (int j = 0; j < 24; ++j)
            {
                fscanf(fp, "%lf", &MicScatterBW[i][j]);
                // printf("%d  %d %lf\n",i,j,MicScatterBW[i][j] );
            }
        }


        fclose(fp);
    }
    else
    {
        printf("Assign ACC_SCATTER_BW_TABLE \n");
        exit(0);
    }

#ifdef OFFLOAD_PROFILE
    ttemp = getenv("ACC_SCATTER_TLI_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");
        double max_mop_rate = 0;

        for (int i = 0; i < MAX_BLOCK_SIZE / STEPPING; ++i)
        {
            for (int j = 0; j < MAX_BLOCK_SIZE / STEPPING; ++j)
            {

                fscanf(fp, "%lf", &MicScatterTLI[i][j]);
                if (MicScatterTLI[i][j] > 2)
                {
                    MicScatterTLI[i][j] = 2;
                }
                // printf("%lf \n", MicScatterTLI[i][j]);
            }
        }


        fclose(fp);
    }
    else
    {
        printf("ACC_SCATTER_TLI_TABLE \n");
        exit(0);
    }

    ttemp = getenv("HOST_SCU_PERF_TABLE");
    if (ttemp)
    {
        fp = fopen(ttemp, "r");
        for (int_t k = 0; k < CBLOCK / CSTEPPING; ++k)
        {

            for (int_t i = 0; i < CBLOCK / CSTEPPING; ++i)
            {
                for (int_t j = 0; j < CBLOCK / CSTEPPING; ++j)
                {
                    fscanf(fp, "%lf", &host_scu_flop_rate[k][i][j]);

                }

            }
        }
        fclose(fp);
    }
    else
    {
        printf("please assign HOST_SCU_PERF_TABLE \n");
        exit(0);
    }

#endif

    ttemp = getenv("FIXED_PARTITION");
    if (ttemp)
    {
        fixed_partition = atoi(ttemp);
        if (fixed_partition)
        {
            printf("Using fixed workload partition \n");
            ttemp = getenv("CPU_ACC_WORK_PARTITION");
            if (ttemp)
            {
                frac  = atof (ttemp);
            }
            else
            {
                frac = 1;
            }

        }

    }
    else
    {
        fixed_partition = 0;
    }

} /* end LookupTableInit */


double l_count[24];    /*used for keeping entries*/
double u_count[24];     /*for keeping u entries*/

double
estimate_acc_scatter_time_strat1(Ublock_info_t* Ublock_info, int_t nub, Remain_info_t* Lblock_info, int_t nlb )
{
    for (int i = 0; i < 24; ++i)
    {
        l_count[i] = 0;
        u_count[i] = 0;
    }

    int_t cum_nrows = 0;
    int_t cum_ncols = 0;
    for (int i = 0; i < nub; ++i)
    {
        int_t ncols = Ublock_info[i].ncols;
        int_t ind = SUPERLU_MAX(CEILING(ncols, 8) - 1, 0);
        u_count[ind] += (double) ncols;
        cum_ncols += ncols;

    }


    for (int i = 0; i < nlb; ++i)
    {
        int_t nrows = Lblock_info[i].nrows;
        int_t ind = SUPERLU_MAX(CEILING(nrows, 8) - 1, 0);
        l_count[ind] += (double) nrows;

        cum_nrows += nrows;

    }

    double ttime = 0;
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            /* code */
            ttime += 8 * 3e-9 * l_count[i] * u_count[j] / MicScatterBW[i][j];
        }
    }

    // ttime *= (MicScatterTLI[CLAMP( CEILING(cum_nrows, STEPPING) ,0 , MAX_BLOCK_SIZE/STEPPING -1 )][CLAMP( CEILING(cum_ncols, STEPPING) ,0 , MAX_BLOCK_SIZE/STEPPING -1 )]) ;
    ttime *= SUPERLU_MIN(nub * nlb / 240 , 1);
    return ttime;
}

#ifdef OFFLOAD_PROFILE
/*following is a good strategy; gives good prediction but for some reason I do not get over all performance
improvement so I've ommited this thing out*/
double
estimate_cpu_sc_time_strat1(int_t ldu, Ublock_info_t* Ublock_info, int_t nub, Remain_info_t* Lblock_info, int_t nlb )
{
    int_t ind_k = SUPERLU_MAX(CEILING(ldu, 8) - 1, 0);
    for (int i = 0; i < 24; ++i)
    {
        l_count[i] = 0;
        u_count[i] = 0;
    }

    int_t cum_nrows = 0;
    int_t cum_ncols = 0;
    for (int i = 0; i < nub; ++i)
    {
        int_t ncols = Ublock_info[i].ncols;
        int_t ind = SUPERLU_MAX(CEILING(ncols, 8) - 1, 0);
        u_count[ind] += (double) ncols;
        cum_ncols += ncols;

    }


    for (int i = 0; i < nlb; ++i)
    {
        int_t nrows = Lblock_info[i].nrows;
        int_t ind = SUPERLU_MAX(CEILING(nrows, 8) - 1, 0);
        l_count[ind] += (double) nrows;
        cum_nrows += nrows;
    }

    double ttime = 0;
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            /* flop rate is in gf/sec */
            ttime += 2e-9 * ldu * l_count[i] * u_count[j] / host_scu_flop_rate[ind_k][i][j];
        }
    }

    return ttime;
}

#endif

/* Sherry: this routine is not called */
int_t fixed_cpu_acc_partition (Ublock_info_t *Ublock_info_Phi, int_t num_u_blks_Phi , int_t Rnbrow, int_t ldu_Phi)
{
    int_t acc_cols, cpu_cols;
    int_t total_cols = Ublock_info_Phi[num_u_blks_Phi - 1].full_u_cols;
    if (frac == 0)
    {
        return num_u_blks_Phi;
    }
    else if (frac == 1)
    {
        return 0;
    }

    for (int_t j = num_u_blks_Phi - 1; j > -1; --j)      // ###
    {

        acc_cols = (j == 0) ? 0 : Ublock_info_Phi[j - 1].full_u_cols ;
        cpu_cols = total_cols - acc_cols;

        if (estimate_acc_time (Rnbrow, acc_cols, ldu_Phi) < acc_async_cost)
        {
            break;
        }
        if (cpu_cols > frac * total_cols )
        {
            return j;
        }

    }

    return 0;
}


/* Partition the "num_u_blks_Phi" portion into GPU and CPU part,
   based on the estimated computational cost on CPU and GPU.
   This was useful for the old Intel Phi architecture, but for the
   new architecture, such as Titan and Summit, we can give everything
   to GPU.
*/
int_t tuned_partition(int_t num_u_blks_Phi, Ublock_info_t *Ublock_info_Phi, Remain_info_t* Remain_info, int_t RemainBlk, double cpu_time_0, int_t Rnbrow, int_t ldu_Phi )
{
    double cpu_time, acc_time;
    int_t acc_cols, cpu_cols;

    for (int_t j = num_u_blks_Phi - 1; j > -1; --j)      // ###
    {

        acc_cols = (j == 0) ? 0 : Ublock_info_Phi[j - 1].full_u_cols ;
        cpu_cols = Ublock_info_Phi[num_u_blks_Phi - 1].full_u_cols - acc_cols;
        acc_time = estimate_acc_scatter_time_strat1(&Ublock_info_Phi[0], j,
                   Remain_info,  RemainBlk ) + estimate_acc_gemm_time(Rnbrow, acc_cols, ldu_Phi);

        cpu_time = estimate_cpu_time(Rnbrow, cpu_cols, ldu_Phi) + cpu_time_0;


        // SCT.Predicted_host_sch_time[k0] = cpu_time_without_offload;
        if (cpu_time > acc_time + acc_async_cost)
        {
            return j;

        }
    }

    return 0; /*default value is zero */
}


