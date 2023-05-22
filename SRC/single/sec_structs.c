/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Auxiliary routines in 3D algorithms
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * </pre>
 */


//#include <mm_malloc.h>
#include "superlu_ddefs.h"
#if 0
#include "sec_structs.h"
#include <stdlib.h> /*for Qsort */
#include <mpi.h>
#include <math.h>   /*for sqrt*/
#include <string.h>
#include "compiler.h"
//#include "load-balance/supernodal_etree.h"
#include "supernodal_etree.h"
#endif

#include <stdio.h> /*for printfs*/
#include <stdlib.h> /*for getline*/

// double CPU_CLOCK_RATE;
/*for sorting structures */
int Cmpfunc_R_info (const void * a, const void * b)
{
    return ( ((Remain_info_t*)a)->nrows - ((Remain_info_t*)b)->nrows );
}

  

int Cmpfunc_U_info (const void * a, const void * b)
{
    return ( ((Ublock_info_t*)a)->ncols - ((Ublock_info_t*)b)->ncols );
}


int sort_R_info( Remain_info_t* Remain_info, int n )
{
    qsort((void *) Remain_info , n , sizeof(Remain_info_t), Cmpfunc_R_info);

    return 0;
} 

int sort_U_info( Ublock_info_t* Ublock_info, int n )
{
    qsort((void *) Ublock_info , n , sizeof(Ublock_info_t), Cmpfunc_U_info);

    return 0;
}

int Cmpfunc_R_info_elm (const void * a, const void * b)
{
    return ( ((Remain_info_t*)a)->eo - ((Remain_info_t*)b)->eo );
}



int Cmpfunc_U_info_elm (const void * a, const void * b)
{
    return ( ((Ublock_info_t*)a)->eo - ((Ublock_info_t*)b)->eo );
}



int sort_R_info_elm( Remain_info_t* Remain_info, int n )
{
    /*sorts on the basis of order of elimination*/
    qsort((void *) Remain_info , n , sizeof(Remain_info_t), Cmpfunc_R_info_elm);

    return 0;
}

int sort_U_info_elm( Ublock_info_t* Ublock_info, int n )
{
    qsort((void *) Ublock_info , n , sizeof(Ublock_info_t), Cmpfunc_U_info_elm);

    return 0;
}

double *SCT_ThreadVarInit(int_t num_threads)
{
#if 0
    double *var = (double *) _mm_malloc(num_threads * CACHE_LINE_SIZE * sizeof(double), 64);
#else
    double *var = (double *) doubleMalloc_dist(num_threads * CACHE_LINE_SIZE);
#endif
    for (int_t i = 0; i < num_threads * CACHE_LINE_SIZE; ++i)
    {
        var[i] = 0.0;
    }
    return var;
}

#if 0
#define DEFAULT_CPU_FREQ 3000.0   // 3 GHz

double getFreq(void)
{
    FILE *fp = fopen("/proc/cpuinfo", "rb");
    if(!fp) {
	// the file /proc/cpuinfo doesn't exists, return 3000 Mhz as the frequency
#if ( PRNTlevel>=2 )
	printf("/proc/cpuinfo doesn't exists, using 3GHz as CPU frequency. Some timers will not be correct\n");
#endif
	return DEFAULT_CPU_FREQ;
    }

    char *arg = 0;
#if 1
    size_t len = 0;
    char *line = NULL;
#else
    size_t len = 100;   // Sherry fix
    char *line = SUPERLU_MALLOC(len * sizeof(char));
#endif
    size_t read;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        // printf("%s", line);
        char * pch;
        pch = strtok (line, " \t:");
        if (pch != NULL && strcmp(pch, "cpu") == 0)
        {

            /* code */
            pch = strtok (NULL, " \t:");
            // printf("%s\n", pch );
            if (pch != NULL && strcmp(pch, "MHz") == 0)
            {
                pch = strtok (NULL, " \t:");
                double freq = atof(pch);
                free(arg);
                fclose(fp);
                return freq;

                break;
            }
        }
	free(line);
	line = NULL;
    }

    //SUPERLU_FREE(line); // sherry added
    return 0;
}

#endif

/* Initialize various counters. */
void SCT_init(SCT_t* SCT)
{
#if 1
    // CPU_CLOCK_RATE = getFreq() * 1e-3;
#else
    CPU_CLOCK_RATE = 3000. * 1e-3;
#endif
    int num_threads = 1;

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
        #pragma omp master
        {
            num_threads = omp_get_num_threads ();
        }
    }
#endif

    SCT->acc_load_imbal = 0.0;

    /* Counter for couting memory operations */
    SCT->scatter_mem_op_counter   = 0.0;
    SCT->scatter_mem_op_timer     = 0.0;
#ifdef SCATTER_PROFILE
    SCT->Host_TheadScatterMOP = (double *)_mm_malloc(sizeof(double) * (num_threads * (192 / 8) * (192 / 8)), 64);
    SCT->Host_TheadScatterTimer = (double *)_mm_malloc(sizeof(double) * (num_threads * (192 / 8) * (192 / 8)), 64);
    memset(SCT->Host_TheadScatterMOP, 0, sizeof(double) * (num_threads * (192 / 8) * (192 / 8)));
    memset(SCT->Host_TheadScatterTimer, 0, sizeof(double) * (num_threads * (192 / 8) * (192 / 8)));
#endif

    SCT->LookAheadRowSepTimer = 0.0;
    SCT->LookAheadRowSepMOP   = 0.0;
    SCT->GatherTimer          = 0.0;
    SCT->GatherMOP            = 0.0;
    SCT->LookAheadGEMMTimer   = 0.0;
    SCT->LookAheadGEMMFlOp    = 0.0;
    SCT->LookAheadScatterTimer   = 0.0;
    SCT->LookAheadScatterMOP   = 0.0;
    SCT->AssemblyTimer        = 0.0;

    SCT->offloadable_flops = 0.0;
    SCT->offloadable_mops = 0.0;

#if 0
    SCT->SchurCompUdtThreadTime = (double *) _mm_malloc(num_threads * CACHE_LINE_SIZE * sizeof(double), 64);
#else
    SCT->SchurCompUdtThreadTime = (double *) doubleMalloc_dist(num_threads * CACHE_LINE_SIZE);
#endif

    for (int_t i = 0; i < num_threads * CACHE_LINE_SIZE; ++i)
    {
        SCT->SchurCompUdtThreadTime[i] = 0.0;
    }

    SCT->schur_flop_counter = 0.0;
    SCT->schur_flop_timer = 0.0;

    SCT->datatransfer_timer = 0;
    SCT->schurPhiCallTimer = 0;
    SCT->schurPhiCallCount = 0;
    SCT->datatransfer_count = 0;
    SCT->PhiWaitTimer = 0;
    SCT->PhiWaitTimer_2 = 0;
    SCT->NetSchurUpTimer = 0;
    SCT->PhiMemCpyTimer = 0;
    SCT->PhiMemCpyCounter = 0;

    SCT->pdgstrs2_timer = 0.0;
    SCT->trf2_flops = 0;
    SCT->trf2_time = 0;
    SCT->CPUOffloadTimer = 0;
    SCT->pdgstrf2_timer = 0.0;
    SCT->lookaheadupdatetimer = 0;

    /* diagonal block factorization; part of pdgstrf2*/
    // SCT->Local_Dgstrf2_tl = 0;
    SCT->Local_Dgstrf2_Thread_tl = SCT_ThreadVarInit(num_threads);
    /*Wait for U diagnal bloc kto receive; part of pdgstrf2 */
    SCT->Wait_UDiagBlock_Recv_tl = 0;
    /*wait for receiving L diagonal block: part of mpf*/
    SCT->Wait_LDiagBlock_Recv_tl = 0;
    SCT->Recv_UDiagBlock_tl = 0;
    /*wait for previous U block send to finish; part of pdgstrf2 */
    SCT->Wait_UDiagBlockSend_tl = 0;
    /*after obtaining U block, time spent in calculating L panel;part of pdgstrf2*/
    SCT->L_PanelUpdate_tl = 0;
    /*Synchronous Broadcasting U panel*/
    SCT->Bcast_UPanel_tl = 0;
    SCT->Bcast_LPanel_tl = 0;
    /*Wait for L send to finish */
    SCT->Wait_LSend_tl = 0;

    /*Wait for U send to finish */
    SCT->Wait_USend_tl = 0;
    /*Wait for U receive */
    SCT->Wait_URecv_tl = 0;
    /*Wait for L receive */
    SCT->Wait_LRecv_tl = 0;

    /*U_panelupdate*/
    SCT->PDGSTRS2_tl = 0;

    /*profiling by phases*/
    SCT->Phase_Factor_tl = 0;
    SCT->Phase_LU_Update_tl = 0;
    SCT->Phase_SC_Update_tl = 0;

    /*time to get the lock*/
    SCT->GetAijLock_Thread_tl = SCT_ThreadVarInit(num_threads);

    /*3d timers*/
    SCT->ancsReduce = 0.0;
    SCT->gatherLUtimer = 0.0;

    for (int i = 0; i < MAX_3D_LEVEL; ++i)
    {
        /* code */
        SCT->tFactor3D[i] = 0;
        SCT->tSchCompUdt3d[i] = 0;
    }

    SCT->tAsyncPipeTail = 0.0; 
    SCT->tStartup =0.0;

    SCT->commVolFactor =0.0;
    SCT->commVolRed =0.0;
} /* SCT_init */

void SCT_free(SCT_t* SCT)
{
#ifdef SCATTER_PROFILE
    free(SCT->Host_TheadScatterMOP);
    free(SCT->Host_TheadScatterTimer);
#endif
#if 0
    _mm_free(SCT->SchurCompUdtThreadTime);
    _mm_free(SCT->Local_Dgstrf2_Thread_tl);
    _mm_free(SCT->GetAijLock_Thread_tl);
#else
    SUPERLU_FREE(SCT->SchurCompUdtThreadTime);
    SUPERLU_FREE(SCT->Local_Dgstrf2_Thread_tl);
    SUPERLU_FREE(SCT->GetAijLock_Thread_tl);
#endif
    SUPERLU_FREE(SCT); // sherry added
}


void DistPrint(char* function_name,  double value, char* Units, gridinfo_t* grid)
/*
Prints average of the value across all the MPI ranks;
Displays as function_name  \t value \t units;
*/
{
    int iam = grid->iam;
    int num_procs = grid->nprow * grid->npcol;
    double sum;
    double min = 0;
    double max = 0;
    double value_squared = value * value;
    double sum_value_squared;

    MPI_Reduce( &value, &sum,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    MPI_Reduce( &value, &min,  1, MPI_DOUBLE, MPI_MIN, 0, grid->comm );
    MPI_Reduce( &value, &max,  1, MPI_DOUBLE, MPI_MAX, 0, grid->comm );
    MPI_Reduce( &value_squared, &sum_value_squared,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    double std_dev = sqrt((sum_value_squared - (sum * sum / num_procs) ) / num_procs);
    if (!iam)
    {
        printf("|%s \t| %10.4f \t| %10.4f \t| %10.4f \t| %10.4f%%| %s|\n", function_name,
               sum / num_procs, min, max, 100 * num_procs * std_dev / sum, Units );
        // printf("%s \t %lf %s\n", function_name, value, Units );
    }

}

void DistPrint3D(char* function_name,  double value, char* Units, gridinfo3d_t* grid3d)
/*
Prints average of the value across all the MPI ranks;
Displays as function_name  \t value \t units;
*/
{
    int iam = grid3d->iam;
    int num_procs = grid3d->nprow * grid3d->npcol * grid3d->npdep;
    double sum;
    double min = 0;
    double max = 0;
    double value_squared = value * value;
    double sum_value_squared;

    MPI_Reduce( &value, &sum,  1, MPI_DOUBLE, MPI_SUM, 0, grid3d->comm );
    MPI_Reduce( &value, &min,  1, MPI_DOUBLE, MPI_MIN, 0, grid3d->comm );
    MPI_Reduce( &value, &max,  1, MPI_DOUBLE, MPI_MAX, 0, grid3d->comm );
    MPI_Reduce( &value_squared, &sum_value_squared,  1, MPI_DOUBLE, MPI_SUM, 0, grid3d->comm );
    double std_dev = sqrt((sum_value_squared - (sum * sum / num_procs) ) / num_procs);
    if (!iam)
    {
        printf("|%s \t| %10.4f \t| %10.4f \t| %10.4f \t| %10.4f%%| %s|\n", function_name,
               sum / num_procs, min, max, 100 * num_procs * std_dev / sum, Units );
        // printf("%s \t %lf %s\n", function_name, value, Units );
    }

}

void DistPrintMarkupHeader(char* headerTitle,  double value,  gridinfo_t* grid)
{

    int iam = grid->iam;
    int num_procs = grid->nprow * grid->npcol;
    double sum;
    double min = 0;
    double max = 0;
    double value_squared = value * value;
    double sum_value_squared;

    MPI_Reduce( &value, &sum,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    MPI_Reduce( &value, &min,  1, MPI_DOUBLE, MPI_MIN, 0, grid->comm );
    MPI_Reduce( &value, &max,  1, MPI_DOUBLE, MPI_MAX, 0, grid->comm );
    MPI_Reduce( &value_squared, &sum_value_squared,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );

    if (!iam)
    {
        printf("#### %s : %10.4f \n\n", headerTitle,sum / num_procs );
        printf("|Function name \t| avg \t| min \t| max \t| std-dev| units|\n");
        printf("|---|---|---|---|---|---|\n");
        // printf("%s \t %lf %s\n", function_name, value, Units );
    }

}
void DistPrintThreaded(char* function_name, double* value, double Norm, int_t num_threads, char* Units, gridinfo_t* grid)
/*
Prints average of the value across all the MPI ranks, for threaded variables;
First averages over all the threads;
Norm is normalizing constant
Displays as function_name  \t value \t units;
*/
{
    int iam = grid->iam;
    int num_procs = grid->nprow * grid->npcol;
    double local_sum = 0;
    for (int i = 0; i < num_threads ; ++i)
    {
        local_sum += value[i * CACHE_LINE_SIZE];
    }

    local_sum = local_sum / (Norm * num_threads);
    double sum;
    double min = 0;
    double max = 0;
    double value_squared = local_sum * local_sum;
    double sum_value_squared;

    MPI_Reduce( &local_sum, &sum,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    MPI_Reduce( &local_sum, &min,  1, MPI_DOUBLE, MPI_MIN, 0, grid->comm );
    MPI_Reduce( &local_sum, &max,  1, MPI_DOUBLE, MPI_MAX, 0, grid->comm );
    MPI_Reduce( &value_squared, &sum_value_squared,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    double std_dev = sqrt((sum_value_squared - (sum * sum / num_procs) ) / num_procs);
    if (!iam)
    {
        printf("|%s \t| %10.4f \t| %10.4f \t| %10.4f \t| %10.4f%% %s|\n", function_name,
               sum / num_procs, min, max, 100 * num_procs * std_dev / sum, Units );
        // printf("%s \t %lf %s\n", function_name, value, Units );
    }
}


/*for mkl_get_blocks_frequency*/
// #include "mkl.h"
void SCT_print(gridinfo_t *grid, SCT_t* SCT)
{
    int num_threads = 1;

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
        #pragma omp master
        {
            num_threads = omp_get_num_threads ();
        }
    }
#endif
    // CPU_CLOCK_RATE = 1e9 * CPU_CLOCK_RATE;

    int iam = grid->iam;
    int_t num_procs = grid->npcol * grid->nprow;
    double temp_holder;
    MPI_Reduce( &SCT->NetSchurUpTimer, &temp_holder,  1, MPI_DOUBLE, MPI_SUM, 0, grid->comm );
    if (!iam)
    {
        // printf("CPU_CLOCK_RATE  %.1f\n", CPU_CLOCK_RATE );
        printf("Total time in factorization \t: %5.2lf\n", SCT->pdgstrfTimer);
        printf("MPI-communication phase \t: %5.2lf\n", SCT->pdgstrfTimer - (temp_holder / num_procs));

    }

    /* Printing Panel factorization profile*/
    // double CPU_CLOCK_RATE = 1e9 * mkl_get_clocks_frequency();


    // DistPrint("Local_Dgstrf2", SCT->Local_Dgstrf2_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrintThreaded(
    //     "Local_Dgstrf2         ", SCT->Local_Dgstrf2_Thread_tl, CPU_CLOCK_RATE, num_threads,
    //     "Seconds", grid);

    // DistPrint("Wait_UDiagBlock_Recv  ", SCT->Wait_UDiagBlock_Recv_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Wait_LDiagBlock_Recv  ", SCT->Wait_LDiagBlock_Recv_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Recv_UDiagBlock       ", SCT->Recv_UDiagBlock_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Wait_UDiagBlockSend   ", SCT->Wait_UDiagBlockSend_tl / CPU_CLOCK_RATE, "Seconds", grid);

    // DistPrint("Bcast_UPanel          ", SCT->Bcast_UPanel_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Bcast_LPanel          ", SCT->Bcast_LPanel_tl / CPU_CLOCK_RATE, "Seconds", grid);
    DistPrint("Wait_LSend            ", SCT->Wait_LSend_tl , "Seconds", grid);
    DistPrint("Wait_USend            ", SCT->Wait_USend_tl , "Seconds", grid);
    DistPrint("Wait_URecv            ", SCT->Wait_URecv_tl , "Seconds", grid);
    DistPrint("Wait_LRecv            ", SCT->Wait_LRecv_tl , "Seconds", grid);
    DistPrint("L_PanelUpdate         ", SCT->L_PanelUpdate_tl , "Seconds", grid);
    DistPrint("PDGSTRS2              ", SCT->PDGSTRS2_tl , "Seconds", grid);
    
    DistPrint("wait-FunCallStream    ", SCT->PhiWaitTimer , "Seconds", grid);
    DistPrint("wait-copyStream       ", SCT->PhiWaitTimer_2 , "Seconds", grid);
    DistPrint("waitGPU2CPU           ", SCT->PhiWaitTimer , "Seconds", grid);
    DistPrint("SchurCompUpdate       ", SCT->NetSchurUpTimer, "Seconds", grid);
    DistPrint("PanelFactorization    ", SCT->pdgstrfTimer - SCT->NetSchurUpTimer, "Seconds", grid);
    
    // DistPrint("Phase_Factor          ", SCT->Phase_Factor_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Phase_LU_Update       ", SCT->Phase_LU_Update_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrint("Phase_SC_Update       ", SCT->Phase_SC_Update_tl / CPU_CLOCK_RATE, "Seconds", grid);
    // DistPrintThreaded(
    //     "GetAijLock         ", SCT->GetAijLock_Thread_tl, CPU_CLOCK_RATE, num_threads,
    //     "Seconds", grid);
    double t_total = SCT->tStartup + SCT->pdgstrfTimer + SCT->gatherLUtimer; 
    DistPrintMarkupHeader("High Level Time Breakdown", t_total, grid);
    DistPrint("Startup               ", SCT->tStartup, "Seconds", grid);
    DistPrint("Main-Factor loop      ", SCT->pdgstrfTimer, "Seconds", grid);
    DistPrint("3D-GatherLU           ", SCT->gatherLUtimer, "Seconds", grid);
    DistPrint("tTotal                ", t_total, "Seconds", grid);

    DistPrintMarkupHeader("Components of Factor Loop",SCT->pdgstrfTimer, grid);
    DistPrint("3D-AncestorReduce     ", SCT->ancsReduce, "Seconds", grid);
    DistPrint("Pipeline Tail         ", SCT->tAsyncPipeTail, "Seconds", grid);

}

void SCT_print3D(gridinfo3d_t *grid3d, SCT_t* SCT)
{

    gridinfo_t* grid = &(grid3d->grid2d);
    
    char funName[100];

    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    for (int i = maxLvl-1; i >-1; --i)
    {
        /* code */
        sprintf( funName, "Grid-%d Factor:Level-%d    ", grid3d->zscp.Iam,
		 (int) maxLvl-1-i);
        DistPrint(funName, SCT->tFactor3D[i], "Seconds", grid);
        // sprintf( funName, "SchurCU:Level-%d   ",  maxLvl-1-i);
        // DistPrint(funName, SCT->tSchCompUdt3d[i], "Seconds", grid);
        // sprintf( funName, "PanelFact:Level-%d ",  maxLvl-1-i);
        // DistPrint(funName, SCT->tFactor3D[i]-SCT->tSchCompUdt3d[i], "Seconds", grid);
    }

}


void treeImbalance3D(gridinfo3d_t *grid3d, SCT_t* SCT)
{

    gridinfo_t* grid = &(grid3d->grid2d);
    char funName[100];

    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    for (int i = maxLvl-1; i >-1; --i)
    {
        /* code */
        double tsum;
        MPI_Reduce( &SCT->tFactor3D[i], &tsum,  1, MPI_DOUBLE, MPI_SUM, 0, grid3d->zscp.comm );

        double tmax;
        MPI_Reduce( &SCT->tFactor3D[i], &tmax,  1, MPI_DOUBLE, MPI_MAX, 0, grid3d->zscp.comm );
        
        double tavg = tsum /(grid3d->zscp.Np>>i);
        double lLmb =  100*(tmax-tavg)/tavg;
        sprintf( funName, "Imbalance Factor:Level-%d    ",  (int) maxLvl-1-i);
        if(!grid3d->zscp.Iam)
        DistPrint(funName, lLmb, "Seconds", grid);
        // sprintf( funName, "SchurCU:Level-%d   ",  maxLvl-1-i);
        // DistPrint(funName, SCT->tSchCompUdt3d[i], "Seconds", grid);
        // sprintf( funName, "PanelFact:Level-%d ",  maxLvl-1-i);
        // DistPrint(funName, SCT->tFactor3D[i]-SCT->tSchCompUdt3d[i], "Seconds", grid);
    }

}


void SCT_printComm3D(gridinfo3d_t *grid3d, SCT_t* SCT)
{
    //
    double cvolFactor;
    MPI_Reduce( &SCT->commVolFactor, &cvolFactor,  1, MPI_DOUBLE, MPI_SUM, 0, grid3d->comm );
    double cvolRed;
    MPI_Reduce( &SCT->commVolRed, &cvolRed,  1, MPI_DOUBLE, MPI_SUM, 0, grid3d->comm );

    int_t Np = (grid3d->npcol) * (grid3d->nprow) * (grid3d->npdep);
    if (!grid3d->iam)
    {
        /* code */
        printf("| commVolRed | %g   | %g |\n", cvolRed, cvolRed/Np );
        printf("| commVolFactor | %g   | %g |\n", cvolFactor, cvolFactor/Np );
    }

}

int
get_acc_offload ()
{
    char *ttemp;
    ttemp = getenv ("SUPERLU_ACC_OFFLOAD");

    if (ttemp)
        return atoi (ttemp);
    else
        return 1;  // default
}


void Free_HyP(HyP_t* HyP)
{
#if 0
    _mm_free(HyP->lookAhead_info );
    _mm_free(HyP->Remain_info );
    _mm_free(HyP->lookAhead_L_buff );
    _mm_free(HyP->Remain_L_buff );
    _mm_free(HyP->Ublock_info );
    _mm_free(HyP->Ublock_info_Phi );
    _mm_free(HyP->Lblock_dirty_bit );
    _mm_free(HyP->Ublock_dirty_bit );
#else
    SUPERLU_FREE(HyP->lookAhead_info );
    SUPERLU_FREE(HyP->Remain_info );
    SUPERLU_FREE(HyP->lookAhead_L_buff );
    SUPERLU_FREE(HyP->Remain_L_buff );
    SUPERLU_FREE(HyP->Ublock_info );
    SUPERLU_FREE(HyP->Ublock_info_Phi );
    SUPERLU_FREE(HyP->Lblock_dirty_bit );
    SUPERLU_FREE(HyP->Ublock_dirty_bit );
#endif
    SUPERLU_FREE(HyP);
}

int updateDirtyBit(int_t k0, HyP_t* HyP, gridinfo_t* grid)
{
    for (int_t i = 0; i < HyP->RemainBlk; ++i)
    {
        int_t lib = LBi( HyP->Remain_info[i].ib, grid) ;
        HyP->Ublock_dirty_bit[lib] = k0;
    }


    for (int_t j = 0; j < HyP->jj_cpu; ++j)
    {
        int_t ljb = LBj( HyP->Ublock_info_Phi[j].jb, grid) ;
        HyP->Lblock_dirty_bit[ljb] = k0;
    }
    return 0;
}

int_t scuStatUpdate(
    int_t knsupc,
    HyP_t* HyP, 
    SCT_t* SCT,
    SuperLUStat_t *stat
    )
{
    int_t Lnbrow   = HyP->lookAheadBlk == 0 ? 0 : HyP->lookAhead_info[HyP->lookAheadBlk - 1].FullRow;
    int_t Rnbrow   = HyP->RemainBlk == 0 ? 0 : HyP->Remain_info[HyP->RemainBlk - 1].FullRow;
    int_t nbrow = Lnbrow + Rnbrow;
    int_t ncols_host = HyP->num_u_blks == 0 ? 0 : HyP->Ublock_info[HyP->num_u_blks - 1].full_u_cols;
    int_t ncols_Phi = HyP->num_u_blks_Phi == 0 ? 0 : HyP->Ublock_info_Phi[HyP->num_u_blks_Phi - 1].full_u_cols;
    int_t ncols = ncols_Phi+ ncols_host;
    // int_t ncols   = HyP->Ublock_info[HyP->num_u_blks - 1].full_u_cols
    //           + HyP->Ublock_info_Phi[HyP->num_u_blks_Phi - 1].full_u_cols; // ###
    SCT->LookAheadRowSepMOP  += 2 * (double)knsupc * (double)(nbrow);
    SCT->GatherMOP += 2 * (double)HyP->ldu * (double)ncols;


    SCT->LookAheadGEMMFlOp   += 2 * ((double)Lnbrow * (double)HyP->ldu * (double)ncols_host +
                                     (double)Lnbrow * (double)HyP->ldu_Phi * (double)ncols_Phi) ;
    SCT->LookAheadScatterMOP += 3 * Lnbrow * ncols;
    SCT->schur_flop_counter  += 2 * ((double)Rnbrow * (double)HyP->ldu * (double)ncols_host +
                                     (double)Rnbrow * (double)HyP->ldu_Phi * (double)ncols_Phi) ;
    SCT->scatter_mem_op_counter +=  3 * Rnbrow * ncols;
    stat->ops[FACT]     += 2 * ((double)(Rnbrow + Lnbrow) * (double)HyP->ldu * (double)ncols_host +
                                (double)(Rnbrow + Lnbrow) * (double)HyP->ldu_Phi * (double)ncols_Phi) ;

    return 0;

}
