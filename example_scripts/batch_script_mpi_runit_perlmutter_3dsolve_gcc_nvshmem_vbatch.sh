#!/bin/bash
#
#modules:
# module load cpe/23.03
# module load PrgEnv-gnu
# module load gcc/11.2.0
module load cmake
# module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
# module load cray-libsci/22.11.1.2
# module load cray-libsci/23.02.1.1
ulimit -s unlimited
#MPI settings:
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
#SUPERLU settings:


export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=1
export ANC25D=0
export NEW3DSOLVE=1    
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU
export SUPERLU_ACC_SOLVE=1

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
nmpipergpu=1
export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # 2: this can better saturate GPU

##NVSHMEM settings:
NVSHMEM_HOME=/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/nvshmem_src_2.8.0-3/build/
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false
export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=libfabric

#export NVSHMEM_DEBUG=TRACE
#export NVSHMEM_DEBUG_SUBSYS=ALL
#export NVSHMEM_DEBUG_FILE=nvdebug_success

if [[ $NERSC_HOST == edison ]]; then
  CORES_PER_NODE=24
  THREADS_PER_NODE=48
elif [[ $NERSC_HOST == cori ]]; then
  CORES_PER_NODE=32
  THREADS_PER_NODE=64
  # This does not take hyperthreading into account
elif [[ $NERSC_HOST == perlmutter ]]; then
  CORES_PER_NODE=64
  THREADS_PER_NODE=128
  GPUS_PER_NODE=4
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi

# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 ../EXAMPLE/g20.rua
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 $CFS/m2957/tianyi/matrix/Landau120mat.dat
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 -f .rua $CFS/m2957/tianyi/matrix/repg20/g20
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 64 -f .dat $CFS/m2957/tianyi/matrix/Landau/Landau
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 512 -f .dat $CFS/m2957/tianyi/matrix/Landau_small/Landau
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/Landau_small/Landau
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -e 0 -p 0 -b 72 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_72_rowperm_0_equil_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -e 0 -p 1 -b 72 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_72_rowperm_1_equil_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -p 0 -b 72 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_72_rowperm_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -p 1 -b 72 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_72_rowperm_1 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 -f .dat $CFS/m2957/tianyi/matrix/isooctane_test/isooctane
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 78 -f .dat $CFS/m2957/tianyi/matrix/dodecane/dodecane
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/dodecane/dodecane
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 67 -f .dat $CFS/m2957/tianyi/matrix/drm19/drm19
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/drm19/drm19
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 73 -f .dat $CFS/m2957/tianyi/matrix/gri12/gri12
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/gri12/gri12
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 90 -f .dat $CFS/m2957/tianyi/matrix/gri30/gri30
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/gri30/gri30
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 100 -f .dat $CFS/m2957/tianyi/matrix/lidryer/lidryer
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/lidryer/lidryer

# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -e 0 -p 0 -b 10 -f .mtx $CFS/m2957/tianyi/matrix/collision_matrices/A_ | tee $CFS/m2957/tianyi/superlu_results/SLU.o_collision_batch_10_rowperm_0_equil_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -e 0 -p 1 -b 10 -f .mtx $CFS/m2957/tianyi/matrix/collision_matrices/A_ | tee $CFS/m2957/tianyi/superlu_results/SLU.o_collision_batch_10_rowperm_1_equil_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -p 0 -b 10 -f .mtx $CFS/m2957/tianyi/matrix/collision_matrices/A_ | tee $CFS/m2957/tianyi/superlu_results/SLU.o_collision_batch_10_rowperm_0 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -p 1 -b 10 -f .mtx $CFS/m2957/tianyi/matrix/collision_matrices/A_ | tee $CFS/m2957/tianyi/superlu_results/SLU.o_collision_batch_10_rowperm_1 >/dev/null

# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 30 $CFS/m2957/tianyi/matrix/Landau120mat.dat | tee $CFS/m2957/tianyi/superlu_results/SLU.o_L120_batch_30 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 30 -f .dat $CFS/m2957/tianyi/matrix/Landau120perturbed/L120perturbed | tee $CFS/m2957/tianyi/superlu_results/SLU.o_L120perturbed_batch_30 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 30 -f .dat $CFS/m2957/tianyi/matrix/Landau120rcm_perturbed/L120rcm_perturbed | tee $CFS/m2957/tianyi/superlu_results/SLU.o_L120rcm_perturbed_batch_30 >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d $CFS/m2957/tianyi/matrix/Landau120perturbed/L120perturbed29.dat

# for ((batch = 1; batch < 21; batch++)); do
# echo "Landau 3D 120 cells matrix with $batch batches, size 3403-by-3403 and nnz 381187"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch -i 1 -f .dat $CFS/m2957/tianyi/matrix/Landau120perturbed/L120perturbed | tee $CFS/m2957/tianyi/superlu_results/Landau120perturbed/SLU.o_L120perturbed_batch_${batch} >/dev/null
# done

# for ((i = 1; i < 11; i++)); do
# batch=`expr $i \* 10`
# echo "Landau 3D 120 cells matrix with $batch batches, size 3403-by-3403 and nnz 381187"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch -i 1 -f .dat $CFS/m2957/tianyi/matrix/Landau120perturbed/L120perturbed | tee $CFS/m2957/tianyi/superlu_results/Landau120perturbed/SLU.o_L120perturbed_largerbatch_${batch} >/dev/null
# done

# for ((batch = 1; batch < 21; batch++)); do
# echo "XGC matrix with $batch batches, size 1640-by-1640 and nnz 14278"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch -i 1 -f .dat $CFS/m2957/tianyi/matrix/xgcrun/xgc | tee $CFS/m2957/tianyi/superlu_results/xgc/SLU.o_xgc_batch_${batch} >/dev/null
# done

# for ((i = 1; i < 12; i++)); do
# batch=`expr $i \* 10`
# echo "XGC matrix with $batch batches, size 1640-by-1640 and nnz 14278"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch -i 1 -f .dat $CFS/m2957/tianyi/matrix/xgcrun/xgc | tee $CFS/m2957/tianyi/superlu_results/xgc/SLU.o_xgc_largerbatch_${batch} >/dev/null
# done

# for ((batch = 1; batch < 21; batch++)); do
# echo "Repeat Landau 3D 120 cells matrix with $batch batches, size 3403-by-3403 and nnz 381187"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch $CFS/m2957/tianyi/matrix/Landau120mat.dat | tee $CFS/m2957/tianyi/superlu_results/Landau120repeated/SLU.o_Landau120_batch_${batch} >/dev/null
# done

# for ((i = 1; i < 11; i++)); do
# batch=`expr $i \* 10`
# echo "Repeat Landau 3D 120 cells matrix with $batch batches, size 3403-by-3403 and nnz 381187"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch $CFS/m2957/tianyi/matrix/Landau120mat.dat | tee $CFS/m2957/tianyi/superlu_results/Landau120repeated/SLU.o_Landau120_largerbatch_${batch} >/dev/null
# done

# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 200 -f .dat $CFS/m2957/tianyi/matrix/xgcrun/xgc | tee $CFS/m2957/tianyi/superlu_results/SLU.o_xgc_batch_200 >/dev/null

# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 1 $CFS/m2957/tianyi/matrix/L120debug/L120debug1.dat
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 0 $CFS/m2957/tianyi/matrix/L120debug/L120debug1.dat
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 1 -i 1 $CFS/m2957/tianyi/matrix/L120debug/L120debug1.dat
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 -i 1 -f .dat $CFS/m2957/tianyi/matrix/L120debug/L120debug | tee $CFS/m2957/tianyi/superlu_results/SLU.o_L120debug_batch_2_ir >/dev/null
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 -f .dat $CFS/m2957/tianyi/matrix/L120debug/L120debug | tee $CFS/m2957/tianyi/superlu_results/SLU.o_L120debug_batch_2 >/dev/null

# echo "Repeat g20 twice with 2 batches"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 2 ../EXAMPLE/g20.rua | tee $CFS/m2957/tianyi/superlu_results/SLU.o_g20_batch_2 >/dev/null

# echo "Repeat Landau 3D 120 cells matrix with 3 batches, size 3403-by-3403 and nnz 381187"
# # Todo: try larger batch count, probably up to 1000
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 $CFS/m2957/tianyi/matrix/Landau120mat.dat | tee $CFS/m2957/tianyi/superlu_results/SLU.o_Landau120_batch_3

# echo "Original Landau matrices, size 193-by-193 and nnz 4417, batch size 512"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 512 -f .dat $CFS/m2957/tianyi/matrix/Landau_small/Landau | tee $CFS/m2957/tianyi/superlu_results/SLU.o_Landau_batch_512
# # Looking closely, the matrices are also just a repetition of one single matrix

# echo "Isooctane matrices from the Pele example, size 144-by-144 and nnz 6135, batch size 72"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 72 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_72

# echo "First three isooctane matrices from the Pele example, size 144-by-144 and nnz 6135, this example shows different diagonal scaling"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/isooctane/isooctane | tee $CFS/m2957/tianyi/superlu_results/SLU.o_isooctane_batch_3

# echo "Lidryer matrices from the Pele example, size 10-by-10 and nnz 91, batch size 100"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 100 -f .dat $CFS/m2957/tianyi/matrix/lidryer/lidryer | tee $CFS/m2957/tianyi/superlu_results/SLU.o_lidryer_batch_100
# # These are really small and almost dense matrices, however, their importance manifest in the next example

# echo "First three lidryer matrices from the Pele example, size 10-by-10 and nnz 91"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b 3 -f .dat $CFS/m2957/tianyi/matrix/lidryer/lidryer | tee $CFS/m2957/tianyi/superlu_results/SLU.o_lidryer_batch_3

# echo "Variable size batch example"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 2 -f .dat $CFS/m2957/tianyi/matrix/vbatch_test/test | tee $CFS/m2957/tianyi/superlu_results/SLU.o_vbatch_test_2

echo "Variable size batch EMT example"
srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun1/EMTrun1_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun1_total
srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun1/EMTrun1_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun1_2
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun1/EMTrun1_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun1_total
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 3 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun2/EMTrun2_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun2_3
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun2/EMTrun2_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun2_total
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun3/EMTrun3_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun3_2
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun3/EMTrun3_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun3_total
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun4/EMTrun4_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun4_2
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun4/EMTrun4_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun4_total
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun1/EMTrun1_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun1_2_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun1/EMTrun1_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun1_total_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 3 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun2/EMTrun2_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun2_3_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun2/EMTrun2_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun2_total_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun3/EMTrun3_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun3_2_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun3/EMTrun3_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun3_total_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 2 -f .dat $CFS/m2957/tianyi/matrix/EMTrun/EMTrun4/EMTrun4_ | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun4_2_p_0
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d_vbatch -p 0 -b 1 $CFS/m2957/tianyi/matrix/EMTrun/EMTrun4/EMTrun4_total.dat | tee $CFS/m2957/tianyi/superlu_results/EMT_results/SLU.o_vbatch_EMTrun4_total_p_0

# batchCount=(1 10 100 500 1000)
# # batchCount=(1000)
# for ((i = 0; i < ${#batchCount[@]}; i++)); do
# batch=${batchCount[i]}

# echo "Repeat Landau 3D 120 cells matrix with $batch batches, size 3403-by-3403 and nnz 381187"
# srun -n 1 -N 1 ./EXAMPLE/pddrive3d -b $batch $CFS/m2957/tianyi/matrix/Landau120mat.dat | tee $CFS/m2957/tianyi/superlu_results/SLU.o_Landau120_batch_${batch} >/dev/null
# done


# nprows=(1)
# npcols=(1)
# npz=(1)
# nrhs=(1)
# NTH=1
# NREP=1
# # NODE_VAL_TOT=1

# for ((i = 0; i < ${#npcols[@]}; i++)); do
# NROW=${nprows[i]}
# NCOL=${npcols[i]}
# NPZ=${npz[i]}
# for ((s = 0; s < ${#nrhs[@]}; s++)); do
# NRHS=${nrhs[s]}
# CORE_VAL2D=`expr $NCOL \* $NROW`
# NODE_VAL2D=`expr $CORE_VAL2D / $GPUS_PER_NODE`
# MOD_VAL=`expr $CORE_VAL2D % $GPUS_PER_NODE`
# if [[ $MOD_VAL -ne 0 ]]
# then
#   NODE_VAL2D=`expr $NODE_VAL2D + 1`
# fi

# CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
# NODE_VAL=`expr $CORE_VAL / $GPUS_PER_NODE`
# MOD_VAL=`expr $CORE_VAL % $GPUS_PER_NODE`
# if [[ $MOD_VAL -ne 0 ]]
# then
#   NODE_VAL=`expr $NODE_VAL + 1`
# fi

# # NODE_VAL=2
# # NCORE_VAL_TOT=`expr $NODE_VAL_TOT \* $CORES_PER_NODE / $NTH`
# batch=0 # whether to do batched test
# NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
# NCORE_VAL_TOT2D=`expr $NROW \* $NCOL `

# OMP_NUM_THREADS=$NTH
# TH_PER_RANK=`expr $NTH \* 2`

# export OMP_NUM_THREADS=$NTH
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread
# export SLURM_CPU_BIND="cores"
# export MPICH_MAX_THREAD_SAFETY=multiple

# # srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# # export NSUP=256
# # export NREL=256
# # for MAT in big.rua
# # for MAT in Geo_1438.bin
# # for MAT in g20.rua
# # for MAT in s1_mat_0_253872.bin s2D9pt2048.rua
# # for MAT in dielFilterV3real.bin
# for MAT in rma10.mtx 
# # for MAT in raefsky3.mtx
# # for MAT in s2D9pt2048.rua raefsky3.mtx rma10.mtx
# # for MAT in s1_mat_0_126936.bin  # for MAT in s1_mat_0_126936.bin
# # for MAT in s2D9pt2048.rua
# # for MAT in nlpkkt80.bin dielFilterV3real.bin Ga19As19H42.bin
# # for MAT in dielFilterV3real.bin 
# # for MAT in s2D9pt1536.rua
# # for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# # for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# # for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# # for MAT in temp_13k.mtx
# do
# mkdir -p $MAT
# for ii in `seq 1 $NREP`
# do	

# # SUPERLU_ACC_OFFLOAD=0
# # srun -n $NCORE_VAL_TOT2D -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_gpu_${SUPERLU_ACC_OFFLOAD}

# # SUPERLU_ACC_OFFLOAD=1
# # srun -n $NCORE_VAL_TOT2D -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_gpu_${SUPERLU_ACC_OFFLOAD}_nmpipergpu${nmpipergpu}

# SUPERLU_ACC_OFFLOAD=1
# export GPU3DVERSION=0
# echo "srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}_gpu_${SUPERLU_ACC_OFFLOAD}_cpp_${GPU3DVERSION}"
# srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}_gpu_${SUPERLU_ACC_OFFLOAD}_cpp_${GPU3DVERSION}_nmpipergpu${nmpipergpu}

# SUPERLU_ACC_OFFLOAD=1
# export GPU3DVERSION=1
# echo "srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}_gpu_${SUPERLU_ACC_OFFLOAD}_cpp_${GPU3DVERSION}"
# srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}_gpu_${SUPERLU_ACC_OFFLOAD}_cpp_${GPU3DVERSION}_nmpipergpu${nmpipergpu}


# # export SUPERLU_ACC_SOLVE=1
# # srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores valgrind --leak-check=yes ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}


# done

# done
# done
# done
