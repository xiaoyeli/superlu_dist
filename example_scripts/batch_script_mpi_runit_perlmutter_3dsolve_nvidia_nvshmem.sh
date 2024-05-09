#!/bin/bash
#SBATCH -A m2957
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:10:00
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus 64
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH --mail-type=BEGIN
#SBATCH -e ./tmp.err
#
#modules:
module load PrgEnv-nvidia 
module load cudatoolkit
module load cray-libsci
module load cmake
module use /global/common/software/nersc/pe/modulefiles/latest
module load nvshmem/2.11.0

# ulimit -s unlimited
#MPI settings:
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
#SUPERLU settings:


export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=0
export ANC25D=0
export NEW3DSOLVE=1    
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_MPI_PROCESS_PER_GPU=1 # 2: this can better saturate GPU
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU


##NVSHMEM settings:
# NVSHMEM_HOME=/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/nvshmem_src_2.8.0-3/build/
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
# export MPICH_OFI_NIC_POLICY=GPU

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
# nprows=(4 8 16)
# npcols=(1 1 1)
# npz=(64 32 16)
# nrhs=(1 50) 

nprows=(2 )
npcols=(1 )
npz=(1 )
nrhs=(1)

NTH=1
NREP=1
# NODE_VAL_TOT=1

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}

for ((s = 0; s < ${#nrhs[@]}; s++)); do
NRHS=${nrhs[s]}

CORE_VAL2D=`expr $NCOL \* $NROW`
NODE_VAL2D=`expr $CORE_VAL2D / $GPUS_PER_NODE`
MOD_VAL=`expr $CORE_VAL2D % $GPUS_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL2D=`expr $NODE_VAL2D + 1`
fi

CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
NODE_VAL=`expr $CORE_VAL / $GPUS_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $GPUS_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL=`expr $NODE_VAL + 1`
fi

# NODE_VAL=2
# NCORE_VAL_TOT=`expr $NODE_VAL_TOT \* $CORES_PER_NODE / $NTH`
batch=0 # whether to do batched test
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
NCORE_VAL_TOT2D=`expr $NROW \* $NCOL `

OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export SLURM_CPU_BIND="cores"
export MPICH_MAX_THREAD_SAFETY=multiple

# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=256
# export NREL=256
# for MAT in big.rua
# for MAT in g20.rua
# for MAT in s1_mat_0_253872.bin s2D9pt2048.rua
# for MAT in dielFilterV3real.bin
# for MAT in Geo_1438.bin s2D9pt2048.rua raefsky3.mtx rma10.mtx
# for MAT in Geo_1438.bin 
for MAT in s1_mat_0_126936.bin
# for MAT in s2D9pt2048.rua
# for MAT in s2D9pt1536.rua
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in temp_13k.mtx
do
mkdir -p $MAT
for ii in `seq 1 $NREP`
do	
export SUPERLU_ACC_SOLVE=1


# # # srun -n $NCORE_VAL_TOT2D -N $NODE_VAL2D -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_gpu_${SUPERLU_ACC_OFFLOAD}
# # export SUPERLU_ACC_OFFLOAD=0
# srun -n $NCORE_VAL_TOT2D -N $NODE_VAL2D -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_gpu_${SUPERLU_ACC_OFFLOAD}

# unset SUPERLU_ACC_SOLVE
# echo "srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}"
# srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}

export SUPERLU_ACC_SOLVE=1
srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}
# srun -n $NCORE_VAL_TOT  -c $TH_PER_RANK --cpu_bind=cores valgrind --leak-check=yes ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}


done

done
done
done

