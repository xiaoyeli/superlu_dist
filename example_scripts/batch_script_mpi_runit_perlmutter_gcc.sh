#!/bin/bash
#
# Run SuperLU_dist examples built with GNU compiler on NERSC Perlmutter
#
# Last updated: 2023/05/01
# 
# Perlmutter is not in production and the software environment changes rapidly.
# Expect this file to be frequently updated.


# please make sure the following module loads/unloads match your build script

#module load PrgEnv-gnu
#module load gcc/11.2.0
module load cmake/3.24.3


export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# Launch MPS from a single rank per node
if [ $SLURM_LOCALID -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS nvidia-cuda-mps-control -d
fi
# Wait for MPS to start
sleep 5


# export SUPERLU_LBS=ND  # this is causing crash
export MAX_BUFFER_SIZE=50000000
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_BIND_MPI_GPU=1
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=1
# export NEW3DSOLVE=1    # Note: SUPERLU_ACC_OFFLOAD=1 and GPU3DVERSION=1 still do CPU factorization after https://github.com/xiaoyeli/superlu_dist/commit/035106d8949bc3abf86866aea1331b2948faa1db#diff-44fa50297abaedcfaed64f93712850a8fce55e8e57065d96d0ba28d8680da11eR223


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
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi
nprows=(1)
npcols=(1)
npz=(4)
NTH=1
NODE_VAL_TOT=1

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}

# CORE_VAL=`expr $NCOL \* $NROW`
# NODE_VAL=`expr $CORE_VAL / $CORES_PER_NODE`
# MOD_VAL=`expr $CORE_VAL % $CORES_PER_NODE`
# if [[ $MOD_VAL -ne 0 ]]
# then
#   NODE_VAL=`expr $NODE_VAL + 1`
# fi

# NODE_VAL=2
# NCORE_VAL_TOT=`expr $NODE_VAL_TOT \* $CORES_PER_NODE / $NTH`
batch=1 # whether to do batched test
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ \* $batch`

OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_MAX_THREAD_SAFETY=multiple

# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=256
# export NREL=256
# for MAT in big.rua
# for MAT in g20.rua
for MAT in s1_mat_0_126936.bin
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in temp_13k.mtx
do
mkdir -p $MAT
# nsys profile --stats=true -t cuda,cublas,mpi --mpi-impl mpich  srun -n 16 -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d
# nsys profile --stats=true -t cuda,cublas,mpi --mpi-impl mpich  srun -n 4 -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_3d
srun -n 1 -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d

echo "srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d"
srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d_old
export NEW3DSOLVE=1    # currently this requires SUPERLU_ACC_OFFLOAD=1 and GPU3DVERSION=1, as this combination has 2.5 ancester factorization
srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d


# srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_2d

# srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d
# srun -n $NCORE_VAL_TOT -N $NODE_VAL_TOT -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -b $batch $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_3d
done
done

# Quit MPS control daemon before exiting
if [ $SLURM_LOCALID -eq 0 ]; then
    echo quit | nvidia-cuda-mps-control
fi
