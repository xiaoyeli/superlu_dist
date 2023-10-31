#!/bin/bash
module load spack cmake
ulimit -s unlimited


#SUPERLU settings:
export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=1
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




CPUS_PER_NODE=104
THREADS_PER_NODE=208
nprows=(1 )
npcols=(1 )
npz=(1  )
nrhs=(1)
NTHS=( 4 8 16 32 52 104)
NREP=1
# NODE_VAL_TOT=1

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}
for ((g = 0; g < ${#NTHS[@]}; g++)); do
NTH=${NTHS[g]}
for ((s = 0; s < ${#nrhs[@]}; s++)); do
NRHS=${nrhs[s]}
CORE_VAL2D=`expr $NCOL \* $NROW`
NODE_VAL2D=`expr $CORE_VAL2D / $CPUS_PER_NODE`
MOD_VAL=`expr $CORE_VAL2D % $CPUS_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL2D=`expr $NODE_VAL2D + 1`
fi

CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
NODE_VAL=`expr $CORE_VAL / $CPUS_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $CPUS_PER_NODE`
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

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_MAX_THREAD_SAFETY=multiple
#export OMP_MAX_ACTIVE_LEVELS=1
#export OMP_DYNAMIC=TRUE

# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=256
# export NREL=256
# for MAT in big.rua
# for MAT in g20.rua
# for MAT in s1_mat_0_253872.bin s2D9pt2048.rua
# for MAT in dielFilterV3real.bin
#for MAT in rma10.mtx
for MAT in Geo_1438.bin
# for MAT in s2D9pt2048.rua raefsky3.mtx rma10.mtx
# for MAT in s1_mat_0_126936.bin  # for MAT in s1_mat_0_126936.bin
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

SUPERLU_ACC_SOLVE=0

mpirun -n $NCORE_VAL_TOT2D --depth $NTH --cpu-bind depth ./EXAMPLE/pddrive -c $NCOL -r $NROW -b $batch ~/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d

mpirun -n $NCORE_VAL_TOT  --depth $NTH --cpu-bind depth ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS ~/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_${NRHS}


done

done
done
done
done









