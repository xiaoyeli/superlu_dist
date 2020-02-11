#!/bin/bash
# Bash script to submit many files to Summit

#BSUB -P CSC289
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags "gpumps smt1"
#BSUB -J superlu_gpu

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module load essl
module load cmake/3.11.3
module load cuda/10.1.243
module load valgrind

CUR_DIR=`pwd`
#INPUT_DIR=$CUR_DIR/EXAMPLE						 
INPUT_DIR=$MEMBERWORK/csc289
# INPUT_DIR=$MEMBERWORK/csc289/matrix
#EX_DIR=$CUR_DIR/summit_build/EXAMPLE
EX_DIR=$CUR_DIR/
DRIVER_NAME=psdrive
EX_DRIVER=$EX_DIR/$DRIVER_NAME

nprows=(  1  )    # 1 node, 1MPI-1GPU
npcols=(  1  )  
RANK_PER_RS=1

#nprows=(  6  )    # 1 node, 7MPI-1GPU
#npcols=(  7  )  
#RANK_PER_RS=7

#nprows=(  6  )     # 7 nodes, 1MPI-1GPU
#npcols=(  7  )  
#RANK_PER_RS=1

#nprows=(  16  )    # 7 nodes, 7MPI-1GPU   
#npcols=(  18  )  
#RANK_PER_RS=7


for ((i = 0; i < ${#npcols[@]}; i++)); do

NROW=${nprows[i]}
NCOL=${npcols[i]}

# NROW=36
CORE_VAL=`expr $NCOL \* $NROW`

#PARTITION=debug
PARTITION=regular
LICENSE=SCRATCH
TIME=00:20:00

if [[ $NERSC_HOST == edison ]]
then
  CONSTRAINT=0
fi
if [[ $NERSC_HOST == cori ]]
then
  CONSTRAINT=haswell
fi

export NSUP=1024
export NREL=128

for NTH in 1  
  do

RS_VAL=`expr $CORE_VAL / $RANK_PER_RS`
MOD_VAL=`expr $CORE_VAL % $RANK_PER_RS`
if [[ $MOD_VAL -ne 0 ]]
then
  RS_VAL=`expr $RS_VAL + 1`
fi
OMP_NUM_THREADS=$NTH
TH_PER_RS=`expr $NTH \* $RANK_PER_RS`
GPU_PER_RS=1

#export NUM_CUDA_STREAMS=1

# can try these 2 matrices: Ga19As19H42.mtx, Geo_1438.mtx
for MAT in g20.rua Ga19As19H42.mtx  #Geo_1438.mtx
  do
    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    mkdir -p ${MAT}_summit
	echo "jsrun --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS --cpu_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host 6 $EX_DRIVER -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./${MAT}_summit_new_LU/mpi_${NROW}x${NCOL}_omp_${OMP_NUM_THREADS}_gpu_${GPU_PER_RANK}"
     jsrun --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS --cpu_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS  --rs_per_host 1 '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof $EX_DRIVER -c $NCOL -r $NROW $INPUT_DIR/$MAT 2>&1 | tee ./${MAT}_summit/mpi_${NROW}x${NCOL}_omp_${OMP_NUM_THREADS}_gpu
  done ## end for MAT

done  ## end for NTH
done  ## end for i ..

exit $EXIT_SUCCESS

# Other matrices: 
#nlpkkt80.mtx #stomach.mtx #Li4244.bin
# for MAT in copter2.mtx
 # for MAT in rajat16.mtx
# for MAT in ExaSGD/118_1536/globalmat.datnh
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx xenon2.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx torsion1.mtx boyd1.bin hvdc2.mtx rajat16.mtx hcircuit.mtx 
