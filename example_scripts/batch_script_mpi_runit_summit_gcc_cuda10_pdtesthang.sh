#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue

#BSUB -P CSC289
#BSUB -W 2:00
#BSUB -nnodes 45
#BSUB -alloc_flags gpumps
#BSUB -J superlu_gpu




EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module load essl
module load netlib-lapack/3.8.0
module load gcc/7.5.0
module load cmake
module load cuda/10.1.243
#module unload darshan-runtime

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$MEMBERWORK/csc289/matrix
#INPUT_DIR=$MEMBERWORK/csc289/matrix/HTS
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME
FILETEST=$CUR_DIR/TEST/pdtest

FILE_NAME3D=pddrive3d
FILE3D=$FILE_DIR/$FILE_NAME3D

# nprows=(1  1 1)
# npcols=(1 1 1)  
# npz=(1 2 4)


nprows=(1 )
npcols=(1)  
npz=(1 )


#export NUM_GPU_STREAMS=1


 
for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}

# NROW=36
CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
RANK_PER_RS=1


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

for GPU_PER_RANK in  1
do
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
GPU_PER_RS=`expr $RANK_PER_RS \* $GPU_PER_RANK`


export NSUP=20
export NREL=8

export MAX_BUFFER_SIZE=500000000
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_BIND_MPI_GPU=1
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
# export GPU3DVERSION=1
# export NEW3DSOLVE=1    # Note: SUPERLU_ACC_OFFLOAD=1 and GPU3DVERSION=1 still do CPU factorization after https://github.com/xiaoyeli/superlu_dist/commit/035106d8949bc3abf86866aea1331b2948faa1db#diff-44fa50297abaedcfaed64f93712850a8fce55e8e57065d96d0ba28d8680da11eR223



#for MAT in copter2.mtx epb3.mtx gridgena.mtx vanbody.mtx shipsec1.mtx dawson5.mtx gas_sensor.mtx rajat16.mtx 
# for MAT in copter2.mtx
 # for MAT in rajat16.mtx
# for MAT in ExaSGD/118_1536/globalmat.datnh
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx xenon2.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx torsion1.mtx boyd1.bin hvdc2.mtx rajat16.mtx hcircuit.mtx 
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx av41092.mtx xenon2.mtx c-71.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx c-59.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx finan512.mtx twotone.mtx torsion1.mtx jan99jac120.mtx boyd1.mtx c-73b.mtx hvdc2.mtx rajat16.mtx hcircuit.mtx 
#for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin matrix121.dat matrix211.dat
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin 

#for MAT in matrix05_ntime=2/s1_mat_0_126936.bin A30_P0/A30_015_0_25356.bin
#for MAT in ../s1_mat_0_126936.bin copter2.mtx epb3.mtx gridgena.mtx vanbody.mtx shipsec1.mtx dawson5.mtx gas_sensor.mtx jan99jac120_x.mtx rajat16.mtx  																															 
#for MAT in ../full_1000.rua
#for MAT in ../globalmat118_1536.bin
#for MAT in ../mat_it_001_sml.mtx ../mat_it_001_med.mtx
#for MAT in ../StocF-1465.bin ../atmosmodd.mtx ../Transport.mtx
#for MAT in   ../atmosmodd.mtx
# for MAT in epb3.mtx
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin
# for MAT in s1_mat_0_126936.bin 
# for MAT in s1_mat_0_507744.bin
# for MAT in A30_015_0_25356.bin
# for MAT in torso3.rb
# for MAT in s2D9pt2048.rua
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in matrix_ACTIVSg10k_AC_00.mtx
#for MAT in ../s1_mat_0_126936.bin
# for MAT in A30_P0/A30_015_0_25356.bin
 # for MAT in A64/A64_001_0_1204992.bin
# for MAT in big.rua
for MAT in g20.rua
 # for MAT in atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin cage13.bin
#for MAT in copter2.mtx epb3.mtx gridgena.mtx 
# for MAT in ../big.rua
# for MAT in /mathias/DG_GrapheneDisorder_8192.bin /mathias/DNA_715_64cell.bin /mathias/LU_C_BN_C_4by2.bin /mathias/Li4244.bin 
#for MAT in ../Transport.mtx
# for MAT in LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin
  do
    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    mkdir -p ${MAT}_summit
    export SUPERLU_ACC_SOLVE=1
	# echo "jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH $FILE -c $NCOL -r $NROW -i 0 $INPUT_DIR/$MAT | tee ./${MAT}_summit_new_LU/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}"
  #   jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof --profile-from-start off $FILE -c $NCOL -r $NROW -i 0 $INPUT_DIR/$MAT | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}
  echo "jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof --profile-from-start off $FILETEST -r $NROW -c $NCOL -x $NREL -m $NSUP -b 2 -s 1 -f $INPUT_DIR/$MAT  | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}"  
    jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof --profile-from-start off $FILETEST -r $NROW -c $NCOL -x $NREL -m $NSUP -b 5 -s 1 -f $INPUT_DIR/$MAT  | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}

    # export NEW3DSOLVE=1
    # export NEW3DSOLVETREECOMM=1
    # unset SUPERLU_ACC_SOLVE
    # jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof --profile-from-start off $FILE3D -c $NCOL -r $NROW -d $NPZ -i 0 $INPUT_DIR/$MAT | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}_3d_newest_cpu   


    # export NEW3DSOLVE=1
    # export NEW3DSOLVETREECOMM=1
    # # export SUPERLU_ACC_SOLVE=1
    # jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' nvprof --profile-from-start off $FILE3D -c $NCOL -r $NROW -d $NPZ -i 0 $INPUT_DIR/$MAT | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}_3d_newest_gpu   

  done
#one

done
done
done
exit $EXIT_SUCCESS

