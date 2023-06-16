#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
module unload cmake
module load cmake

export SUPERLU_LBS=ND  # GD is causing hang for 4x4x16 for Ga19As19H42
# export GPU3DVERSION=1
# export NEW3DSOLVE=1    # Note: SUPERLU_ACC_OFFLOAD=1 and GPU3DVERSION=1 still do CPU factorization after https://github.com/xiaoyeli/superlu_dist/commit/035106d8949bc3abf86866aea1331b2948faa1db#diff-44fa50297abaedcfaed64f93712850a8fce55e8e57065d96d0ba28d8680da11eR223
# export COMM_TREE_MPI_WAIT=1 


CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$CFS/m2957/liuyangz/my_research/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME

TMP_BATCH_FILE=tmp_batch_file.slurm
# ^^^ Should check that this is not taken,
#     but I would never use this, so ...

> $TMP_BATCH_FILE

if [[ $NERSC_HOST == edison ]]; then
  CORES_PER_NODE=24
  THREADS_PER_NODE=48
elif [[ $NERSC_HOST == cori ]]; then
  CORES_PER_NODE=32
  THREADS_PER_NODE=64
  # This does not take hyperthreading into account
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi

# nprows=(16 8 8 )
# npcols=(16 16 8)
# npz=(2 4 8)

nprows=(8 8 4 4 2 2 1 16 8 8 4 4 2 2 16 16 8 8 4 4 2 32 16 16 8 8 4 4 64 32 16 16 8 8 4)
npcols=(16 8 8 4 4 2 2 16 16 8 8 4 2 2 32 16 16 8 8 4 4 32 32 16 16 8 8 4 32 32 32 16 16 8 8)
npz=(1 2 4 8 16 32 64 1 2 4 8 16 32 64 1 2 4 8 16 32 64 1 2 4 8 16 32 64 1 2 4 8 16 32 64)


# nprows=(32 32 16 16 8 8)
# npcols=(64 32 32 16 16 8)
# npz=(1 2 4 8 16 32)




nrhs=(1) 

# nprows=(32 32 16 16 8 8)
# npcols=(64 32 32 16 16 8)
# npz=(1 2 4 8 16 32)

# nprows=( 8)
# npcols=( 8)
# npz=(32)

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}

for ((s = 0; s < ${#nrhs[@]}; s++)); do
NRHS=${nrhs[s]}

# NROW=36
CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
NODE_VAL=`expr $CORE_VAL / $CORES_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $CORES_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL=`expr $NODE_VAL + 1`
fi
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


#for NTH in 8
for NTH in 1
do
OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`
# export NSUP=256
# export NREL=256

#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb
  # for MAT in torso3.bin
  # for MAT in g20.rua
  # for MAT in s1_mat_0_253872.bin s2D9pt2048.rua nlpkkt80.bin Ga19As19H42.bin ldoor.mtx
  for MAT in nlpkkt80.bin
  # for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin matrix_piyush/s2D9pt2048.rua  matrix_piyush/s2D9pt3072.rua
  # for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin
  # for MAT in matrix_piyush/s2D9pt1536.rua
  # for MAT in cage13.mtx StocF-1465.mtx Geo_1438.mtx Ga19As19H42.mtx torso3.mtx
# for MAT in nlpkkt80.bin StocF-1465.bin Geo_1438.bin Ga19As19H42.bin torso3.mtx Serena.mtx ldoor.mtx  
# for MAT in s1_mat_0_253872.bin matrix_piyush/s2D9pt2048.rua  matrix_piyush/s2D9pt3072.rua nlpkkt80.bin StocF-1465.bin Geo_1438.bin Ga19As19H42.bin torso3.mtx Serena.mtx ldoor.mtx  
# for MAT in s1_mat_0_253872.bin matrix_piyush/s2D9pt2048.rua  matrix_piyush/s2D9pt3072.rua nlpkkt80.bin StocF-1465.bin Geo_1438.bin Ga19As19H42.bin torso3.mtx 
# for MAT in matrix_piyush/s2D9pt2048.rua nlpkkt80.bin StocF-1465.bin Geo_1438.bin Ga19As19H42.bin torso3.mtx Serena.mtx ldoor.mtx  
# for MAT in matrix_piyush/s2D9pt2048.rua
# for MAT in Serena.mtx ldoor.mtx
# for MAT in Geo_1438.bin
  # for MAT in matrix_piyush/s2D9pt2048.rua  matrix_piyush/s2D9pt3072.rua
  # for MAT in matrix_piyush/s2D9pt3072.rua
  # for MAT in s1_mat_0_253872.bin
  # for MAT in s1_mat_7127136_7127136_0_csc_1th_block_size_1781784.bin
  # for MAT in matrix121.dat matrix211.dat tdr190k.dat tdr455k.dat nlpkkt80.mtx torso3.mtx helm2d03.mtx
  # for MAT in tdr190k.dat Ga19As19H42.mtx
 # for MAT in torso3.mtx hvdc2.mtx matrix121.dat nlpkkt80.mtx helm2d03.mtx
# for MAT in A22.bin DG_GrapheneDisorder_8192.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin
  # for MAT in  A22.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin
 # for MAT in Ga19As19H42.mtx
  do
    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export MPICH_MAX_THREAD_SAFETY=multiple
    mkdir -p $MAT

    # # pddrive
    # srun -N 64 -n $CORE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE_DIR/pddrive -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_2d

    # pddrive3d
    unset NEW3DSOLVE
    unset NEW3DSOLVETREECOMM
    echo "srun -n $CORE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE_DIR/pddrive3d -c $NCOL -r $NROW -d $NPZ $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d"
    srun -n $CORE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE_DIR/pddrive3d -c $NCOL -r $NROW -d $NPZ $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_old_nrhs_${NRHS}
    
    # export NEW3DSOLVE=1
    # unset NEW3DSOLVETREECOMM
    # srun -n $CORE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE_DIR/pddrive3d -c $NCOL -r $NROW -d $NPZ $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newer_nrhs_${NRHS}

    export NEW3DSOLVE=1
    export NEW3DSOLVETREECOMM=1
    srun -n $CORE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE_DIR/pddrive3d -c $NCOL -r $NROW -d $NPZ $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_nrhs_${NRHS}




  done
done

done
done

exit $EXIT_SUCCESS

