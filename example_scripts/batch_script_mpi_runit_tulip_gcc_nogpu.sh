#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue
EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2


module load openmpi
module load cmake
module load cuda10.2/toolkit/10.2.89

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"




CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=~/my_research/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME



CORES_PER_NODE=32



#nprows=(6 12 24)
#npcols=(6 12 24)


#nprows=(2048 1 32)
#npcols=(1 2048 64)

# nprows=(32 )
# npcols=(64 )



#nprows=(24 48 1 1 576 2304)
#npcols=(24 48 576 2304 1 1)


#nprows=(48  1  2304)
#npcols=(48  2304 1)

#nprows=(6 12 24 48 )
#npcols=(6 12 24 48 )

#nprows=(6 12 24 48 1 1 1 1 36 144 576 2304)
#npcols=(6 12 24 48 36 144 576 2304 1 1 1 1)

#nprows=(32 128 512 1 1 1 4 8 16)
#npcols=(1 1 1 32 128 512 8 16 32)

#nprows=(2048 1 32)
#npcols=(1 2048 64)




#nprows=(12 1 144)
#npcols=(12 144 1)


nprows=(1)
npcols=(1)
 
for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}

# NROW=36
CORE_VAL=`expr $NCOL \* $NROW`
NODE_VAL=`expr $CORE_VAL / $CORES_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $CORES_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL=`expr $NODE_VAL + 1`
fi


for NTH in 1 
do
OMP_NUM_THREADS=$NTH


#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb 
  for MAT in HTS/copter2.mtx
  # for MAT in HTS/rajat16.mtx
  # for MAT in big.rua
  # for MAT in matrix121.dat matrix211.dat tdr190k.dat tdr455k.dat nlpkkt80.mtx torso3.mtx helm2d03.mtx  
  # for MAT in tdr190k.dat Ga19As19H42.mtx
 # for MAT in torso3.mtx hvdc2.mtx matrix121.dat nlpkkt80.mtx helm2d03.mtx
# for MAT in A22.bin DG_GrapheneDisorder_8192.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin    
  # for MAT in  A22.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin
 # for MAT in Ga19As19H42.mtx   
  do
    # Start of looping stuff

    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    # export OMP_PLACES=threads
    # export OMP_PROC_BIND=spread
    mkdir -p $MAT
    mpirun -n $CORE_VAL $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_mrhs
    # Add final line (srun line) to temporary slurm script

  done
#one

done
done

exit $EXIT_SUCCESS

