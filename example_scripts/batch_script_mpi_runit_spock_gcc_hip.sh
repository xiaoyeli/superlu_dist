#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue
EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module swap PrgEnv-cray PrgEnv-gnu
module load cmake
module load rocm			 
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CRAYPE_LINK_TYPE=dynamic


# export MV2_USE_CUDA=1
# export MV2_ENABLE_AFFINITY=0



CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$MEMBERWORK/csc289/matrix
# INPUT_DIR=$CUR_DIR/../EXAMPLE	
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME



CORES_PER_NODE=64



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


nprows=(4)
npcols=(4)
 
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

export MAX_BUFFER_SIZE=500000000 
export NUM_GPU_STREAMS=1
for NTH in 1 
do
OMP_NUM_THREADS=$NTH


#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb 
  # for MAT in s1_mat_0_126936.bin 
  # for MAT in s1_mat_0_253872.bin s1_mat_0_126936.bin s1_mat_0_507744.bin
  # for MAT in Ga19As19H42.mtx Geo_1438.mtx
  for MAT in DNA_715_64cell.bin Li4244.bin s1_mat_0_126936.bin Ga19As19H42.mtx Geo_1438.mtx
  # for MAT in DNA_715_64cell.bin Li4244.bin
  # for MAT in Geo_1438.mtx
  # for MAT in matrix121.dat
  #  for MAT in HTS/gas_sensor.mtx HTS/vanbody.mtx HTS/ct20stif.mtx HTS/torsion1.mtx HTS/dawson5.mtx
 # for MAT in HTS/gas_sensor.mtx 


  # for MAT in HTS/g7jac160.mtx
  # for MAT in HTS/gridgena.mtx
  # for MAT in HTS/hcircuit.mtx
  # for MAT in HTS/jan99jac120.mtx
  # for MAT in HTS/shipsec1.mtx
  # for MAT in HTS/copter2.mtx
  # for MAT in HTS/epb3.mtx
  # for MAT in HTS/twotone.mtx
  # for MAT in HTS/boyd1.mtx
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
    #srun -n $CORE_VAL -c $NTH --cpu_bind=cores /opt/rocm/bin/rocprof --hsa-trace --hip-trace $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_mrhs
    #srun -n $CORE_VAL -c $NTH --cpu_bind=cores /opt/rocm/bin/rocprof --hsa-trace --roctx-trace $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_mrhs
    srun -n $CORE_VAL -c $NTH --cpu_bind=cores $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_mrhs
    # Add final line (srun line) to temporary slurm script

  done
#one

done
done

exit $EXIT_SUCCESS

