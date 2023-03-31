#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue
EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module swap PrgEnv-cray PrgEnv-gnu
module load cmake
module load rocm/5.1.0
module load cray-libsci/22.12.1.1	
module load cray-mpich/8.1.17		
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
FILE_NAME3D=pddrive3d
FILE3D=$FILE_DIR/$FILE_NAME3D
MYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
# export GPU3DVERSION=1
# export SUPERLU_ACC_SOLVE=1
export NEW3DSOLVE=1
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1

export SUPERLU_LBS=ND  # GD is causing crash for 4x4x32 for StocF-1465

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


nprows=(1)
npcols=(1) 
npz=(16)
nrhs=(1 50) 


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

export MAX_BUFFER_SIZE=500000000 
export SUPERLU_NUM_GPU_STREAMS=1
# export SUPERLU_RELAX=20
# export SUPERLU_MAXSUP=128
for NTH in 1 
do
OMP_NUM_THREADS=$NTH


#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb 
  for MAT in s1_mat_0_253872.bin s2D9pt2048.rua nlpkkt80.bin Li4244.bin Ga19As19H42.bin ldoor.mtx
  # for MAT in Li4244.bin 
  # for MAT in g20.rua
  # for MAT in s1_mat_0_253872.bin s1_mat_0_126936.bin s1_mat_0_507744.bin
  # for MAT in Ga19As19H42.mtx Geo_1438.mtx
  # for MAT in DNA_715_64cell.bin Li4244.bin
  # for MAT in Geo_1438.mtx
  # for MAT in matrix121.dat
  #  for MAT in HTS/gas_sensor.mtx HTS/vanbody.mtx HTS/ct20stif.mtx HTS/torsion1.mtx HTS/dawson5.mtx
#  for MAT in HTS/gas_sensor/gas_sensor.mtx 


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
    srun -n $CORE_VAL -c $NTH --cpu_bind=cores $FILE3D -c $NCOL -r $NROW -d $NPZ -i 0 -s $NRHS $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${OMP_NUM_THREADS}_3d_newest_gpusolve_${SUPERLU_ACC_SOLVE}_nrhs_${NRHS}
    # srun -n $CORE_VAL -c $NTH --cpu_bind=cores $FILE -c $NCOL -r $NROW -i 0 $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_2d_newest_gpusolve_${SUPERLU_ACC_SOLVE}
    # Add final line (srun line) to temporary slurm script

  done
#one

done
done
done

exit $EXIT_SUCCESS

