#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2






 module unload darshan
 module swap craype-haswell craype-mic-knl
# module load cray-fftw
# module swap intel/18.0.1.163 intel/17.0.3.191
 module load gsl
# module load cray-hdf5-parallel/1.10.0.3
 module load idl
 module load craype-hugepages2M
 module unload cray-libsci
 module load hpctoolkit 

export LIBRARY_PATH=/global/cscratch1/sd/kz21/openmp-shared/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/global/cscratch1/sd/kz21/openmp-shared/lib:$LD_LIBRARY_PATH
 
 
 
 
 
 
 
# MAX_PARAMS=1
# # ^^^ This should be fixed, as it should just loop through everything
# if [[ $# -eq 0 ]]; then
  # echo "Must have at least one parameter; exiting"
  # exit $EXIT_PARAM
# fi
# if [[ $# -gt $MAX_PARAMS ]]; then
  # echo "Too many parameters; exiting"
  # exit $EXIT_PARAM
# fi

# INPUT_FILE=$1
# # ^^^ Get the input ile

rm -rf core
# source  /global/cscratch1/sd/kz21/env-shared.sh
# export SPACK_ROOT=/project/projectdirs/m2957/liuyangz/my_software/spack
# export PATH=${SPACK_ROOT}/bin:${PATH}
# source ${SPACK_ROOT}/share/spack/setup-env.sh

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
#INPUT_DIR=/global/cscratch1/sd/liuyangz/my_research/matrix/HTS
INPUT_DIR=/project/projectdirs/m2957/liuyangz/my_research/matrix
# INPUT_DIR=`pwd`
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
  CORES_PER_NODE=64
  THREADS_PER_NODE=256
  # This does not take hyperthreading into account
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi


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


#nprows=(1 2 3 4 6 12)
#npcols=(12 6 4 3 2 1) 
 
# nprows=(2 4 8 16 32 45)
# npcols=(2 4 8 16 32 45) 

# nprows=(1 4 8 16 32 45 64)
# npcols=(1 4 8 16 32 45 64) 

nprows=(1)
npcols=(1) 

NREP=1
 
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


#for NTH in  1 32    
# for NTH in  1 2 4 8 16 32 64 128 256   
# for NTH in  1 64 128 256
for NTH in 16
do

OMP_NUM_THREADS=$NTH
#TH_PER_RANK=`expr $NTH \* 4`
TH_PER_RANK=268


#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb 
  # for MAT in torso3.mtx
  # for MAT in matrix121.dat matrix211.dat tdr190k.dat tdr455k.dat nlpkkt80.mtx torso3.mtx helm2d03.mtx  
 # for MAT in boyd1.bin
# for MAT in ExaSGD/118_1536/globalmat.datnh
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx xenon2.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx torsion1.mtx boyd1.bin hvdc2.mtx rajat16.mtx hcircuit.mtx 
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx av41092.mtx xenon2.mtx c-71.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx c-59.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx finan512.mtx twotone.mtx torsion1.mtx jan99jac120.mtx boyd1.mtx c-73b.mtx hvdc2.mtx rajat16.mtx hcircuit.mtx 
  # for MAT in matrix05_ntime=2/s1_mat_0_126936.bin
# for MAT in matrix05_ntime=2/s1_mat_0_126936.bin A30_P0/A30_015_0_25356.bin
# for MAT in matrix05_ntime=2/s1_mat_0_126936.bin

# for MAT in A30_P0/A30_015_0_25356.bin
 # for MAT in A64/A64_001_0_1204992.bin
# for MAT in A30_015_0_25356.bin
# for MAT in Geo_1438.bin
# for MAT in matrix05_ntime=2/s1_mat_0_126936.bin
for MAT in matrix05_ntime=2/s1_mat_0_126936.bin
# for MAT in A22.bin DG_GrapheneDisorder_8192.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin

# for MAT in LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin


#for MAT in big.rua
#for MAT in torso3.bin    
# for MAT in Ga19As19H42.mtx   
  do

    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread

	
    mkdir -p ${MAT}_knl_new_LU
    
	
	OUTPUT=./${MAT}_knl_new_LU/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}
	
	rm -rf $OUTPUT
	for ii in `seq 1 $NREP`
    do
	echo "srun -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee -a $OUTPUT"
	srun  -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores hpcrun -t -e REALTIME -o ${MAT}_OMP_${OMP_NUM_THREADS} $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee -a $OUTPUT
	done	 	
	
	hpcstruct ./EXAMPLE/pddrive
	hpcprof-mpi -S ./pddrive.hpcstruct -o ${MAT}_OMP_${OMP_NUM_THREADS}_dat -I/$pwd/../SRC/'*' ${MAT}_OMP_${OMP_NUM_THREADS}
	
	
    
    # Add final line (srun line) to temporary slurm script

    #cat $TMP_BATCH_FILE
    #echo " "
    # sbatch $TMP_BATCH_FILE
  done
#one

done
done

exit $EXIT_SUCCESS

