#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

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

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=/project/projectdirs/sparse/liuyangz/my_research/matrix
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
npcols=(8)
 
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


for NTH in 1  
do

OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`


#for NSUP in 128 64 32 16 8
#do
  # for MAT in atmosmodl.rb nlpkkt80.mtx torso3.mtx Ga19As19H42.mtx A22.mtx cage13.rb 
  for MAT in torso3.bin
  # for MAT in matrix121.dat matrix211.dat tdr190k.dat tdr455k.dat nlpkkt80.mtx torso3.mtx helm2d03.mtx  
  # for MAT in tdr190k.dat Ga19As19H42.mtx
 # for MAT in torso3.mtx hvdc2.mtx matrix121.dat nlpkkt80.mtx helm2d03.mtx
# for MAT in A22.bin DG_GrapheneDisorder_8192.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin Li4244.bin atmosmodj.bin Ga19As19H42.bin Geo_1438.bin StocF-1465.bin    
  # for MAT in  A22.bin DNA_715_64cell.bin LU_C_BN_C_4by2.bin
 # for MAT in Ga19As19H42.mtx   
  do
    # Start of looping stuff
    > $TMP_BATCH_FILE
    echo "#!/bin/bash -l" >> $TMP_BATCH_FILE
    echo " " >> $TMP_BATCH_FILE
    echo "#SBATCH -p $PARTITION" >> $TMP_BATCH_FILE
    echo "#SBATCH -N $NODE_VAL" >>  $TMP_BATCH_FILE
    echo "#SBATCH -t $TIME" >> $TMP_BATCH_FILE
    echo "#SBATCH -L $LICENSE" >> $TMP_BATCH_FILE
    echo "#SBATCH -J SLU_$MAT" >> $TMP_BATCH_FILE
    #echo "#SBATCH -o ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_async_simple_over_icollec_flat_mrhs" >> $TMP_BATCH_FILE
    #echo "#SBATCH -e ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_async_simple_over_icollec_flat_mrhs" >> $TMP_BATCH_FILE
    # echo "#SBATCH --mail-type=BEGIN" >> $TMP_BATCH_FILE
    # echo "#SBATCH --mail-type=END" >> $TMP_BATCH_FILE
    echo "#SBATCH --mail-user=liuyangzhuan@lbl.gov" >> $TMP_BATCH_FILE
    if [[ $NERSC_HOST == cori ]]
    then
      echo "#SBATCH -C $CONSTRAINT" >> $TMP_BATCH_FILE
    fi
    mkdir -p $MAT   
    echo "export OMP_NUM_THREADS=$OMP_NUM_THREADS" >> $TMP_BATCH_FILE
    echo "export KMP_NUM_THREADS=$OMP_NUM_THREADS" >> $TMP_BATCH_FILE
    echo "export MKL_NUM_THREADS=$OMP_NUM_THREADS" >> $TMP_BATCH_FILE  																			
    echo "export NSUP=128" >> $TMP_BATCH_FILE
    echo "export NREL=20" >> $TMP_BATCH_FILE
    echo "export OMP_PLACES=threads" >> $TMP_BATCH_FILE
    echo "export OMP_PROC_BIND=spread" >> $TMP_BATCH_FILE
    echo "export MPICH_MAX_THREAD_SAFETY=multiple" >> $TMP_BATCH_FILE

    echo " " >> $TMP_BATCH_FILE
    echo "FILE=$FILE" >> $TMP_BATCH_FILE
    echo "FILEMAT=$INPUT_DIR/$MAT" >> $TMP_BATCH_FILE	
    echo " " >> $TMP_BATCH_FILE
    echo "CORE_VAL=$CORE_VAL" >> $TMP_BATCH_FILE
    echo "NCOL=$NCOL" >> $TMP_BATCH_FILE
    echo "NROW=$NROW" >> $TMP_BATCH_FILE
    # This should be computed individually for each script...

    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export MPICH_MAX_THREAD_SAFETY=multiple
    srun -n $CORE_VAL -N 2 -c $TH_PER_RANK --cpu_bind=cores $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${OMP_NUM_THREADS}_mrhs
    # Add final line (srun line) to temporary slurm script

    #cat $TMP_BATCH_FILE
    #echo " "
    # sbatch $TMP_BATCH_FILE
  done
#one

done
done

exit $EXIT_SUCCESS

