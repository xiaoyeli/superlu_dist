# superlu_dist batch script for Perlmutter CPU-only compute nodes
# gnu compiler
# updated 2023/04/01

# please make sure the following module loads/unloads match your build script
module unload gpu
#module load PrgEnv-gnu
#module load gcc/11.2.0
#module load cmake/3.24.3
#module load cudatoolkit/11.7



if [[ $NERSC_HOST == edison ]]; then
  CORES_PER_NODE=24
  THREADS_PER_NODE=48
elif [[ $NERSC_HOST == cori ]]; then
  CORES_PER_NODE=32
  THREADS_PER_NODE=64
  # This does not take hyperthreading into account
elif [[ $NERSC_HOST == perlmutter ]]; then
  CORES_PER_NODE=128
  THREADS_PER_NODE=256  
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi

# nprows=(1 2 4 8 8)
# npcols=(1 2 4 8 16)
nprows=(8)
npcols=(8)
NTH=1
NODE_VAL_TOT=1


for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}

CORE_VAL=`expr $NCOL \* $NROW`
NODE_VAL=`expr $CORE_VAL / $CORES_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $CORES_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL=`expr $NODE_VAL + 1`
fi

# NODE_VAL=2

OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`


export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_MAX_THREAD_SAFETY=multiple

MAT=s1_mat_0_126936.bin
mkdir -p $MAT   



TMP_BATCH_FILE=tmp.sh

> $TMP_BATCH_FILE
echo "#!/bin/bash -l" >> $TMP_BATCH_FILE
echo " " >> $TMP_BATCH_FILE
echo "module unload gpu" >> $TMP_BATCH_FILE
#echo "module swap PrgEnv-nvidia PrgEnv-gnu" >> $TMP_BATCH_FILE
#echo "module load gcc" >>  $TMP_BATCH_FILE
#echo "module load cmake/3.24.3" >> $TMP_BATCH_FILE
#echo "module load cudatoolkit" >> $TMP_BATCH_FILE
echo "export OMP_NUM_THREADS=$NTH" >> $TMP_BATCH_FILE
echo "export OMP_PROC_BIND=spread" >> $TMP_BATCH_FILE
echo "export MPICH_MAX_THREAD_SAFETY=multiple" >> $TMP_BATCH_FILE

NJOB=`expr $CORES_PER_NODE \* $NODE_VAL_TOT / $CORE_VAL`

for ((i=1; i<=$NJOB; i++))
do
echo "srun -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_job_${i} &" >> $TMP_BATCH_FILE
done
echo "wait" >> $TMP_BATCH_FILE
bash $TMP_BATCH_FILE
grep "FACTOR time\|SOLVE time" ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d_job_1
rm -rf $TMP_BATCH_FILE
done 



