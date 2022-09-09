
module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
module load cray-libsci/22.06.1.3
# avoid bug in cudatoolkit
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.5\/compat:/}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.7\/compat:/}
# please make sure the above module load and export LD_LIBRARY_PATH commands match your build script



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
nprows=(4)
npcols=(4)
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

NCORE_VAL_TOT=`expr $NODE_VAL_TOT \* $CORES_PER_NODE / $NTH`


OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_MAX_THREAD_SAFETY=multiple


# export NSUP=5
# export NREL=5
# for MAT in big.rua 
# for MAT in g4.rua 
for MAT in s1_mat_0_126936.bin
# for MAT in s1_mat_0_507744.bin 
# for MAT in StocF-1465.bin
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in atmosmodj.bin StocF-1465.bin  globalmat118_1536.bin 
do
mkdir -p $MAT   
echo "srun -n $NCORE_VAL_TOT -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d"
srun -n $NCORE_VAL_TOT -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d
# srun -n $NCORE_VAL_TOT -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_3d
done 


done 



