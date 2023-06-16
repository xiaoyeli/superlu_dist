# superlu_dist batch script for Perlmutter CPU-only compute nodes
# gnu compiler
# updated 2023/04/01

# please make sure the following module loads/unloads match your build script
module unload gpu
#module load PrgEnv-gnu
#module load gcc/11.2.0
#module load cmake/3.24.3
#module load cudatoolkit/11.7




# export SUPERLU_LBS=ND  # this is causing crash
export MAX_BUFFER_SIZE=50000000
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
# export GPU3DVERSION=1
# export NEW3DSOLVE=1    # Note: SUPERLU_ACC_OFFLOAD=1 and GPU3DVERSION=1 still do CPU factorization after https://github.com/xiaoyeli/superlu_dist/commit/035106d8949bc3abf86866aea1331b2948faa1db#diff-44fa50297abaedcfaed64f93712850a8fce55e8e57065d96d0ba28d8680da11eR223

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
npcols=(1)
npz=(2)
NTH=1
NODE_VAL_TOT=1

for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}

CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
NODE_VAL=`expr $CORE_VAL / $CORES_PER_NODE`
MOD_VAL=`expr $CORE_VAL % $CORES_PER_NODE`
if [[ $MOD_VAL -ne 0 ]]
then
  NODE_VAL=`expr $NODE_VAL + 1`
fi

NCORE_VAL_TOT=`expr $NODE_VAL \* $CORES_PER_NODE / $NTH`


OMP_NUM_THREADS=$NTH
TH_PER_RANK=`expr $NTH \* 2`

export OMP_NUM_THREADS=$NTH
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_MAX_THREAD_SAFETY=multiple


# export NSUP=256
# export NREL=256
# for MAT in big.rua
# for MAT in g20.rua
for MAT in s1_mat_0_126936.bin
# for MAT in s1_mat_0_507744.bin
# for MAT in StocF-1465.bin
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in atmosmodj.bin StocF-1465.bin  globalmat118_1536.bin
do
mkdir -p $MAT
# # pddrive
# echo "srun -n $NCORE_VAL_TOT -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d"
# srun -n $NCORE_VAL_TOT -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive -c $NCOL -r $NROW $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_${NTH}_1rhs_2d

# pddrive3d
echo "srun -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d"
# srun -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d_old
export NEW3DSOLVE=1
srun -n $CORE_VAL -N $NODE_VAL -c $TH_PER_RANK --cpu_bind=cores ./EXAMPLE/pddrive3d -c $NCOL -r $NROW -d $NPZ $CFS/m2957/liuyangz/my_research/matrix/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_${NTH}_1rhs_3d


done


done



