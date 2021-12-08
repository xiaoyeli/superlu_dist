
module load PrgEnv-gnu
module load cpe-cuda
module load cuda
module load cmake/git-20210830

export MAX_BUFFER_SIZE=500000000
export OMP_NUM_THREADS=1
export NUM_GPU_STREAMS=1
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=5
# export NREL=5
# for MAT in big.rua 
# for MAT in g4.rua 
for MAT in s1_mat_0_507744.bin 
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 $CFS/ntrain9/YangLiu/matrix/$MAT
done 

