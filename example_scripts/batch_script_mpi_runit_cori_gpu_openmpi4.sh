module unload cray-mpich
module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load cmake
module load cgpu
module load cuda/10.2.89
module load openmpi/4.0.3
# module load cuda
# module load openmpi
module load nsight-systems


export OMP_NUM_THREADS=5
export NUM_CUDA_STREAMS=1 # no effect in 3d
export N_CUDA_STREAMS=1  # no effect in 3d
export GPU3DVERSION=1   
export MPI_PROCESS_PER_GPU=1  
export MAX_BUFFER_SIZE=64000000 
export SUPERLU_ACC_OFFLOAD=1

# NSUP=256  NREL=256  OMP_NUM_THREADS=1  SUPERLU_ACC_OFFLOAD=1  GPU3DVERSION=1   MPI_PROCESS_PER_GPU=1  MAX_BUFFER_SIZE=64000000  mpirun -n 8 EXAMPLE/pddrive3d -r 1  -c 2  -d 4 ~/matrix/Graphene2880/H.mtx


# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua
DIR=$CFS/m2957/liuyangz/my_research/matrix

# export NSUP=128
# export NREL=20
export NSUP=256
export NREL=256

# for MAT in big.rua 
# for MAT in g4.rua 
# for MAT in s1_mat_0_126936_longint.bin 
for MAT in s1_mat_0_253872.bin
# for MAT in Graphene2880/H.mtx
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg10k_AC_00.mtx matrix_ACTIVSg70k_AC_00.mtx temp_13k.mtx temp_25k.mtx temp_75k.mtx
# for MAT in temp_75k.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
# srun -n 1 nsys profile --stats=true ./EXAMPLE/pddrive -r 1 -c 1 $DIR/$MAT
# srun -n 1 nsys profile --stats=true ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 nsys profile --stats=true  ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 ncu -f --set full --launch-count 40 --target-processes all -o gpu3d-opt ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
srun -n 1 ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 ../EXAMPLE/big.rua
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 $DIR/$MAT
done 

