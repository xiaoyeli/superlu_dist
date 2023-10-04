module unload cray-mpich
module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi

# module load cudatoolkit
module load cgpu
module load cuda/11.1.1
module load openmpi/4.0.3
module load nsight-systems


export OMP_NUM_THREADS=5
export NUM_CUDA_STREAMS=1 # no effect in 3d
export N_CUDA_STREAMS=1  # no effect in 3d
export GPU3DVERSION=1   

# export NSUP=256
# export NREL=256

# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua
DIR=$CFS/ntrain9/YangLiu/matrix
# INPUT_DIR=$CFS/m2957/liuyangz/my_research/matrix

# export NSUP=128
# export NREL=20
export NSUP=256
export NREL=256

export NSUP=128
export NREL=20
# for MAT in big.rua 
# for MAT in g20.rua 
for MAT in g4.rua 
# for MAT in s1_mat_0_126936.bin
# for MAT in torso3.mtx
# for MAT in Graphene2880/H.mtx
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
# srun -n 1 nsys profile --stats=true ./EXAMPLE/pddrive -r 1 -c 1 $DIR/$MAT
# srun -n 1 nsys profile --stats=true ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 nsys profile --stats=true  ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 ncu -f --set full --launch-count 40 --target-processes all -o gpu3d-opt ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
mpirun -n 1 ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 $DIR/$MAT
# srun -n 1 ./EXAMPLE/pddrive3d -r 1 -c 1 -d 1 ../EXAMPLE/big.rua
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 $DIR/$MAT
done 

