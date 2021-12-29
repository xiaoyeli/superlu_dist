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


export OMP_NUM_THREADS=1
export NUM_CUDA_STREAMS=1
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua
DIR=/project/projectdirs/m2957/liuyangz/my_research/matrix

# export NSUP=5
# export NREL=5
# for MAT in big.rua 
# for MAT in g4.rua 
for MAT in s1_mat_0_126936.bin 
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 $DIR/$MAT
done 

