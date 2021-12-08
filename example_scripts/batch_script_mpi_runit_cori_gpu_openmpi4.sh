module unload cray-mpich/7.7.6
module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load cmake/3.18.2
module load cuda/10.2.89
module load openmpi/4.0.3


export OMP_NUM_THREADS=1
export NUM_GPU_STREAMS=1
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua


export NSUP=128
export NREL=20
# for MAT in big.rua 
# for MAT in g4.rua 
for MAT in s1_mat_0_126936.bin 
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/$MAT
done 

