module unload cray-mpich
module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64
# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load cmake
module load cgpu
module load cuda/11.1.1
module load openmpi/4.0.3
module load nsight-systems

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
# srun -n 1 nsys profile --stats=true  ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/$MAT
# srun -n 1 ncu -k dlsum_bmod_inv_gpu_mrhs,dlsum_fmod_inv_gpu_mrhs --launch-count 1 --target-processes all ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/$MAT
# srun -n 1 ncu -f -k dlsum_bmod_inv_gpu_mrhs --set full --launch-count 1 --target-processes all -o trisolve_u  ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/$MAT
# srun -n 1 ncu -f -k dlsum_fmod_inv_gpu_mrhs --set full --launch-count 1 --target-processes all -o trisolve_l  "/project/projectdirs/m2957/liuyangz/my_research/superlu_dist_amd_mergefrom_master_12_01_2021/build/EXAMPLE/pddrive" -r 1 -c 1 ../../matrix/$MAT
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/$MAT
done 

