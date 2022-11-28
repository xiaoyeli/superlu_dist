#!/bin/bash
#SBATCH --job-name=1node
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -A m2956_g
###SBATCH --mail-user=nanding@lbl.gov
###SBATCH --mail-type=ALL

module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
module load cray-libsci/22.06.1.3

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#NVSHMEM settings:
export MPICH_GPU_SUPPORT_ENABLED=0
export CRAY_ACCEL_TARGET=nvidia80
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/nvshmem/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH ${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
export NVSHMEM_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/nvshmem
export NVSHMEM_PREFIX=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/nvshmem/
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.0.0
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false
export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
#export NVSHMEM_BOOTSTRAP=plugin
#export NVSHMEM_BOOTSTRAP_PLUGIN=/global/cfs/cdirs/m2956/nanding/software/MPI_Bootstrap_For_Nan/nvshmem_bootstrap_mpich.so

#export NSUP=2400
#export NREL=2400
#run the application
#INPUT_DIR=/global/cfs/cdirs/m2956/nanding/myprojects/matrix/nimrod/new
#matrix=(nimrodMatrix-B.mtx nimrodMatrix-N.mtx)
INPUT_DIR=/global/cfs/cdirs/m2956/nanding/myprojects/matrix
##INPUT_DIR=/project/projectdirs/m2956/nanding/myprojects/matrix/mathias
matrix=(s1_mat_0_126936_longint.bin s1_mat_0_507744.bin Li4244.bin LU_C_BN_C_2by2.bin DG_GrapheneDisorder_8192.bin) 
#g20.rua) #s1_mat_0_126936.bin) #A30_015_0_25356.bin) #s1_mat_0_253872.bin) #s1_mat_0_507744.bin Li4244.bin DG_GrapheneDisorder_8192.bin  LU_C_BN_C_4by2.bin LU_C_BN_C_2by2.bin) #s1_mat_0_253872.bin) #s1_mat_0_253872.bin)  #copter2.mtx) #A30_015_0_25356.bin copter2.mtx s1_mat_0_126936.bin s1_mat_0_253872.bin)
#matrix=(s1_mat_0_507744.bin s1_mat_0_253872.bin A30_015_0_25356.bin s1_mat_0_126936.bin)
#for NROW in ${myrows[@]};do 
MYDATE=$(date '+%Y-%m-%d-%H-%M')
    for MAT in ${matrix[@]}
    do
        
        srun -n 1  -c 128 --cpu_bind=cores -G 1 ./EXAMPLE/pddrive -c 1 -r 1 $INPUT_DIR/$MAT |& tee slu_${MAT}_1node_1x1_OMP${OMP_NUM_THREADS}_${MYDATE}.log
        srun -n 2  -c 64 --cpu_bind=cores -G 2  ./EXAMPLE/pddrive -c 1 -r 2 $INPUT_DIR/$MAT |& tee slu_${MAT}_1node_2x1_OMP${OMP_NUM_THREADS}_${MYDATE}.log
        srun -n 4  -c 32 --cpu_bind=cores -G 4  ./EXAMPLE/pddrive -c 1 -r 4 $INPUT_DIR/$MAT |& tee slu_${MAT}_1node_4x1_OMP${OMP_NUM_THREADS}_${MYDATE}.log
        
    done
