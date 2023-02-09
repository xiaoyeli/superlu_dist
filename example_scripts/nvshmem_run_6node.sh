#!/bin/bash
#BSUB -P BIF115
#BSUB -W 00:10
#BSUB -nnodes 6
#BSUB -alloc_flags gpumps
#BSUB -J superlu_gpu

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$MEMBERWORK/csc289/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME
#CPDIR=/autofs/nccs-svm1_home1/nanding/myproject/superLU/pm/superlu_dist/build_nvshmem280_cuda1103_0127/EXAMPLE
#CPDIR=/autofs/nccs-svm1_home1/nanding/myproject/superLU/pm/superlu_dist/build_nvshmem270_cuda_11013/EXAMPLE
#cp $CPDIR/pddrive $CUR_DIR/EXAMPLE/ -rfv

export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export NVSHMEM_LMPI=-lmpi_ibm
## export NVSHMEM_HCA_PE_MAPPIN  maybe??
nprows=(6)
npcols=(1)  
# MAT=s1_mat_0_126936.bin #Goodwin_127.mtx #A30_015_0_25356.bin #big.rua #s1_mat_0_126936.bin
MAT=Li4244.bin
#(s1_mat_0_126936.bin s1_mat_0_253872.bin Li4244.bin DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #s1_mat_0_253872.bin) #s1_mat_0_507744.bin Li4244.bin DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
#matrix=(DG_GrapheneDisorder_8192.bin) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
##matrix=(g20.rua) #s1_mat_0_126936.bin) #s1_mat_0_507744.bin Li4244.bin LU_C_BN_C_2by2.bin DG_GrapheneDisorder_8192.bin) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
#matrix=(Li4244.bin s1_mat_0_253872.bin) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
#export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem270_gdr23_cuda1102_11232022/ 
#export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem280_gdr23_cuda1102_20230131 
export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem280_gdr23_cuda1103_20230127/ 
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
#export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
#export NVSHMEM_BOOTSTRAP=MPI
export NSUP=256 
#export NREL=256
export MAX_BUFFER_SIZE=5000000000
export OMP_NUM_THREADS=1

export SUPERLU_ACC_SOLVE=1

#jsrun -n6 -g1 -a1  -c2 ./mpi-based-init |& tee log_nvshmem_2GPU_1NODE
#jsrun -n6 -a1 -c42 -g1 ./mpi-based-init |& tee log_nvshmem_2GPU_6NODE 
mkdir -p ${MAT}_summit
for i in {1..1}
do
MYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
#jsrun -n6 -a1 -c42  -g1  $FILE_DIR/pddrive  -c 1 -r 1 -b 1 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/batch_SLU.o_mpi_1x1_OMP_${OMP_NUM_THREADS}_GPU_1_6NODE_${MYDATE}
#jsrun -n6 -g1 -a1 -c2  $FILE_DIR/pddrive  -c 1 -r 1 -b 1 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/batch_SLU.o_mpi_1x1_OMP_${OMP_NUM_THREADS}_GPU_1_1NODE_${MYDATE}
jsrun -n6 -a1 -c42 -g1  $FILE_DIR/pddrive  -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_6NODE_${MYDATE}
jsrun -n6 -g1 -a1 -c2  $FILE_DIR/pddrive  -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_1NODE_${MYDATE}
done
#MYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
#export NVSHMEM_DEBUG=1
##jsrun -n6 -a1 -c42 -g1 ./mpi-based-init |& tee log_nvshmem_2GPU_6NODE 
##jsrun -n6 -g1 -a1  -c2 ./mpi-based-init |& tee log_nvshmem_2GPU_1NODE
#jsrun -n6 -a1 -c42  -g1  $FILE_DIR/pddrive  -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_6NODE_${MYDATE}
##jsrun -n2 -g3 -a3 -c42  $FILE_DIR/pddrive  -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_2NODE_${MYDATE}
#jsrun -n6 -g1 -a1 -c2  $FILE_DIR/pddrive  -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_1NODE_${MYDATE}
