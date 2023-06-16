#!/bin/bash
#BSUB -W 01:00
#BSUB -nnodes 1
#BSUB -alloc_flags nvsolve
#BSUB -J superlu_gpu

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module load cuda
module load essl
module load cmake


# export NVSHMEM_DEBUG=TRACE
# export NVSHMEM_DEBUG_SUBSYS=ALL
# export NVSHMEM_DEBUG_FILE=nvdebug_success

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$MEMBERWORK/csc289/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME
FILE_NAME3D=pddrive3d
FILE3D=$FILE_DIR/$FILE_NAME3D
# CPDIR=/ccs/home/nanding/myproject/superLU/nvshmem_new_U/run_nvshmem270_cuda1103_20221212/EXAMPLE
# cp $CPDIR/pddrive $CUR_DIR/EXAMPLE/ -rfv

export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export NVSHMEM_LMPI=-lmpi_ibm
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=0
export SUPERLU_ACC_SOLVE=1
export NEW3DSOLVE=1
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_MPI_PROCESS_PER_GPU=1 # 2: this can better saturate GPU
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
export BATCH_COUNT=0



NREP=1
nprows=(2)
npcols=(1)  
npz=(2 )

#matrix=(LU_C_BN_C_2by2.bin) #s1_mat_0_253872.bin) #s1_mat_0_507744.bin Li4244.bin DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
# matrix=(g20.rua) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
matrix=(s1_mat_0_126936.bin ) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
# matrix=(s1_mat_0_126936.bin) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
# matrix=(s2D9pt2048.rua) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
export NVSHMEM_HOME=/ccs/home/liuyangz/my_software/nvshmem_src_2.8.0-3/
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
#export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
#export NVSHMEM_BOOTSTRAP=MPI
MYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
for ((i = 0; i < ${#npcols[@]}; i++)); do
	NROW=${nprows[i]}
	NCOL=${npcols[i]}
	NPZ=${npz[i]}

	CORE_VAL=`expr $NCOL \* $NROW \* $NPZ`
	RANK_PER_RS=1
	GPU_PER_RANK=1


	PARTITION=regular
	LICENSE=SCRATCH
	TIME=00:20:00

	if [[ $NERSC_HOST == edison ]]
	then
	  CONSTRAINT=0
	fi
	
	if [[ $NERSC_HOST == cori ]]
	then
	  CONSTRAINT=haswell
	fi

	for GPU_PER_RANK in  1
	do
		for NTH in 1
		do

			RS_VAL=`expr $CORE_VAL / $RANK_PER_RS`
			MOD_VAL=`expr $CORE_VAL % $RANK_PER_RS`
			if [[ $MOD_VAL -ne 0 ]]
			then
			  RS_VAL=`expr $RS_VAL + 1`
			fi
			OMP_NUM_THREADS=$NTH
			TH_PER_RS=`expr $NTH \* $RANK_PER_RS \* 2`
			GPU_PER_RS=`expr $RANK_PER_RS \* $GPU_PER_RANK`

			for MAT in ${matrix[@]}  ##big.rua   #A30_015_0_25356.bin
  			do
				export OMP_NUM_THREADS=$OMP_NUM_THREADS
				mkdir -p ${MAT}_summit
				mya=`expr $NCOL \* $NROW \* $NPZ`
				echo "matrix: ${MAT},   ${mya} GPUs"

				for ii in `seq 1 $NREP`
    			do				
				#  jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH
				# jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS $FILE -c $NCOL -r $NROW -i 0 $INPUT_DIR/$MAT | tee -a ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}_2d_newest_gpu_nvshmem_${MYDATE}   
				jsrun --smpiargs="-gpu" -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS $FILE3D -c $NCOL -r $NROW -d $NPZ -i 0 -b $BATCH_COUNT $INPUT_DIR/$MAT | tee -a ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}_3d_newest_gpu_nvshmem_${MYDATE}   
				# jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS $FILE -c $NCOL -r $NROW -i 0 $INPUT_DIR/$MAT | tee -a ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}_3d_newest_gpu_nvshmem_${MYDATE}   
				done
			done ## matrix
		done #NTH		
	done #GPU per RANK
done # npcol
exit $EXIT_SUCCESS

