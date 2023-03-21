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

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=$MEMBERWORK/csc289/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME
# CPDIR=/ccs/home/nanding/myproject/superLU/nvshmem_new_U/run_nvshmem270_cuda1103_20221212/EXAMPLE
# cp $CPDIR/pddrive $CUR_DIR/EXAMPLE/ -rfv

export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export NVSHMEM_LMPI=-lmpi_ibm
export SUPERLU_ACC_SOLVE=1


nprows=(3   )
npcols=(1   )  
#matrix=(LU_C_BN_C_2by2.bin) #s1_mat_0_253872.bin) #s1_mat_0_507744.bin Li4244.bin DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
matrix=(s1_mat_0_253872.bin) #DG_GrapheneDisorder_8192.bin LU_C_BN_C_2by2.bin) #Li4244.bin s1_mat_0_253872.bin) 
export NVSHMEM_HOME=/ccs/home/liuyangz/my_software/nvshmem_src_2.8.0-3/
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
#export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
#export NVSHMEM_BOOTSTRAP=MPI
MYDATE=$(date '+%Y-%m-%d-%H-%M-%S')
for ((i = 0; i < ${#npcols[@]}; i++)); do
	NROW=${nprows[i]}
	NCOL=${npcols[i]}

	CORE_VAL=`expr $NCOL \* $NROW`
	RANK_PER_RS=1
	GPU_PER_RANK=1
	export SUPERLU_NUM_GPU_STREAMS=1


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
			TH_PER_RS=`expr $NTH \* $RANK_PER_RS`
			GPU_PER_RS=`expr $RANK_PER_RS \* $GPU_PER_RANK`

			for MAT in ${matrix[@]}  ##big.rua   #A30_015_0_25356.bin
  			do
				export NSUP=256 
				export NREL=60
				export MAX_BUFFER_SIZE=5000000000
    				export OMP_NUM_THREADS=$OMP_NUM_THREADS
    				mkdir -p ${MAT}_summit
				echo "matrix: ${MAT},   ${NROW} GPUs"
				#if [[ $RS_VAL -eq 1 ]];then
				#	jsrun -n $RS_VAL -a $RANK_PER_RS -c ALL_CPUS -g ALL_GPUS -brs ./put_block
				#else
					mya=`expr $NCOL \* $NROW`
					if [[ $mya -le 6 ]];then
						myc=`expr 2 \* $mya` #each nvshmem rank needs 2CPU threads
						jsrun -n1 -a${mya} -c${myc}  -g${mya} -r1 $FILE  -c $NCOL -r $NROW $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${mya}_${MYDATE}
						#jsrun -n1 -a2 -c4  -g2 -r1 $FILE  -c 1 -r 2 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_2x1_OMP_${OMP_NUM_THREADS}_GPU_2_${MYDATE}
						#jsrun -n1 -a3 -c6  -g3 -r1 $FILE  -c 1 -r 3 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_3x1_OMP_${OMP_NUM_THREADS}_GPU_3_${MYDATE}
						#jsrun -n1 -a6 -c12 -g6 -r1 $FILE -c 1 -r 6 $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_6x1_OMP_${OMP_NUM_THREADS}_GPU_6_${MYDATE}
 	 				fi
					if [[ $mya -gt 6 ]];then
						myn=`expr $mya / 6`
						jsrun -n${myn} -a6 -c12  -g6 $FILE  -c $NCOL -r $NROW $INPUT_DIR/$MAT |& tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${mya}_${MYDATE}
					fi
			#	fi
			done ## matrix
		done #NTH		
	done #GPU per RANK
done # npcol
exit $EXIT_SUCCESS

