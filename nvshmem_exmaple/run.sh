#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue
#BSUB -P BIF115
#BSUB -W 00:30
#BSUB -nnodes 2
#BSUB -alloc_flags gpumps
#BSUB -J superlu_gpu

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module load essl
module load cmake/
module load cuda/10.1.168

CUR_DIR=`pwd`
FILE_DIR=$CUR_DIR/EXAMPLE
INPUT_DIR=/ccs/home/nanding/myproject/superLU/matrix
FILE_NAME=pddrive
FILE=$FILE_DIR/$FILE_NAME
CPDIR=/ccs/home/nanding/myproject/superLU/superlu_amd/run_nvshmem/EXAMPLE

cp $CPDIR/pddrive $CUR_DIR/EXAMPLE/ -rfv

export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export NVSHMEM_LMPI=-lmpi_ibm
export NVSHMEM_DEBUG=WARN  #INFO  #TRACE
export NVSHMEM_DEBUG_FILE=nvtrace
## export NVSHMEM_HCA_PE_MAPPIN  maybe??
nprows=(4)
npcols=(1)

for ((i = 0; i < ${#npcols[@]}; i++)); do
        NROW=${nprows[i]}
        NCOL=${npcols[i]}

        CORE_VAL=`expr $NCOL \* $NROW`
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
                for NTH in 7
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

                        for MAT in g20.rua  ##big.rua   #A30_015_0_25356.bin
                        do
                                #expddort NSUP=200
                                #export NREL=200
                                export OMP_NUM_THREADS=$OMP_NUM_THREADS
                                mkdir -p ${MAT}_summit
                                #echo "jsrun -n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./${MAT}_summit_new_LU/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}"
                                #jsrun -n 2 -a 1 -cpu_per_rs ALL_CPUS -g ALL_GPUS -E USE_MPI_IN_TEST=1 -b packed:$NTH '--smpiargs=-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks' $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT | tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}
                                echo "-n $RS_VAL -a $RANK_PER_RS -c $TH_PER_RS -g $GPU_PER_RS -b packed:$NTH"
                                #jsrun -n $RS_VAL -a $RANK_PER_RS -c ALL_CPUS -g ALL_GPUS -brs ./put_block
                                #jsrun -n $RS_VAL -a $RANK_PER_RS -c ALL_CPUS -g ALL_GPUS -brs ./mpi-based-init
                                #jsrun -n $RS_VAL -a $RANK_PER_RS -c ALL_CPUS -g ALL_GPUS -brs $FILE -c $NCOL -r $NROW $INPUT_DIR/$MAT #| tee ./${MAT}_summit/SLU.o_mpi_${NROW}x${NCOL}_OMP_${OMP_NUM_THREADS}_GPU_${GPU_PER_RANK}
                        done ## matrix
                done #NTH
        done #GPU per RANK
done # npcol
exit $EXIT_SUCCESS