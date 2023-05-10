#!/bin/bash
# Bash script to submit many files to Summit

# Begin LSF Directives
#BSUB -P CSC289
#BSUB -W 2:00
#BSUB -nnodes 10
####BSUB -alloc_flags "gpumps smt1"
#### smt = Simultaneous Multithreading level, default is smt4
#BSUB -alloc_flags "gpumps"  ## enable GPU multiprocess service,
                             ## to run multiple MPI per GPU
#BSUB -J superlu3d_gpu

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

module load essl
module load cuda    ####cuda/10.1.243
module load valgrind
module load xl      # ??
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/summit/xl/20180502/lib

# CUR_DIR=`pwd`
CUR_DIR=EXAMPLE
#CUR_DIR=../32-batch/EXAMPLE

INPUT_DIR=$MEMBERWORK/csc289/matrix
DRIVER=pddrive3d
EX_DRIVER=$CUR_DIR/${DRIVER}

#nprows=( 1 )    # 4 nodes, 6MPI-6GPU per-node
#npcols=( 3 )  
#npdeps=( 8 )

#nprows=( 2  2  2  2 )     # 8 nodes, 1MPI-1GPU
#npcols=( 3  3  3  3 )  
#npdeps=( 1  2  4  8 )

nprows=( 2 )    # 1 node, 6MPI-6GPU
npcols=( 1 )
npdeps=( 2 )

RS_PER_HOST=6    # 6 Resource-Set
RANK_PER_RS=1     # each resource set contains 1mpi-1gpu
GPU_PER_RS=1

#nprows=(  16  )    # 7 nodes, 7MPI-1GPU   
#npcols=(  18  )  
#RANK_PER_RS=7


for ((i = 0; i < ${#npcols[@]}; i++)); do

NPROW=${nprows[i]}
NPCOL=${npcols[i]}
NPDEP=${npdeps[i]}

CORE_VAL=`expr $NPCOL \* $NPROW \* $NPDEP`

#PARTITION=debug
#PARTITION=regular
LICENSE=SCRATCH
##???  TIME=00:30:00

#export CUBLAS_NB=64 # default, number of columns per GPU stream

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export COLPERM=4 # MMD = 2, default METIS = 4
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export OMP_NUM_THREADS=1
export SUPERLU_NUM_GPU_STREAMS=4
export SUPERLU_MPI_PROCESS_PER_GPU=1 # 2: this can better saturate GPU
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
export BATCH_COUNT=0

export SUPERLU_ACC_OFFLOAD=1  # 1 is the default
export GPU3DVERSION=1    ## use new C++ code in TRF3dV100/

# export PAMI_CUDA_HOOK_THREAD=1  # required when using --smpiargs="-gpu" ??

for NTH in ${OMP_NUM_THREADS}
  do

RS_VAL=`expr $CORE_VAL / $RANK_PER_RS`
MOD_VAL=`expr $CORE_VAL % $RANK_PER_RS`
if [[ $MOD_VAL -ne 0 ]]
then
  RS_VAL=`expr $RS_VAL + 1`
fi

OMP_NUM_THREADS=$NTH
TH_PER_RS=`expr $NTH \* $RANK_PER_RS`

OUTPUT_DIR=out-${DRIVER}

#for MAT in #nd24k.rb #Ga19As19H42.rb #Geo_1438.rb audikw_1.rb Serena.rb nlpkkt80.rb 
#for MAT in atmosmodl.rb vas_stokes_4M.rb stokes.rb
#for MAT in bbmat.rb
#for MAT in Geo_1438.rb #bug.mtx #g20.rua
for MAT in s1_mat_0_126936.bin #big.rua
  do
    mkdir -p ${OUTPUT_DIR}
    jsrun --smpiargs="-gpu" --nrs $RS_PER_HOST --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS --rs_per_host $RS_PER_HOST  $EX_DRIVER -b $BATCH_COUNT -c $NPCOL -r $NPROW -d $NPDEP $INPUT_DIR/$MAT 2>&1 | tee ${OUTPUT_DIR}/${MAT}_offload=${SUPERLU_ACC_OFFLOAD}_${NPROW}x${NPCOL}x${NPDEP}_batch=${BATCH_COUNT}_LOOKAHEAD=${SUPERLU_NUM_LOOKAHEADS}.out
    
#    jsrun --nrs $RS_VAL --tasks_per_rs $RANK_PER_RS -c $TH_PER_RS --gpu_per_rs $GPU_PER_RS --rs_per_host $RS_PER_HOST cuda_memcheck $EX_DRIVER -c $NPCOL -r $NPROW -d $NPDEP -l ${LOOKAHEAD} -q ${COLPERM} $INPUT_DIR/$MAT 2>&1 | tee ${MAT}.out/myscan-offload_${SUPERLU_ACC_OFFLOAD}-${NPROW}x${NPCOL}x${NPDEP}_omp_${OMP_NUM_THREADS}_SUPERLU_MAXSUP=${SUPERLU_MAXSUP}_CPUgemm=${N_GEMM}_LOOKAHEAD=${LOOKAHEAD}_COLPERM=${COLPERM}_cuSTREAMS=${NUM_CUDA_STREAMS}.out
    
  done ## end for MAT

done  ## end for NTH
done  ## end for i ..

exit $EXIT_SUCCESS

# Other matrices: 
#nlpkkt80.mtx #stomach.mtx #Li4244.bin
# for MAT in copter2.mtx
 # for MAT in rajat16.mtx
# for MAT in ExaSGD/118_1536/globalmat.datnh
# for MAT in copter2.mtx gas_sensor.mtx matrix-new_3.mtx xenon2.mtx shipsec1.mtx xenon1.mtx g7jac160.mtx g7jac140sc.mtx mark3jac100sc.mtx ct20stif.mtx vanbody.mtx ncvxbqp1.mtx dawson5.mtx 2D_54019_highK.mtx gridgena.mtx epb3.mtx torso2.mtx torsion1.mtx boyd1.bin hvdc2.mtx rajat16.mtx hcircuit.mtx 
