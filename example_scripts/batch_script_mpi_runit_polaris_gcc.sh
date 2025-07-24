#!/bin/bash
#
#modules:
module use /soft/modulefiles
module use /eagle/ATPESC2025/usr/modulefiles
module load track-5-numerical


export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=256000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
nmpipergpu=1
export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # 2: this can better saturate GPU


export matdir=/eagle/projects/ATPESC2025/usr/MathPackages/datafiles/
export exedir=/eagle/projects/ATPESC2025/usr/MathPackages/superlu_dist/gcc-build/EXAMPLE/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/eagle/projects/ATPESC2025/usr/MathPackages/superlu_dist/gcc-build/lib/



nprows=(1)
npcols=(1)
npz=(1)
nrhs=(1)
NTH=1
NREP=1


for ((i = 0; i < ${#npcols[@]}; i++)); do
NROW=${nprows[i]}
NCOL=${npcols[i]}
NPZ=${npz[i]}
for ((s = 0; s < ${#nrhs[@]}; s++)); do
NRHS=${nrhs[s]}



NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
NCORE_VAL_TOT2D=`expr $NROW \* $NCOL `


OMP_NUM_THREADS=$NTH
export OMP_NUM_THREADS=$NTH


for MAT in big.rua
do
mkdir -p $MAT
for ii in `seq 1 $NREP`
do	

###### Run 1 (test 2D algorithm driver, with GPU)
NROW=1
NCOL=1
NCORE_VAL_TOT2D=`expr $NROW \* $NCOL `
mpiexec -n $NCORE_VAL_TOT2D  ${exedir}/pddrive -c $NCOL -r $NROW $matdir/$MAT  | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_omp${OMP_NUM_THREADS}

##### Run 2 (test 2D algorithm driver, with GPU and different sparsity reduction ordering)
NROW=1
NCOL=1
NCORE_VAL_TOT2D=`expr $NROW \* $NCOL `
colperm=4 # 0: NATURAL, 1: MMD_ATA, 2: MMD_AT_PLUS_A, 3: COLAMD, 4: METIS_AT_PLUS_A, 5: PARMETIS,       
mpiexec -n $NCORE_VAL_TOT2D  ${exedir}/pddrive -c $NCOL -r $NROW -q $colperm $matdir/$MAT  | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}_omp${OMP_NUM_THREADS}_q${colperm}


# ###### Run 3 (test 3D algorithm driver, with GPU factorization and CPU triangualr solve)
NROW=1
NCOL=1
NPZ=1
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
mpiexec -n $NCORE_VAL_TOT ${exedir}/pddrive3d -c $NCOL -r $NROW -d $NPZ -i 0 $matdir/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_omp${OMP_NUM_THREADS}


# ###### Run 4 (test 3D algorithm driver, with GPU factorization and GPU triangualr solve)
NROW=1
NCOL=1
NPZ=1
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
export SUPERLU_ACC_SOLVE=0
mpiexec -n $NCORE_VAL_TOT ${exedir}/pddrive3d -c $NCOL -r $NROW -d $NPZ -i 0 $matdir/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_omp${OMP_NUM_THREADS}_gpusolve


###### Run 5 (test batched interface with GPU factorization and GPU triangualr solve)
NROW=1
NCOL=1
NPZ=1
NCORE_VAL_TOT=`expr $NROW \* $NCOL \* $NPZ `
export GPU3DVERSION=1
export SUPERLU_ACC_SOLVE=1 # only set 1 when NROW=NCOL=1
batch=2 # number of matrices in a batch
mpiexec -n $NCORE_VAL_TOT   ${exedir}/pddrive3d -c $NCOL -r $NROW -d $NPZ -b $batch -i 0 -s $NRHS $matdir/$MAT | tee ./$MAT/SLU.o_mpi_${NROW}x${NCOL}x${NPZ}_omp${OMP_NUM_THREADS}_gpusolve_batchcount${batch}


done

done
done
done
