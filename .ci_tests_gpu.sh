#!/bin/sh
set -e

export RED="\033[31;1m"
export BLUE="\033[34;1m"
export ROOT_DIR="$PWD"
printf "${BLUE} SLU; Entered tests file:\n"
export DATA_FOLDER=$ROOT_DIR/EXAMPLE
export EXAMPLE_FOLDER=$ROOT_DIR/build/EXAMPLE


export OMP_NUM_THREADS=1


case "${TEST_NUMBER}" in
1)  # Test SLU.(1): batched SpLU/SpTRSV (for real matrices) for solving 16 linear systems with the same sparsity pattern ("-b 16") 
    echo -e "\nSuperLU_DIST: Testing batched SpLU/SpTRSV (for real matrices)."
    export GPU3DVERSION=1 # value 1 uses magma for operations in the batched solver
    export SUPERLU_ACC_OFFLOAD=1 # # you can turn GPU SpLU ON (1) and OFF (0) 
    export SUPERLU_ACC_SOLVE=1 # you can turn GPU SpTRSV ON (1) and OFF (0) 
    mat=$DATA_FOLDER/big.rua # 
    mpirun -np 1 ${EXAMPLE_FOLDER}/pddrive3d -r 1 -c 1 -d 1 -b 16 -i 0 -s 1 $mat ;;
2)  # Test SLU.(2): 2D SpLU/SpTRSV (for real matrices) with 1x1 MPIs.   
    echo -e "\nSuperLU_DIST: Testing 2D SpLU/SpTRSV (for real matrices) with 1x1 MPIs."
    export SUPERLU_ACC_OFFLOAD=1 # # you can turn GPU SpLU ON (1) and OFF (0) 
    export SUPERLU_ACC_SOLVE=1 # you can turn GPU SpTRSV ON (1) and OFF (0) 
    rowperm=1 # row permutation option: 0: NOROWPERM, 1: LargeDiag_MC64, 2: LargeDiag_HWPM
    colperm=2 # column permutation: 0: NATURAL, 1: MMD_ATA, 2: MMD_AT_PLUS_A, 3: COLAMD, 4: METIS_AT_PLUS_A, 5: PARMETIS, 6: METIS_ATA
    mat=$DATA_FOLDER/big.rua # 
    mpirun -np 1 ${EXAMPLE_FOLDER}/pddrive -r 1 -c 1 -i 0 -p $rowperm -q $colperm $mat ;;
3)  # Test SLU.(3): 3D SpLU/SpTRSV (for real matrices) with 1x1x2 MPIs/GPUs.    
    echo -e "\nSuperLU_DIST: Testing 3D SpLU/SpTRSV (for real matrices) with 1x1x2 MPIs"
    export SUPERLU_ACC_OFFLOAD=1 # you can turn GPU SpLU ON (1) and OFF (0) 
    export GPU3DVERSION=0 # value 1 uses the latest c++ factorization code
    export SUPERLU_ACC_SOLVE=1 # you can turn GPU SpTRSV ON (1) and OFF (0) 
    rowperm=1 # row permutation option: 0: NOROWPERM, 1: LargeDiag_MC64, 2: LargeDiag_HWPM
    colperm=2 # column permutation: 0: NATURAL, 1: MMD_ATA, 2: MMD_AT_PLUS_A, 3: COLAMD, 4: METIS_AT_PLUS_A, 5: PARMETIS, 6: METIS_ATA
    mat=$DATA_FOLDER/big.rua # 
    mpirun -np 2 ${EXAMPLE_FOLDER}/pddrive3d -r 1 -c 1 -d 2 -i 1 -p $rowperm -q $colperm $mat ;; 
4)  # Test SLU.(3): 3D SpLU/SpTRSV (for real matrices) with 1x1x2 MPIs/GPUs.    
    echo -e "\nSuperLU_DIST: Testing 3D SpLU/SpTRSV (for real matrices) with 1x1x2 MPIs"
    export SUPERLU_ACC_OFFLOAD=1 # you can turn GPU SpLU ON (1) and OFF (0) 
    export GPU3DVERSION=1 # value 1 uses the latest c++ factorization code
    export SUPERLU_ACC_SOLVE=1 # you can turn GPU SpTRSV ON (1) and OFF (0) 
    export MPICH_GPU_SUPPORT_ENABLED=1 # GPU-aware MPI is needed for the C++ factorization code 
    rowperm=1 # row permutation option: 0: NOROWPERM, 1: LargeDiag_MC64, 2: LargeDiag_HWPM
    colperm=4 # column permutation: 0: NATURAL, 1: MMD_ATA, 2: MMD_AT_PLUS_A, 3: COLAMD, 4: METIS_AT_PLUS_A, 5: PARMETIS, 6: METIS_ATA
    mat=$DATA_FOLDER/big.rua # 
    mpirun -np 2 ${EXAMPLE_FOLDER}/pddrive3d -r 1 -c 1 -d 2 -i 1 -p $rowperm -q $colperm $mat ;;     
*) printf "${RED} ###YL: Unknown test\n" ;;
esac