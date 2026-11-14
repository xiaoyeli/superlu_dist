#!/bin/bash
# GPU tests of SuperLU_DIST, run by .gitlab-ci.yml on a GPU runner.
#
#   ./.ci_tests_gpu.sh             run every test, print a summary
#   ./.ci_tests_gpu.sh list        list the tests with their numbers
#   ./.ci_tests_gpu.sh 7           run test 7         (also: TEST_NUMBER=7)
#   ./.ci_tests_gpu.sh "5-20"      run a selection
#   ./.ci_tests_gpu.sh "2d batch"  run groups: legacy 2d 3d 3dcpp features batch
#
# The exit status is 0 only if no test failed.  See TEST/ci_common.sh for the
# format of a test line and for the environment knobs (build directory,
# MPI launcher, timeout), and the header of TEST/ptest_ci.c for the options
# of the self-checking drivers p{s,d,z}test_ci used by most tests.
#
# The GPU code paths are selected with environment variables:
#   SUPERLU_ACC_OFFLOAD=1   numerical factorization on the GPU
#   SUPERLU_ACC_SOLVE=1     triangular solves on the GPU
#   SUPERLU_ACC_SOLVE=2     the same, with the GEMM kernels when there are
#                           several right-hand sides
#   GPU3DVERSION=0|1        3D factorization: C code with GEMM offload (0),
#                           or the C++ code with the whole panel work on the
#                           GPU (1); 1 also runs the batched factorization
# What the build or the machine cannot run is skipped, not failed:
#   - a GPU triangular solve on a 2D grid of more than one process needs
#     NVSHMEM (ROCSHMEM); the driver reports SKIP without it;
#   - GPU3DVERSION=1 on more than one process needs a GPU-aware MPI: those
#     tests are only run with SLU_CI_GPU_AWARE_MPI=1;
#   - the batched interface with GPU3DVERSION=1 needs MAGMA.
# With a GPU-aware MPI or NVSHMEM, every process must see the GPUs of the
# processes it talks to: do not confine each MPI rank to its own GPU (e.g.
# "srun --gpus-per-task=1"); SUPERLU_BIND_MPI_GPU=1 spreads the ranks instead.

export ROOT_DIR="$PWD"
. "$ROOT_DIR/TEST/ci_common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# one GPU per MPI rank when there are several GPUs (ranks share them otherwise)
export SUPERLU_BIND_MPI_GPU="${SUPERLU_BIND_MPI_GPU:-1}"
SLU_CI_GPU_AWARE_MPI="${SLU_CI_GPU_AWARE_MPI:-0}"

# Generated 3D grid problems; the larger one gives supernodes big enough for
# the GEMM offload of the C code.
G3="-x 12 -y 11 -z 10"      # n = 1320
GB="-x 8 -y 7 -z 6"         # with -U: that many unknowns per grid point
G3L="-x 24 -y 22 -z 20"     # n = 10560

FS="SUPERLU_ACC_OFFLOAD=1 SUPERLU_ACC_SOLVE=1"   # GPU factorization and solve
FS2="SUPERLU_ACC_OFFLOAD=1 SUPERLU_ACC_SOLVE=2"  # the same, GEMM solve kernels
F="SUPERLU_ACC_OFFLOAD=1 SUPERLU_ACC_SOLVE=0"    # GPU factorization, CPU solve
S="SUPERLU_ACC_OFFLOAD=0 SUPERLU_ACC_SOLVE=1"    # CPU factorization, GPU solve
V0="GPU3DVERSION=0 SUPERLU_N_GEMM=64"            # C code, low offload threshold
V1="GPU3DVERSION=1"                              # C++ code

# Tests that need a GPU-aware MPI: listed, but skipped without one.
add_gpu_mpi_test() {
    local expect="$1" check="$2" np="$3" envs="$4"; shift 4
    [ "$SLU_CI_GPU_AWARE_MPI" = 1 ] || expect=skip
    add_test "$expect" "$check" "$np" "$envs MPICH_GPU_SUPPORT_ENABLED=1" "$@"
}

# ---------------------------------------------------------------------------
# 1-3: the original tests (EXAMPLE drivers), now checked for accuracy.  The
# original test 4 (C++ code on 1x1x2) is the last test of the group 3dcpp.
# ---------------------------------------------------------------------------
ci_group legacy
# batched SpLU/SpTRSV (MAGMA) of 16 systems with the same sparsity pattern
add_test pass berr  1 "$FS $V1" $E/pddrive3d -r 1 -c 1 -d 1 -b 16 -i 0 -s 1 $D/big.rua
# 2D SpLU/SpTRSV, 1x1
add_test pass xtrue 1 "$FS" $E/pddrive -r 1 -c 1 -i 0 -p 1 -q 2 $D/big.rua
# 3D SpLU/SpTRSV, 1x1x2, C code
add_test pass xtrue 2 "$FS GPU3DVERSION=0" $E/pddrive3d -r 1 -c 1 -d 2 -i 1 -p 1 -q 2 $D/big.rua

# ---------------------------------------------------------------------------
# 2D interface p?gssvx.  -f DFPRF = DOFACT, FACTORED, SamePattern,
# SamePattern_SameRowPerm, FACTORED, each with new values / a new RHS.
# ---------------------------------------------------------------------------
ci_group 2d
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -f DFPRF $G3L
add_test pass ci 1 "$FS $V0" $T/pztest_ci -r 1 -c 1 -f DFPRF -s 3 $G3
add_test pass ci 1 "$FS $V0" $T/pstest_ci -r 1 -c 1 -f DFPRF $G3
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -f DFPRFD -i 2 $D/big.rua
add_test pass ci 1 "$FS $V0" $T/pztest_ci -r 1 -c 1 -f DFPRF $D/cg20.cua
add_test pass ci 1 "$S"      $T/pdtest_ci -r 1 -c 1 -f DFPRF $G3
add_test pass ci 4 "$F $V0"  $T/pdtest_ci -r 2 -c 2 -f DFPRF $G3L
add_test pass ci 4 "$F $V0"  $T/pztest_ci -r 2 -c 2 -f DFPRF $G3
add_test pass ci 2 "$F $V0"  $T/pstest_ci -r 1 -c 2 -f DFPRF $G3
add_test pass ci 4 "$FS $V0" $T/pdtest_ci -r 2 -c 2 -f DFPRF $G3L        # NVSHMEM
add_test pass ci 2 "$FS $V0" $T/pztest_ci -r 2 -c 1 -f DFPRF -s 2 $G3    # NVSHMEM
# Several right-hand sides across process columns (reduction trees with one
# and with two children)
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 2 -s 2 $G3             # NVSHMEM
add_test pass ci 4 "$FS $V0" $T/pdtest_ci -r 1 -c 4 -s 4 -f DFPRF $G3    # NVSHMEM
add_test pass ci 3 "$FS $V0" $T/pztest_ci -r 1 -c 3 -s 3 -f DF $G3       # NVSHMEM
# SUPERLU_ACC_SOLVE=2: the GEMM kernels for several right-hand sides
add_test pass ci 1 "$FS2 $V0" $T/pdtest_ci -r 1 -c 1 -s 5 -f DF $G3
add_test pass ci 4 "$FS2 $V0" $T/pdtest_ci -r 2 -c 2 -s 3 -f DFPRF $G3    # NVSHMEM
add_test pass ci 4 "$FS2 $V0" $T/pztest_ci -r 2 -c 2 -s 4 -f DF $G3       # NVSHMEM
add_test pass ci 2 "$FS2 $V0" $T/pstest_ci -r 1 -c 2 -s 2 -f DF $G3       # NVSHMEM
add_test pass ci 4 "$FS2 $V0" $T/pdtest_ci -r 2 -c 1 -d 2 -s 3 -f DF $G3  # NVSHMEM
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -q 2 -D 1 -f DFPRF $G3
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -q 0 -p 1 -k 1 $G3
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -g 1 -f DF $G3       # RHS on the GPU

# ---------------------------------------------------------------------------
# 3D interface p?gssvx3d, C code (GPU3DVERSION=0)
# ---------------------------------------------------------------------------
ci_group 3d
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 1 -f DFPRF $G3L
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 2 -f DFPRF $G3L
add_test pass ci 4 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 4 -f DFPRF -s 3 $G3
add_test pass ci 2 "$FS $V0" $T/pztest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 2 "$FS $V0" $T/pstest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 2 -f DFPRF -i 2 $D/big.rua
add_test pass ci 2 "$FS $V0" $T/pztest_ci -r 1 -c 1 -d 2 -f DFPRF $D/cg20.cua
add_test pass ci 4 "$F $V0"  $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3L
add_test pass ci 4 "$F $V0"  $T/pztest_ci -r 1 -c 2 -d 2 -f DFPRF $G3
add_test pass ci 2 "$S $V0"  $T/pdtest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 4 "$FS $V0" $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3L    # NVSHMEM
add_test pass ci 4 "$FS $V0" $T/pztest_ci -r 2 -c 2 -d 1 -f DFPRF $G3     # NVSHMEM
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 2 -q 2 -D 1 -f DFPRF $G3
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 2 -g 1 -f DF $G3  # RHS on the GPU
# ParMETIS ordering with the parallel symbolic factorization (double only;
# fresh factorization and solves)
add_test pass ci 2 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 2 -q 5 -P 1 -f DFDF $G3L
add_test pass ci 4 "$FS $V0" $T/pdtest_ci -r 2 -c 1 -d 2 -q 5 -P 1 -f DF $G3L  # NVSHMEM

# ---------------------------------------------------------------------------
# 3D interface, C++ code (GPU3DVERSION=1)
# ---------------------------------------------------------------------------
ci_group 3dcpp
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 1 -f DFPRF $G3L
add_test pass ci 1 "$FS $V1" $T/pztest_ci -r 1 -c 1 -d 1 -f DFPRF -s 2 $G3
add_test pass ci 1 "$FS $V1" $T/pstest_ci -r 1 -c 1 -d 1 -f DFPRF $G3
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 1 -f DFPRF -i 2 $D/big.rua
add_test pass ci 1 "$F $V1"  $T/pztest_ci -r 1 -c 1 -d 1 -f DFPRF $D/cg20.cua
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 2 -f DFPRF $G3L
add_gpu_mpi_test pass ci 4 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 4 -f DFPRF $G3
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pztest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pstest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_gpu_mpi_test pass ci 4 "$F $V1"  $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3L
add_gpu_mpi_test pass ci 4 "$FS $V1" $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3L   # NVSHMEM
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 2 -q 5 -P 1 -f DFDF $G3L
add_gpu_mpi_test pass ci 4 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 4 -q 5 -P 1 -f DF $D/big.rua
add_gpu_mpi_test pass xtrue 2 "$FS $V1" $E/pddrive3d -r 1 -c 1 -d 2 -i 1 -p 1 -q 4 $D/big.rua

# ---------------------------------------------------------------------------
# SolveOnly and ILU(0) with the GPU solve
# ---------------------------------------------------------------------------
ci_group features
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 1 -S 1 -f DF -s 2 $G3
add_test pass ci 2 "$FS $V0" $T/pztest_ci -r 1 -c 1 -d 2 -S 1 -f DF $G3
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -I 0 -q 0 -f DF $G3
add_test pass ci 1 "$FS $V0" $T/pdtest_ci -r 1 -c 1 -d 1 -I 0 -q 0 -f DF $G3
add_test pass ci 2 "$FS $V0" $T/pztest_ci -r 1 -c 1 -d 2 -I 0 -q 0 -f DF $G3
add_test pass ci 4 "$F $V0"  $T/pdtest_ci -r 2 -c 2 -I 0 -q 2 -f DFPRF $G3
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 1 -I 0 -q 0 -f DF $G3
add_test pass ci 1 "$FS $V1" $T/pztest_ci -r 1 -c 1 -d 1 -I 0 -q 4 -f DF $G3
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 2 -I 0 -q 4 -f DF $G3
# ILU(0) with user-defined supernodes of several columns (-U), red-black ordering
add_test pass ci 4 "$F $V0"  $T/pdtest_ci -r 2 -c 2 -I 0 -U 4 -p 0 -q 8 -f DFPF $GB
add_test pass ci 1 "$FS $V0" $T/pztest_ci -r 1 -c 1 -d 1 -I 0 -U 3 -p 0 -q 8 -f DF $GB
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -r 1 -c 1 -d 1 -I 0 -U 4 -p 0 -q 8 -f DF $GB
add_gpu_mpi_test pass ci 2 "$FS $V1" $T/pztest_ci -r 1 -c 1 -d 2 -I 0 -U 3 -p 0 -q 8 -f DF $GB

# ---------------------------------------------------------------------------
# Batched interfaces (MAGMA): fixed size (-b), variable size with reuse of
# the factorization state (-B)
# ---------------------------------------------------------------------------
ci_group batch
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -b 8 -s 2 $G3
add_test pass ci 1 "$FS $V1" $T/pztest_ci -b 4 $G3
add_test pass ci 1 "$FS $V1" $T/pstest_ci -b 4 $G3
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -b 4 $D/big.rua
add_test pass ci 1 "$FS $V1" $T/pdtest_ci -B 4 -f DRR $G3
add_test pass ci 1 "$FS $V1" $T/pztest_ci -B 3 -f DR $G3
add_test pass ci 1 "$FS $V1" $T/pstest_ci -B 3 -f DR $G3

ci_main "${1:-${TEST_NUMBER:-all}}"
