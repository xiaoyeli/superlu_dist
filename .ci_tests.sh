#!/bin/bash
# CPU tests of SuperLU_DIST, run by .github/workflows/test.yml.
#
#   ./.ci_tests.sh                 run every test, print a summary
#   ./.ci_tests.sh list            list the tests with their numbers
#   ./.ci_tests.sh 17              run test 17        (also: TEST_NUMBER=17)
#   ./.ci_tests.sh "3 20-25"       run a selection
#   ./.ci_tests.sh "2d ilu"        run groups: legacy 2d 3d solveonly ilu batch examples
#
# The exit status is 0 only if no test failed.  See TEST/ci_common.sh for the
# format of a test line and for the environment knobs (build directory,
# MPI launcher, timeout), and the header of TEST/ptest_ci.c for the options
# of the self-checking drivers p{s,d,z}test_ci used by most tests.

export ROOT_DIR="$PWD"
. "$ROOT_DIR/TEST/ci_common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Generated matrices: a 2D and a 3D grid problem, small enough for a CI runner.
G2="-x 40 -y 30"            # n = 1200
G3="-x 12 -y 11 -z 10"      # n = 1320
GB="-x 8 -y 7 -z 6"         # with -U: that many unknowns per grid point

# ---------------------------------------------------------------------------
# 1-15: the original tests (pdtest and the 2D EXAMPLE drivers).  Their numbers
# are kept; they are now checked for accuracy instead of exit status only.
# ---------------------------------------------------------------------------
ci_group legacy
add_test pass pdtest 1 "" $T/pdtest -r 1 -c 1 -s 1 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 1 "" $T/pdtest -r 1 -c 1 -s 3 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 3 "" $T/pdtest -r 1 -c 3 -s 1 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 3 "" $T/pdtest -r 1 -c 3 -s 3 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 2 "" $T/pdtest -r 2 -c 1 -s 1 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 2 "" $T/pdtest -r 2 -c 1 -s 3 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 6 "" $T/pdtest -r 2 -c 3 -s 1 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass pdtest 6 "" $T/pdtest -r 2 -c 3 -s 3 -b 2 -x 8 -m 20 -f $D/g20.rua
add_test pass xtrue  4 "" $E/pddrive1 -r 2 -c 2 $D/big.rua
add_test pass xtrue  4 "" $E/pddrive2 -r 2 -c 2 $D/big.rua
add_test pass xtrue  4 "" $E/pddrive3 -r 2 -c 2 $D/big.rua
add_test pass xtrue  4 "" $E/pzdrive1 -r 2 -c 2 $D/cg20.cua
add_test pass xtrue  4 "" $E/pzdrive2 -r 2 -c 2 $D/cg20.cua
add_test pass xtrue  4 "" $E/pzdrive3 -r 2 -c 2 $D/cg20.cua
add_test pass xtrue  4 "" $E/pddrive_ABglobal -r 2 -c 2 $D/big.rua

# ---------------------------------------------------------------------------
# 2D interface p?gssvx: process grids, precisions, factorization modes.
# -f DFPRF = DOFACT, FACTORED, SamePattern, SamePattern_SameRowPerm, FACTORED,
# each call with new matrix values and/or a new right-hand side.
# ---------------------------------------------------------------------------
ci_group 2d
add_test pass ci 1 "" $T/pdtest_ci -r 1 -c 1 -f DFPRF $G2
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -f DFPRF -s 3 $G3
add_test pass ci 3 "" $T/pdtest_ci -r 1 -c 3 -f DFPRF $G3
add_test pass ci 3 "" $T/pdtest_ci -r 3 -c 1 -f DFPRF $G3
add_test pass ci 6 "" $T/pdtest_ci -r 2 -c 3 -f DFPRF $G3
add_test pass ci 6 "" $T/pdtest_ci -r 3 -c 2 -f DFPRFD $D/big.rua
add_test pass ci 4 "" $T/pztest_ci -r 2 -c 2 -f DFPRF -s 2 $G3
add_test pass ci 6 "" $T/pztest_ci -r 3 -c 2 -f DFPRF $D/cg20.cua
add_test pass ci 4 "" $T/pstest_ci -r 2 -c 2 -f DFPRF $G3
add_test pass ci 2 "" $T/pstest_ci -r 1 -c 2 -f DFPRF $D/g20.rua

# Column orderings (-q), 2D: NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
# METIS_AT_PLUS_A, PARMETIS without and with parallel symbolic factorization.
# (METIS_ATA, -q 6, is declared in colperm_t but not implemented.)
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 0 $G2
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 1 $G2
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 2 $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 3 $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 4 -f DFPRF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 5 -f DFPRF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -q 5 -P 1 -f DFPRF $G3
add_test pass  ci 4 "" $T/pztest_ci -r 2 -c 2 -q 5 -P 1 -f DFPRF $D/cg20.cua
# Parallel symbolic factorization of a NATURAL ordering (no separator tree)
add_test pass  ci 6 "" $T/pdtest_ci -r 3 -c 2 -q 0 -P 1 -f DFPRF $G3

# Row permutation (-p), equilibration (-e), refinement (-i), DiagInv (-D),
# GMRES (-G).  "-k 1" scrambles the rows: only a row permutation can solve it.
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -p 0 -e 0 $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -p 1 -k 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pztest_ci -r 2 -c 2 -p 1 -k 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -p 2 -k 1 $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -i 2 -s 2 -f DFPRF $D/big.rua
add_test pass ci 4 "" $T/pstest_ci -r 2 -c 2 -i 1 -f DF $D/big.rua
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -D 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pztest_ci -r 2 -c 2 -D 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -G 1 -f DF $G3

# ---------------------------------------------------------------------------
# 3D interface p?gssvx3d: process grids nprow x npcol x npdep.
# ---------------------------------------------------------------------------
ci_group 3d
add_test pass ci 1 "" $T/pdtest_ci -r 1 -c 1 -d 1 -f DFPRF $G3
add_test pass ci 2 "" $T/pdtest_ci -r 1 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 1 -c 1 -d 4 -f DFPRF -s 3 $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 1 -c 2 -d 2 -f DFPRF $D/big.rua
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -d 1 -f DFPRF $G3
add_test pass ci 8 "" $T/pdtest_ci -r 2 -c 2 -d 2 -f DFPRF $G3
add_test pass ci 6 "" $T/pdtest_ci -r 3 -c 2 -d 1 -f DFPRF $D/big.rua
add_test pass ci 4 "" $T/pztest_ci -r 2 -c 1 -d 2 -f DFPRF -s 2 $G3
add_test pass ci 4 "" $T/pztest_ci -r 1 -c 2 -d 2 -f DFPRF $D/cg20.cua
add_test pass ci 4 "" $T/pstest_ci -r 2 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 4 "SUPERLU_RANKORDER=XY" $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3

# 3D: orderings, row permutation, refinement, DiagInv.
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -q 2 $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -q 4 $G3
# ParMETIS in the 3D interface (double precision only): serial symbolic
# factorization, then the parallel one (-P 1), which supports a fresh
# factorization and solves with it (D, F), not the reuse modes (P, R).
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -d 1 -q 5 -f DFPRF $G3
add_test pass  ci 8 "" $T/pdtest_ci -r 2 -c 2 -d 2 -q 5 -f DFPRF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 2 -d 1 -q 5 -P 1 -f DFDF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -q 5 -P 1 -f DFDF -s 3 $D/big.rua
add_test pass  ci 8 "" $T/pdtest_ci -r 2 -c 2 -d 2 -q 5 -P 1 -p 1 -k 1 -f DF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 1 -c 1 -d 4 -q 5 -P 1 -f DF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -q 0 -P 1 -f DF $D/big.rua
add_test pass  ci 4 "GPU3DVERSION=1" $T/pdtest_ci -r 2 -c 1 -d 2 -q 5 -P 1 -f DF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -p 1 -k 1 -f DFPRF $G3
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -i 2 -f DFPRF $D/big.rua
add_test pass  ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -D 1 -f DFPRF $G3
add_test pass  ci 4 "" $T/pztest_ci -r 1 -c 2 -d 2 -D 1 -f DFPRF $G3

# 3D with the C++ factorization (GPU3DVERSION=1), here on the CPU.
add_test pass ci 1 "GPU3DVERSION=1" $T/pdtest_ci -r 1 -c 1 -d 1 -f DFPRF $G3
add_test pass ci 4 "GPU3DVERSION=1" $T/pdtest_ci -r 2 -c 1 -d 2 -f DFPRF $G3
add_test pass ci 4 "GPU3DVERSION=1" $T/pdtest_ci -r 1 -c 1 -d 4 -f DFPRF $D/big.rua
add_test pass ci 4 "GPU3DVERSION=1" $T/pztest_ci -r 1 -c 2 -d 2 -f DFPRF $G3
add_test pass ci 4 "GPU3DVERSION=1" $T/pstest_ci -r 2 -c 1 -d 2 -f DFPRF $G3

# ---------------------------------------------------------------------------
# SolveOnly (3D interface): the matrix holds the L and U factors.
# ---------------------------------------------------------------------------
ci_group solveonly
add_test pass ci 1 "" $T/pdtest_ci -r 1 -c 1 -d 1 -S 1 -f DF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -S 1 -f DF -s 3 $G3
add_test pass ci 4 "" $T/pztest_ci -r 2 -c 2 -d 1 -S 1 -f DF $G3
add_test pass ci 4 "" $T/pstest_ci -r 1 -c 2 -d 2 -S 1 -f DF $G2

# ---------------------------------------------------------------------------
# ILU(0): the solution must be the one of a textbook ILU(0) of the permuted
# and scaled matrix, with the natural and with fill-reducing orderings.
# With GMRES (2D) the incomplete factors precondition a solve to full accuracy.
# ---------------------------------------------------------------------------
ci_group ilu
add_test pass ci 1 "" $T/pdtest_ci -r 1 -c 1 -I 0 -q 0 -f DFPRF $G2
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -I 0 -q 0 -f DFPRF $G3
add_test pass ci 6 "" $T/pztest_ci -r 2 -c 3 -I 0 -q 0 -k 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pstest_ci -r 2 -c 2 -I 0 -q 0 $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -I 0 -q 0 -G 1 -f DF $G3
add_test pass ci 1 "" $T/pdtest_ci -r 1 -c 1 -d 1 -I 0 -q 0 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -I 0 -q 0 -f DFPRF $G3
add_test pass ci 4 "" $T/pztest_ci -r 1 -c 2 -d 2 -I 0 -q 0 -k 1 -f DFPRF $G3
add_test pass ci 4 "" $T/pdtest_ci -r 1 -c 1 -d 4 -I 0 -q 0 -s 2 $G2
# fill-reducing orderings: MMD_AT_PLUS_A, METIS_AT_PLUS_A, ParMETIS
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -I 0 -q 2 -f DFPRF $G3
add_test pass ci 6 "" $T/pztest_ci -r 2 -c 3 -I 0 -q 4 -f DFPRF $D/cg20.cua
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -I 0 -q 5 -f DF $G3
add_test pass ci 2 "" $T/pdtest_ci -r 2 -c 1 -I 0 -q 1 $D/big.rua
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -I 0 -q 4 -f DFPRF $G3
add_test pass ci 4 "" $T/pstest_ci -r 1 -c 2 -d 2 -I 0 -q 2 -f DF $G3
# the C++ factorization (GPU3DVERSION=1)
add_test pass ci 1 "GPU3DVERSION=1" $T/pdtest_ci -r 1 -c 1 -d 1 -I 0 -q 0 -f DFPRF $G3
add_test pass ci 4 "GPU3DVERSION=1" $T/pdtest_ci -r 2 -c 1 -d 2 -I 0 -q 4 -f DF $G3
add_test pass ci 4 "GPU3DVERSION=1" $T/pztest_ci -r 2 -c 2 -d 1 -I 0 -q 2 -f DF $G3
# user-defined supernodes of several columns (options.UserDefineSupernode):
# -U gives each grid point that many unknowns coupled by dense blocks, one
# supernode per point; natural and red-black (-q 8, MY_PERMC) orderings
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 2 -I 0 -U 4 -p 0 -q 8 -f DFPRF $GB
add_test pass ci 6 "" $T/pztest_ci -r 2 -c 3 -I 0 -U 3 -p 0 -q 0 -f DFPF $GB
add_test pass ci 1 "" $T/pstest_ci -r 1 -c 1 -I 0 -U 2 -p 0 -q 8 $GB
add_test pass ci 4 "" $T/pdtest_ci -r 2 -c 1 -d 2 -I 0 -U 4 -p 0 -q 8 -f DFPRF $GB
add_test pass ci 4 "" $T/pztest_ci -r 1 -c 2 -d 2 -I 0 -U 3 -p 0 -q 8 -s 2 $GB
add_test pass ci 4 "GPU3DVERSION=1" $T/pdtest_ci -r 2 -c 1 -d 2 -I 0 -U 4 -p 0 -q 8 -f DF $GB

# ---------------------------------------------------------------------------
# Batched interfaces (one process): -b fixed size, -B variable size with the
# factorization state reused by SamePattern_SameRowPerm calls.
# ---------------------------------------------------------------------------
ci_group batch
add_test pass ci 1 "" $T/pdtest_ci -b 4 $G3
add_test pass ci 1 "" $T/pztest_ci -b 3 -s 2 $G3
add_test pass ci 1 "" $T/pstest_ci -b 3 $D/big.rua
add_test pass ci 1 "" $T/pdtest_ci -B 4 -f DRR $G3
add_test pass ci 1 "" $T/pztest_ci -B 3 -f DR $D/cg20.cua
add_test pass ci 1 "" $T/pstest_ci -B 3 -f DR $G2

# ---------------------------------------------------------------------------
# The 3D EXAMPLE drivers.
# ---------------------------------------------------------------------------
ci_group examples
add_test pass  xtrue 4 "" $E/pddrive3d  -r 2 -c 1 -d 2 $D/big.rua
add_test pass  xtrue 4 "" $E/pddrive3d1 -r 2 -c 1 -d 2 $D/big.rua
add_test pass  xtrue 4 "" $E/pddrive3d2 -r 2 -c 1 -d 2 $D/big.rua
# (pddrive3d3, SamePattern_SameRowPerm, is not run: see the group 3d.)
add_test pass  xtrue 4 "" $E/pzdrive3d  -r 1 -c 2 -d 2 $D/cg20.cua
add_test pass  xtrue_s 4 "" $E/psdrive3d -r 2 -c 1 -d 2 $D/big.rua

ci_main "${1:-${TEST_NUMBER:-all}}"
