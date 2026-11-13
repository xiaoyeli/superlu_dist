include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

# Libs linked to all of the tests
set(all_link_libs superlu_dist ${BLAS_LIB})
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()

set(MATRICES ../EXAMPLE/g20.rua)  # sample sparse matrix from a file
set(NPROWS 1 2 5)	  # process rows
set(NPCOLS 1 2 3) 	  # process columns
set(NVAL 9 19)	  	  # generated matrix dimensions
set(NRHS 1 3)		  # number of RHS
# set(FILLRATIO 2 10)	  # estimated fill ratio
set(FILLRATIO 2)	  # estimated fill ratio
# following are blocking parameters, see sp_ienv.c
set(RELAX 8)	   	  # relaxed supernode size: 4 8
set(SUPERSIZE 20)   	  # maximum supernode size: 10 20
set(MINGEMM 10000)	  # minimum GEMM size for GPU offload

function(cat IN_FILE OUT_FILE)
  file(READ ${IN_FILE} CONTENTS)
  file(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()

set(MPIEXEC_PREFLAGS "" CACHE STRING "Extra flags to set after mpiexec")

# Function to perform test
# call API:  add_superlu_dist_tests(pddrive big.rua)
function(add_superlu_dist_tests target input)
   set(TEST_INPUT "${SuperLU_DIST_SOURCE_DIR}/EXAMPLE/${input}")
   set(TEST_OUTPUT "${SuperLU_DIST_BINARY_DIR}/TEST/${target}.out")

  # Prepare a temporary file to "cat" to:
  # file(WRITE ${TEST_OUTPUT} "")

##  get_target_property(TEST_LOC ${target} LOCATION)
   set(TEST_LOC ${CMAKE_CURRENT_BINARY_DIR})

   foreach (r ${NPROWS})
      foreach (c ${NPCOLS})
        MATH( EXPR np "${r}*${c}" )
        foreach (s ${NRHS})
	  foreach (b ${FILLRATIO})
	    foreach (x ${RELAX})
	      foreach (m ${SUPERSIZE})
                set(testName "${target}_${r}x${c}_${s}_${b}_${x}_${m}")
	  	set(SINGLE_OUTPUT ${SuperLU_DIST_BINARY_DIR}/TEST/${testName}.out)
          add_test( NAME ${testName}_SP
	    	    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
            	    ${MPIEXEC_PREFLAGS} ${TEST_LOC}/${target} ${MPIEXEC_POSTFLAGS}
		    -r ${r} -c ${c} -s ${s} -b ${b} -x ${x} -m ${m} -f ${TEST_INPUT}
		  )
#          add_test( ${testName}_SP "${CMAKE_COMMAND}"
#	    -DTEST=${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
#            ${MPIEXEC_PREFLAGS} ${target} ${MPIEXEC_POSTFLAGS} -r ${r} -c ${c} -s ${s} -b ${b} -x ${x} -m ${m} -f ${TEST_INPUT}
#	    -DOUTPUT=${SINGLE_OUTPUT}
#	    -DALL_OUTPUT=${TEST_OUTPUT}
#	    -DHEADING=Sparse\ matrix\ ${TEST_INPUT}\ --\ r=${r},\ c=${c},\ s=${s},\ x=${x},\ m=${m}
#	    -P "${SuperLU_DIST_SOURCE_DIR}/TEST/runtest.cmake"
#		  )
	      endforeach()
	    endforeach()
	  endforeach()
	endforeach()
      endforeach()
   endforeach()

# MPI variables:
# ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} PROCS
#  	${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS)

endfunction(add_superlu_dist_tests)

if(enable_double)
  set(DTEST pdtest.c dcreate_matrix.c pdcompute_resid.c)
  add_executable(pdtest ${DTEST})
  target_link_libraries(pdtest ${all_link_libs})
  target_compile_features(pdtest PUBLIC c_std_99)
  add_superlu_dist_tests(pdtest g20.rua)
  
  set(TESTsingular singular.c)
  add_executable(singular ${TESTsingular})
  target_link_libraries(singular ${all_link_libs})
  target_compile_features(singular PUBLIC c_std_99)
  add_test( NAME singular_zero-row-2d
	    	    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            	    ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/singular zero_row 2d ) # or abglobal, or 3d
  set(SINGLE_OUTPUT ${SuperLU_DIST_BINARY_DIR}/TEST/singular_zero-row-2d.out)
  add_test( NAME singular_structural-2d
	    	    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            	    ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/singular structural 2d ) # also: ok, numeric, zero_colcnd
endif()

# pztest.c (complex counterpart of pdtest.c) compiles but does not terminate,
# even on one process, so it is neither built nor registered; the complex
# drivers are tested through pztest_ci below.
#if(enable_complex16)
#  set(ZTEST pztest.c zcreate_matrix.c pzcompute_resid.c)
#endif()

######################################################################
# Self-checking driver shared by ctest and the CI scripts (.ci_tests.sh,
# .ci_tests_gpu.sh): one source, compiled once per enabled precision.
# It verifies each solution and returns 0 (pass), 1 (fail) or 77 (the
# build cannot run the configuration, e.g. a ParMETIS ordering without
# ParMETIS).  See the header of ptest_ci.c for its options.
######################################################################
function(add_superlu_dist_ci_driver target precision_macro)
  add_executable(${target} ptest_ci.c)
  target_compile_definitions(${target} PRIVATE ${precision_macro})
  target_link_libraries(${target} ${all_link_libs})
  target_compile_features(${target} PUBLIC c_std_99)
endfunction()

set(CI_PRECISIONS "")
if(enable_single)
  add_superlu_dist_ci_driver(pstest_ci SLU_CI_SINGLE)
  list(APPEND CI_PRECISIONS s)
endif()
if(enable_double)
  add_superlu_dist_ci_driver(pdtest_ci SLU_CI_DOUBLE)
  list(APPEND CI_PRECISIONS d)
endif()
if(enable_complex16)
  add_superlu_dist_ci_driver(pztest_ci SLU_CI_COMPLEX16)
  list(APPEND CI_PRECISIONS z)
endif()

separate_arguments(CI_MPIEXEC_PREFLAGS UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
separate_arguments(CI_MPIEXEC_POSTFLAGS UNIX_COMMAND "${MPIEXEC_POSTFLAGS}")

# add_superlu_dist_ci_test(<name> <precision s|d|z> <np> <driver args...>
#                          [ENV <VAR=value>...] [LABELS <label>...] [XFAIL])
#   ENV     environment of the run; on top of the defaults below, which pin
#           the CPU code so that a GPU build does not silently offload
#   LABELS  ctest labels, select with "ctest -L <label>"
#   XFAIL   known library defect: the test is expected to fail
# The test is named ci_<precision>_<name>; it is not added if that precision
# is not built.
function(add_superlu_dist_ci_test name prec np)
  cmake_parse_arguments(T "XFAIL" "" "ENV;LABELS" ${ARGN})
  if(NOT TARGET p${prec}test_ci)
    return()
  endif()
  set(test_name ci_${prec}_${name})
  add_test(NAME ${test_name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${CI_MPIEXEC_PREFLAGS} $<TARGET_FILE:p${prec}test_ci>
                   ${CI_MPIEXEC_POSTFLAGS} ${T_UNPARSED_ARGUMENTS})
  set_tests_properties(${test_name} PROPERTIES
    SKIP_RETURN_CODE 77
    SKIP_REGULAR_EXPRESSION "SLU_CI_RESULT: SKIP"
    TIMEOUT 300
    PROCESSORS ${np}
    LABELS "ci;${T_LABELS}"
    ENVIRONMENT "OMP_NUM_THREADS=1;SUPERLU_ACC_OFFLOAD=0;SUPERLU_ACC_SOLVE=0;GPU3DVERSION=0;${T_ENV}")
  if(T_XFAIL)
    set_tests_properties(${test_name} PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

set(CI_DATA ${SuperLU_DIST_SOURCE_DIR}/EXAMPLE)
set(CI_G2 -x 40 -y 30)          # generated 2D grid problem, n = 1200
set(CI_G3 -x 12 -y 11 -z 10)    # generated 3D grid problem, n = 1320

# -f: one letter per call.  D DOFACT, F FACTORED, P SamePattern,
#     R SamePattern_SameRowPerm; each call has new values / a new RHS.
foreach(p ${CI_PRECISIONS})
  # ---- 2D interface p?gssvx: process grids x factorization modes
  foreach(grid 1x1 2x2 2x3 3x1)
    string(REPLACE "x" ";" g ${grid})
    list(GET g 0 r)
    list(GET g 1 c)
    math(EXPR np "${r}*${c}")
    add_superlu_dist_ci_test(2d_${grid}_modes ${p} ${np} -r ${r} -c ${c} -f DFPRF ${CI_G3}
                             LABELS cpu 2d)
  endforeach()

  # ---- 3D interface p?gssvx3d, C code (GPU3DVERSION=0) and C++ code (=1)
  foreach(grid 1x1x1 1x1x2 2x1x2 1x2x2 1x1x4 2x2x1)
    string(REPLACE "x" ";" g ${grid})
    list(GET g 0 r)
    list(GET g 1 c)
    list(GET g 2 d)
    math(EXPR np "${r}*${c}*${d}")
    foreach(v 0 1)
      add_superlu_dist_ci_test(3d_${grid}_modes_v${v} ${p} ${np}
                               -r ${r} -c ${c} -d ${d} -f DFPF ${CI_G3}
                               ENV GPU3DVERSION=${v} LABELS cpu 3d)
    endforeach()
  endforeach()

  # ---- SolveOnly, ILU(0), batched interfaces
  add_superlu_dist_ci_test(3d_solveonly ${p} 4 -r 2 -c 1 -d 2 -S 1 -f DF -s 3 ${CI_G3}
                           LABELS cpu 3d solveonly)
  add_superlu_dist_ci_test(2d_ilu0 ${p} 4 -r 2 -c 2 -I 0 -q 0 -f DFPRF ${CI_G3}
                           LABELS cpu 2d ilu)
  add_superlu_dist_ci_test(3d_ilu0 ${p} 4 -r 2 -c 1 -d 2 -I 0 -q 0 -k 1 -f DFPF ${CI_G3}
                           LABELS cpu 3d ilu)
  add_superlu_dist_ci_test(batch ${p} 1 -b 4 -s 2 ${CI_G3} LABELS cpu batch)
  add_superlu_dist_ci_test(vbatch ${p} 1 -B 4 -f DRR ${CI_G3} LABELS cpu batch)
endforeach()

# ---- file matrices (nontrivial row pivoting)
add_superlu_dist_ci_test(2d_big d 6 -r 3 -c 2 -f DFPRFD ${CI_DATA}/big.rua LABELS cpu 2d)
add_superlu_dist_ci_test(3d_big d 4 -r 1 -c 2 -d 2 -f DFPF ${CI_DATA}/big.rua LABELS cpu 3d)
add_superlu_dist_ci_test(2d_big s 4 -r 2 -c 2 -i 1 -f DF ${CI_DATA}/big.rua LABELS cpu 2d)
add_superlu_dist_ci_test(2d_cg20 z 6 -r 3 -c 2 -f DFPRF ${CI_DATA}/cg20.cua LABELS cpu 2d)
add_superlu_dist_ci_test(3d_cg20 z 4 -r 1 -c 2 -d 2 -f DFPF ${CI_DATA}/cg20.cua LABELS cpu 3d)

# ---- column orderings: NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
#      METIS_AT_PLUS_A, PARMETIS (skipped when the build lacks the package)
foreach(q 0 1 2 3 4 5)
  add_superlu_dist_ci_test(2d_colperm${q} d 4 -r 2 -c 2 -q ${q} -f DFPRF ${CI_G2}
                           LABELS cpu 2d ordering)
endforeach()
foreach(q 0 2 3 4)
  add_superlu_dist_ci_test(3d_colperm${q} d 4 -r 2 -c 1 -d 2 -q ${q} -f DFPF ${CI_G2}
                           LABELS cpu 3d ordering)
endforeach()
add_superlu_dist_ci_test(3d_colperm5 d 4 -r 2 -c 2 -d 1 -q 5 -f DFPF ${CI_G3}
                         LABELS cpu 3d ordering)
add_superlu_dist_ci_test(2d_parsymbfact d 4 -r 2 -c 2 -q 5 -P 1 -f DFPRF ${CI_G3}
                         LABELS cpu 2d ordering)
add_superlu_dist_ci_test(2d_parsymbfact z 4 -r 2 -c 2 -q 5 -P 1 -f DFPRF ${CI_DATA}/cg20.cua
                         LABELS cpu 2d ordering)
# parallel symbolic factorization of a NATURAL ordering (no separator tree)
add_superlu_dist_ci_test(2d_parsymbfact_natural d 6 -r 3 -c 2 -q 0 -P 1 -f DFPRF ${CI_G3}
                         LABELS cpu 2d ordering)

# ---- row permutation (-k 1 scrambles the rows: only pivoting can solve it),
#      no equilibration, refinement, DiagInv, GMRES, XY-major rank order
add_superlu_dist_ci_test(2d_norowperm d 4 -r 2 -c 2 -p 0 -e 0 ${CI_G3} LABELS cpu 2d)
foreach(p d z)
  add_superlu_dist_ci_test(2d_mc64 ${p} 4 -r 2 -c 2 -p 1 -k 1 -f DFPRF ${CI_G3} LABELS cpu 2d)
  add_superlu_dist_ci_test(3d_mc64 ${p} 4 -r 2 -c 1 -d 2 -p 1 -k 1 -f DFPF ${CI_G3} LABELS cpu 3d)
  add_superlu_dist_ci_test(2d_diaginv ${p} 4 -r 2 -c 2 -D 1 -f DFPRF ${CI_G3} LABELS cpu 2d)
  add_superlu_dist_ci_test(3d_diaginv ${p} 4 -r 2 -c 1 -d 2 -D 1 -f DFPF ${CI_G3} LABELS cpu 3d)
endforeach()
add_superlu_dist_ci_test(2d_hwpm d 4 -r 2 -c 2 -p 2 -k 1 ${CI_G3} LABELS cpu 2d)
add_superlu_dist_ci_test(2d_refine d 4 -r 2 -c 2 -i 2 -s 2 -f DFPRF ${CI_DATA}/big.rua LABELS cpu 2d)
add_superlu_dist_ci_test(3d_refine d 4 -r 2 -c 1 -d 2 -i 2 -f DFPF ${CI_DATA}/big.rua LABELS cpu 3d)
add_superlu_dist_ci_test(2d_gmres d 4 -r 2 -c 2 -G 1 -f DF ${CI_G3} LABELS cpu 2d)
add_superlu_dist_ci_test(2d_ilu0_gmres d 4 -r 2 -c 2 -I 0 -q 0 -G 1 -f DF ${CI_G3} LABELS cpu 2d ilu)
add_superlu_dist_ci_test(3d_rankorder_xy d 4 -r 2 -c 1 -d 2 -f DFPF ${CI_G3}
                         ENV SUPERLU_RANKORDER=XY LABELS cpu 3d)

# ---- known library defects (expected to fail; remove XFAIL once fixed)
# (SamePattern_SameRowPerm, letter R of -f, is unreliable in the 3D interface:
#  it hangs, corrupts the heap or returns a wrong solution depending on the
#  grid and the ordering.  No 3D test runs it; see .ci_tests.sh, group 3d.)

######################################################################
# GPU tests ("ctest -L gpu"), only in a GPU build.
#   SUPERLU_ACC_OFFLOAD=1  factorization on the GPU
#   SUPERLU_ACC_SOLVE=1    triangular solves on the GPU; on a 2D grid of more
#                          than one process this needs NVSHMEM/ROCSHMEM (the
#                          test is skipped without it)
#   GPU3DVERSION=0|1       3D factorization: C code (0) or C++ code (1); the
#                          C++ code on more than one process needs a
#                          GPU-aware MPI: -Denable_gpu_aware_mpi_tests=ON
# With a GPU-aware MPI or NVSHMEM every process must see the GPUs of the
# processes it talks to (no "srun --gpus-per-task=1").
######################################################################
option(enable_gpu_aware_mpi_tests
       "Add the GPU tests that need a GPU-aware MPI" OFF)
if(TPL_ENABLE_CUDALIB OR TPL_ENABLE_HIPLIB)
  set(CI_GPU_ENV SUPERLU_ACC_OFFLOAD=1 SUPERLU_BIND_MPI_GPU=1 SUPERLU_N_GEMM=64)
  foreach(p ${CI_PRECISIONS})
    # 2D interface: GPU factorization, CPU or GPU triangular solve
    add_superlu_dist_ci_test(gpu_2d_1x1 ${p} 1 -r 1 -c 1 -f DFPRF ${CI_G3}
                             ENV ${CI_GPU_ENV} SUPERLU_ACC_SOLVE=1 LABELS gpu 2d)
    add_superlu_dist_ci_test(gpu_2d_2x2_cpusolve ${p} 4 -r 2 -c 2 -f DFPRF ${CI_G3}
                             ENV ${CI_GPU_ENV} LABELS gpu 2d)
    add_superlu_dist_ci_test(gpu_2d_2x2 ${p} 4 -r 2 -c 2 -f DFPRF ${CI_G3}
                             ENV ${CI_GPU_ENV} SUPERLU_ACC_SOLVE=1 LABELS gpu 2d)
    # 3D interface: C code (GPU3DVERSION=0) and C++ code (=1), GPU solve
    foreach(v 0 1)
      foreach(grid 1x1x1 1x1x2 1x1x4 2x1x2)
        string(REPLACE "x" ";" g ${grid})
        list(GET g 0 r)
        list(GET g 1 c)
        list(GET g 2 d)
        math(EXPR np "${r}*${c}*${d}")
        if(v EQUAL 0 OR np EQUAL 1 OR enable_gpu_aware_mpi_tests)
          add_superlu_dist_ci_test(gpu_3d_${grid}_v${v} ${p} ${np}
                                   -r ${r} -c ${c} -d ${d} -f DFPF ${CI_G3}
                                   ENV ${CI_GPU_ENV} GPU3DVERSION=${v} SUPERLU_ACC_SOLVE=1
                                       MPICH_GPU_SUPPORT_ENABLED=1
                                   LABELS gpu 3d)
        endif()
      endforeach()
    endforeach()
    add_superlu_dist_ci_test(gpu_3d_solveonly ${p} 2 -r 1 -c 1 -d 2 -S 1 -f DF ${CI_G3}
                             ENV ${CI_GPU_ENV} SUPERLU_ACC_SOLVE=1 LABELS gpu 3d solveonly)
    add_superlu_dist_ci_test(gpu_3d_ilu0 ${p} 2 -r 1 -c 1 -d 2 -I 0 -q 0 -f DF ${CI_G3}
                             ENV ${CI_GPU_ENV} GPU3DVERSION=0 SUPERLU_ACC_SOLVE=1 LABELS gpu 3d ilu)
    add_superlu_dist_ci_test(gpu_batch ${p} 1 -b 8 -s 2 ${CI_G3}
                             ENV ${CI_GPU_ENV} GPU3DVERSION=1 SUPERLU_ACC_SOLVE=1 LABELS gpu batch)
    add_superlu_dist_ci_test(gpu_vbatch ${p} 1 -B 4 -f DRR ${CI_G3}
                             ENV ${CI_GPU_ENV} GPU3DVERSION=1 SUPERLU_ACC_SOLVE=1 LABELS gpu batch)
  endforeach()
endif()
