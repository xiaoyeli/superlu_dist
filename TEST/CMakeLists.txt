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
  add_superlu_dist_tests(pdtest g20.rua)
endif()

#if(enable_complex16)
#  set(ZTEST pztest.c zcreate_matrix.c pzcompute_resid.c)
#endif()
