include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

# Libs linked to all of the examples
set(all_link_libs superlu_dist ${BLAS_LIB} m)

function(add_superlu_dist_test output input target)
  set(TEST_INPUT "${SuperLU_DIST_SOURCE_DIR}/EXAMPLE/${input}")
  set(TEST_OUTPUT "${SuperLU_DIST_BINARY_DIR}/EXAMPLE/${output}")

##  get_target_property(TEST_LOC ${target} LOCATION)
  set(TEST_LOC ${CMAKE_CURRENT_BINARY_DIR})

  set(SINGLE_OUTPUT ${SuperLU_DIST_BINARY_DIR}/EXAMPLE/${target}.out)
  add_test( ${target}  "${CMAKE_COMMAND}"
	    -DTEST=${TEST_LOC} ${MPIEXEC} ${target} -r 2 -c 2 ${TEST_INPUT}
	    -DOUTPUT=${SINGLE_OUTPUT}
	  )
# MPI variables:
# ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} PROCS
#  	${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS)

endfunction(add_superlu_dist_test)


if(enable_double)
  set(DEXM pddrive.c dcreate_matrix.c)
  add_executable(pddrive ${DEXM})
  target_link_libraries(pddrive ${all_link_libs})

  add_superlu_dist_test(pddrive.out big.rua pddrive)

  set(DEXM1 pddrive1.c dcreate_matrix.c)
  add_executable(pddrive1 ${DEXM1})
  target_link_libraries(pddrive1 ${all_link_libs})

  set(DEXM2 pddrive2.c dcreate_matrix.c)
  add_executable(pddrive2 ${DEXM2})
  target_link_libraries(pddrive2 ${all_link_libs})

  set(DEXM3 pddrive3.c dcreate_matrix.c)
  add_executable(pddrive3 ${DEXM3})
  target_link_libraries(pddrive3 ${all_link_libs})

  set(DEXM4 pddrive4.c dcreate_matrix.c)
  add_executable(pddrive4 ${DEXM4})
  target_link_libraries(pddrive4 ${all_link_libs})

  set(DEXMG pddrive_ABglobal.c)
  add_executable(pddrive_ABglobal ${DEXMG})
  target_link_libraries(pddrive_ABglobal ${all_link_libs})

  set(DEXMG1 pddrive1_ABglobal.c)
  add_executable(pddrive1_ABglobal ${DEXMG1})
  target_link_libraries(pddrive1_ABglobal ${all_link_libs})

  set(DEXMG2 pddrive2_ABglobal.c)
  add_executable(pddrive2_ABglobal ${DEXMG2})
  target_link_libraries(pddrive2_ABglobal ${all_link_libs})

  set(DEXMG3 pddrive3_ABglobal.c)
  add_executable(pddrive3_ABglobal ${DEXMG3})
  target_link_libraries(pddrive3_ABglobal ${all_link_libs})

  set(DEXMG4 pddrive4_ABglobal.c)
  add_executable(pddrive4_ABglobal ${DEXMG4})
  target_link_libraries(pddrive4_ABglobal ${all_link_libs})
endif()


if(enable_complex16)

  set(ZEXM pzdrive.c zcreate_matrix.c)
  add_executable(pzdrive ${ZEXM})
  target_link_libraries(pzdrive ${all_link_libs})

  set(ZEXM1 pzdrive1.c zcreate_matrix.c)
  add_executable(pzdrive1 ${ZEXM1})
  target_link_libraries(pzdrive1 ${all_link_libs})

  set(ZEXM2 pzdrive2.c zcreate_matrix.c)
  add_executable(pzdrive2 ${ZEXM2})
  target_link_libraries(pzdrive2 ${all_link_libs})

  set(ZEXM3 pzdrive3.c zcreate_matrix.c)
  add_executable(pzdrive3 ${ZEXM3})
  target_link_libraries(pzdrive3 ${all_link_libs})

  set(ZEXM4 pzdrive4.c zcreate_matrix.c)
  add_executable(pzdrive4 ${ZEXM4})
  target_link_libraries(pzdrive4 ${all_link_libs})

  set(ZEXMG pzdrive_ABglobal.c)
  add_executable(pzdrive_ABglobal ${ZEXMG})
  target_link_libraries(pzdrive_ABglobal ${all_link_libs})

  set(ZEXMG1 pzdrive1_ABglobal.c)
  add_executable(pzdrive1_ABglobal ${ZEXMG1})
  target_link_libraries(pzdrive1_ABglobal ${all_link_libs})

  set(ZEXMG2 pzdrive2_ABglobal.c)
  add_executable(pzdrive2_ABglobal ${ZEXMG2})
  target_link_libraries(pzdrive2_ABglobal ${all_link_libs})

  set(ZEXMG3 pzdrive3_ABglobal.c)
  add_executable(pzdrive3_ABglobal ${ZEXMG3})
  target_link_libraries(pzdrive3_ABglobal ${all_link_libs})

  set(ZEXMG4 pzdrive4_ABglobal.c)
  add_executable(pzdrive4_ABglobal ${ZEXMG4})
  target_link_libraries(pzdrive4_ABglobal ${all_link_libs})

endif()
