include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

# Libs linked to all of the examples
set(all_link_libs superlu_dist ${BLAS_LIB})
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()

function(add_superlu_dist_example target input nprow npcol)
    set(EXAMPLE_INPUT "${SuperLU_DIST_SOURCE_DIR}/EXAMPLE/${input}")
    set(EXAMPLE_OUTPUT "${SuperLU_DIST_BINARY_DIR}/EXAMPLE/${target}.out")

##  get_target_property(TEST_LOC ${target} LOCATION)
    set(EXAMPLE_LOC ${CMAKE_CURRENT_BINARY_DIR})

    MATH( EXPR procs "${nprow}*${npcol}" )
#    message("MPIEXEC_FLAG is ${MPIEXEC_NUMPROC_FLAG}")

# corresponding to mpiexec -n 4 pddrive -r <nprow> -c <npcol> g20.rua
    add_test(${target} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${procs}
             ${MPIEXEC_PREFLAGS} ${EXAMPLE_LOC}/${target} ${MPIEXEC_POSTFLAGS} -r "${nprow}" -c "${npcol}" "${EXAMPLE_INPUT}")

#     add_test(NAME ${target} COMMAND "${CMAKE_COMMAND}"
#              -DTEST=${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${procs}
#             ${MPIEXEC_PREFLAGS} ${target} ${MPIEXEC_POSTFLAGS} -r "${nprow}" -c "${npcol}" "${TEST_INPUT}"
#	     -DOUTPUT=${target}.out
#	    -P "${SuperLU_DIST_SOURCE_DIR}/EXAMPLE/runexample.cmake" )

# MPI variables:
# ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} PROCS
#  	${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS)

endfunction(add_superlu_dist_example)


if(enable_double)
  set(DEXM pddrive.c dcreate_matrix.c)
  add_executable(pddrive ${DEXM})
  target_link_libraries(pddrive ${all_link_libs})
  install(TARGETS pddrive RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM1 pddrive1.c dcreate_matrix.c)
  add_executable(pddrive1 ${DEXM1})
  target_link_libraries(pddrive1 ${all_link_libs})
  add_superlu_dist_example(pddrive1 big.rua 2 2)
  install(TARGETS pddrive1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM2 pddrive2.c dcreate_matrix.c dcreate_matrix_perturbed.c)
  add_executable(pddrive2 ${DEXM2})
  target_link_libraries(pddrive2 ${all_link_libs})
  add_superlu_dist_example(pddrive2 big.rua 2 2)
  install(TARGETS pddrive2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM3 pddrive3.c dcreate_matrix.c)
  add_executable(pddrive3 ${DEXM3})
  target_link_libraries(pddrive3 ${all_link_libs})
  add_superlu_dist_example(pddrive3 big.rua 2 2)
  install(TARGETS pddrive3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM4 pddrive4.c dcreate_matrix.c)
  add_executable(pddrive4 ${DEXM4})
  target_link_libraries(pddrive4 ${all_link_libs})
  install(TARGETS pddrive4 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM3D pddrive3d.c dcreate_matrix.c dcreate_matrix3d.c)
  add_executable(pddrive3d ${DEXM3D})
  target_link_libraries(pddrive3d ${all_link_libs})
  install(TARGETS pddrive3d RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM3D1 pddrive3d1.c dcreate_matrix.c dcreate_matrix3d.c)
  add_executable(pddrive3d1 ${DEXM3D1})
  target_link_libraries(pddrive3d1 ${all_link_libs})
  install(TARGETS pddrive3d1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM3D2 pddrive3d2.c dcreate_matrix.c dcreate_matrix3d.c)
  add_executable(pddrive3d2 ${DEXM3D2})
  target_link_libraries(pddrive3d2 ${all_link_libs})
  install(TARGETS pddrive3d2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXM3D3 pddrive3d3.c dcreate_matrix.c dcreate_matrix3d.c)
  add_executable(pddrive3d3 ${DEXM3D3})
  target_link_libraries(pddrive3d3 ${all_link_libs})
  install(TARGETS pddrive3d3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMG pddrive_ABglobal.c)
  add_executable(pddrive_ABglobal ${DEXMG})
  target_link_libraries(pddrive_ABglobal ${all_link_libs})
  install(TARGETS pddrive_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMG1 pddrive1_ABglobal.c)
  add_executable(pddrive1_ABglobal ${DEXMG1})
  target_link_libraries(pddrive1_ABglobal ${all_link_libs})
  install(TARGETS pddrive1_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMG2 pddrive2_ABglobal.c)
  add_executable(pddrive2_ABglobal ${DEXMG2})
  target_link_libraries(pddrive2_ABglobal ${all_link_libs})
  install(TARGETS pddrive2_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMG3 pddrive3_ABglobal.c)
  add_executable(pddrive3_ABglobal ${DEXMG3})
  target_link_libraries(pddrive3_ABglobal ${all_link_libs})
  install(TARGETS pddrive3_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMG4 pddrive4_ABglobal.c)
  add_executable(pddrive4_ABglobal ${DEXMG4})
  target_link_libraries(pddrive4_ABglobal ${all_link_libs})
  install(TARGETS pddrive4_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(DEXMS pddrive_spawn.c dcreate_matrix.c)
  add_executable(pddrive_spawn ${DEXMS})
  target_link_libraries(pddrive_spawn ${all_link_libs})
  install(TARGETS pddrive_spawn RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

endif() #### end enable_double

if(enable_single)
  set(SEXM psdrive.c screate_matrix.c screate_A_x_b.c)
  add_executable(psdrive ${SEXM})
  target_link_libraries(psdrive ${all_link_libs})
  install(TARGETS psdrive RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM1 psdrive1.c screate_matrix.c)
  add_executable(psdrive1 ${SEXM1})
  target_link_libraries(psdrive1 ${all_link_libs})
  add_superlu_dist_example(psdrive1 big.rua 2 2)
  install(TARGETS psdrive1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM2 psdrive2.c screate_matrix.c screate_matrix_perturbed.c)
  add_executable(psdrive2 ${SEXM2})
  target_link_libraries(psdrive2 ${all_link_libs})
  add_superlu_dist_example(psdrive2 big.rua 2 2)
  install(TARGETS psdrive2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM3 psdrive3.c screate_matrix.c)
  add_executable(psdrive3 ${SEXM3})
  target_link_libraries(psdrive3 ${all_link_libs})
  add_superlu_dist_example(psdrive3 big.rua 2 2)
  install(TARGETS psdrive3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM4 psdrive4.c screate_matrix.c)
  add_executable(psdrive4 ${SEXM4})
  target_link_libraries(psdrive4 ${all_link_libs})
  install(TARGETS psdrive4 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM3D psdrive3d.c screate_matrix.c screate_matrix3d.c)
  add_executable(psdrive3d ${SEXM3D})
  target_link_libraries(psdrive3d ${all_link_libs})
  install(TARGETS psdrive3d RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM3D1 psdrive3d1.c screate_matrix.c screate_matrix3d.c)
  add_executable(psdrive3d1 ${SEXM3D1})
  target_link_libraries(psdrive3d1 ${all_link_libs})
  install(TARGETS psdrive3d1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM3D2 psdrive3d2.c screate_matrix.c screate_matrix3d.c)
  add_executable(psdrive3d2 ${SEXM3D2})
  target_link_libraries(psdrive3d2 ${all_link_libs})
  install(TARGETS psdrive3d2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(SEXM3D3 psdrive3d3.c screate_matrix.c screate_matrix3d.c)
  add_executable(psdrive3d3 ${SEXM3D3})
  target_link_libraries(psdrive3d3 ${all_link_libs})
  install(TARGETS psdrive3d3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

endif() #### end enable_single


if(enable_complex16)

  set(ZEXM pzdrive.c zcreate_matrix.c)
  add_executable(pzdrive ${ZEXM})
  target_link_libraries(pzdrive ${all_link_libs})
  install(TARGETS pzdrive RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM1 pzdrive1.c zcreate_matrix.c)
  add_executable(pzdrive1 ${ZEXM1})
  target_link_libraries(pzdrive1 ${all_link_libs})
  add_superlu_dist_example(pzdrive1 cg20.cua 2 2)
  install(TARGETS pzdrive1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM2 pzdrive2.c zcreate_matrix.c zcreate_matrix_perturbed.c)
  add_executable(pzdrive2 ${ZEXM2})
  target_link_libraries(pzdrive2 ${all_link_libs})
  add_superlu_dist_example(pzdrive2 cg20.cua 2 2)
  install(TARGETS pzdrive2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM3 pzdrive3.c zcreate_matrix.c)
  add_executable(pzdrive3 ${ZEXM3})
  target_link_libraries(pzdrive3 ${all_link_libs})
  add_superlu_dist_example(pzdrive3 cg20.cua 2 2)
  install(TARGETS pzdrive3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM4 pzdrive4.c zcreate_matrix.c)
  add_executable(pzdrive4 ${ZEXM4})
  target_link_libraries(pzdrive4 ${all_link_libs})
  install(TARGETS pzdrive4 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM3D pzdrive3d.c zcreate_matrix.c zcreate_matrix3d.c)
  add_executable(pzdrive3d ${ZEXM3D})
  target_link_libraries(pzdrive3d ${all_link_libs})
  install(TARGETS pzdrive3d RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM3D1 pzdrive3d1.c zcreate_matrix.c zcreate_matrix3d.c)
  add_executable(pzdrive3d1 ${ZEXM3D1})
  target_link_libraries(pzdrive3d1 ${all_link_libs})
  install(TARGETS pzdrive3d1 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM3D2 pzdrive3d2.c zcreate_matrix.c zcreate_matrix3d.c)
  add_executable(pzdrive3d2 ${ZEXM3D2})
  target_link_libraries(pzdrive3d2 ${all_link_libs})
  install(TARGETS pzdrive3d2 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXM3D3 pzdrive3d3.c zcreate_matrix.c zcreate_matrix3d.c)
  add_executable(pzdrive3d3 ${ZEXM3D3})
  target_link_libraries(pzdrive3d3 ${all_link_libs})
  install(TARGETS pzdrive3d3 RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  
  
  set(ZEXMG pzdrive_ABglobal.c)
  add_executable(pzdrive_ABglobal ${ZEXMG})
  target_link_libraries(pzdrive_ABglobal ${all_link_libs})
  install(TARGETS pzdrive_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXMG1 pzdrive1_ABglobal.c)
  add_executable(pzdrive1_ABglobal ${ZEXMG1})
  target_link_libraries(pzdrive1_ABglobal ${all_link_libs})
  install(TARGETS pzdrive1_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")  

  set(ZEXMG2 pzdrive2_ABglobal.c)
  add_executable(pzdrive2_ABglobal ${ZEXMG2})
  target_link_libraries(pzdrive2_ABglobal ${all_link_libs})
  install(TARGETS pzdrive2_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")   
  
  set(ZEXMG3 pzdrive3_ABglobal.c)
  add_executable(pzdrive3_ABglobal ${ZEXMG3})
  target_link_libraries(pzdrive3_ABglobal ${all_link_libs})
  install(TARGETS pzdrive3_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")   

  set(ZEXMG4 pzdrive4_ABglobal.c)
  add_executable(pzdrive4_ABglobal ${ZEXMG4})
  target_link_libraries(pzdrive4_ABglobal ${all_link_libs})
  install(TARGETS pzdrive4_ABglobal RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")   

  set(ZEXMS pzdrive_spawn.c zcreate_matrix.c)
  add_executable(pzdrive_spawn ${ZEXMS})
  target_link_libraries(pzdrive_spawn ${all_link_libs})  
  install(TARGETS pzdrive_spawn RUNTIME DESTINATION "${INSTALL_LIB_DIR}/EXAMPLE")   

endif()
