# Sherry; may not need it?
include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

# Fortran stuff
add_definitions(${MPI_Fortran_COMPILE_FLAGS})
include_directories(${MPI_Fortran_INCLUDE_PATH})
link_directories(${MPI_Fortran_LIBRARIES})

# Libs linked to all of the examples
set(all_link_libs superlu_dist ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
#message("!!! in Fortran: MPI_Fortran_LINK_FLAGS='${MPI_Fortran_LINK_FLAGS}'")
#message("!!! in Fortran: all_link_libs='${all_link_libs}'")
#message("!!! in Fortran: cxx_implicit='${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}'")
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()

set(F_MOD superlupara.f90 superlu_mod.f90)

if(enable_double)
  set(C_DWRAP c2f_dcreate_matrix_x_b.c superlu_c2f_dwrap.c)
  set(F_DEXM ${F_MOD} f_pddrive.f90 ${C_DWRAP})
  add_executable(f_pddrive ${F_DEXM})
  target_link_libraries(f_pddrive ${all_link_libs})
#  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_DEXM3D ${F_MOD} f_pddrive3d.f90 ${C_DWRAP})
  add_executable(f_pddrive3d ${F_DEXM3D})
  target_link_libraries(f_pddrive3d ${all_link_libs})
  set_target_properties(f_pddrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_5x5 ${F_MOD} f_5x5.f90 sp_ienv.c ${C_DWRAP})
  add_executable(f_5x5 ${F_5x5})
  target_link_libraries(f_5x5 ${all_link_libs})
  set_target_properties(f_5x5 PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()

if(enable_complex16)
  set(C_ZWRAP c2f_zcreate_matrix_x_b.c superlu_c2f_zwrap.c)
  
  set(F_ZEXM ${F_MOD} f_pzdrive.f90 ${C_ZWRAP})
  add_executable(f_pzdrive ${F_ZEXM})
  target_link_libraries(f_pzdrive ${all_link_libs})
#  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")

  set(F_ZEXM3D ${F_MOD} f_pzdrive3d.f90 ${C_ZWRAP})
  add_executable(f_pzdrive3d ${F_ZEXM3D})
  target_link_libraries(f_pzdrive3d ${all_link_libs})
  set_target_properties(f_pzdrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()
