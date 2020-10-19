include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

# Libs linked to all of the examples
set(all_link_libs superlu_dist ${MPI_Fortran_LIBRARIES} ${BLAS_LIB})
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()

set(F_MOD superlupara.f90 superlu_mod.f90)

if(enable_double)
  set(C_DWRAP dcreate_dist_matrix.c superlu_c2f_dwrap.c)
  set(F_DEXM ${F_MOD} dhbcode1.f90 f_pddrive.f90 ${C_DWRAP})
  add_executable(f_pddrive ${F_DEXM})
  target_link_libraries(f_pddrive ${all_link_libs})
  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE Fortran)
  
  set(F_5x5 ${F_MOD} f_5x5.f90 sp_ienv.c ${C_DWRAP})
  add_executable(f_5x5 ${F_5x5})
  target_link_libraries(f_5x5 ${all_link_libs})
  set_target_properties(f_5x5 PROPERTIES LINKER_LANGUAGE Fortran)
endif()

if(enable_complex16)
  set(C_ZWRAP zcreate_dist_matrix.c superlu_c2f_zwrap.c)
  set(F_ZEXM ${F_MOD} zhbcode1.f90 f_pzdrive.f90 ${C_ZWRAP})
  add_executable(f_pzdrive ${F_ZEXM})
  target_link_libraries(f_pzdrive ${all_link_libs})
  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE Fortran)
endif()
