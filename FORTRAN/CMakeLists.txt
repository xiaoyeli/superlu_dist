# Sherry; may not need it?
include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)

set(headers
    ${CMAKE_INSTALL_PREFIX}/FORTRAN/superlu_mod.mod
    ${CMAKE_INSTALL_PREFIX}/FORTRAN/superlupara_mod.mod
    )

set(sources "superlu_c2f_wrap.c")  # initialize precision-independent file

if(enable_double)
  list(APPEND sources c2f_dcreate_matrix_x_b.c superlu_c2f_dwrap.c)
endif()
if(enable_complex16)
  list(APPEND sources c2f_zcreate_matrix_x_b.c superlu_c2f_zwrap.c)
endif()  

add_library(superlu_dist_fortran ${sources})
set(targets superlu_dist_fortran)

install(TARGETS superlu_dist_fortran
# DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
install(FILES ${headers}
# DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
    
# Fortran MPI stuff
add_definitions(${MPI_Fortran_COMPILE_FLAGS})
include_directories(${MPI_Fortran_INCLUDE_PATH})
link_directories(${MPI_Fortran_LIBRARIES})

# Libs to be linked with the Fortran codes
set(fortran_link_libs superlu_dist_fortran ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
set(all_link_libs ${fortran_link_libs} superlu_dist)

#message("!!! in Fortran: MPI_Fortran_LINK_FLAGS='${MPI_Fortran_LINK_FLAGS}'")
#message("!!! in Fortran: all_link_libs='${all_link_libs}'")
#message("!!! in Fortran: cxx_implicit='${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}'")
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()


set(F_MOD superlupara.f90 superlu_mod.f90)

if(enable_double)
  set(F_DEXM ${F_MOD} f_pddrive.f90)
  add_executable(f_pddrive ${F_DEXM})
  target_link_libraries(f_pddrive ${all_link_libs})
#  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_DEXM3D ${F_MOD} f_pddrive3d.f90)
  add_executable(f_pddrive3d ${F_DEXM3D})
  target_link_libraries(f_pddrive3d ${all_link_libs})
  set_target_properties(f_pddrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_5x5 ${F_MOD} f_5x5.f90 sp_ienv.c)
  add_executable(f_5x5 ${F_5x5})
  target_link_libraries(f_5x5 ${all_link_libs})
  set_target_properties(f_5x5 PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()

if(enable_complex16)
  set(F_ZEXM ${F_MOD} f_pzdrive.f90)
  add_executable(f_pzdrive ${F_ZEXM})
  target_link_libraries(f_pzdrive ${all_link_libs})
#  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")

  set(F_ZEXM3D ${F_MOD} f_pzdrive3d.f90)
  add_executable(f_pzdrive3d ${F_ZEXM3D})
  target_link_libraries(f_pzdrive3d ${all_link_libs})
  set_target_properties(f_pzdrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()

