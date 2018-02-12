# - Try to find ParMETIS
# Once done this will define
#
#  PARMETIS_FOUND        - system has ParMETIS
#  PARMETIS_INCLUDE_DIRS - include directories for ParMETIS
#  PARMETIS_LIBRARIES    - libraries for ParMETIS
#
# and the imported target
#
#  ParMETIS::ParMETIS

find_path(ParMETIS_INCLUDE_DIR parmetis.h
  DOC "Directory where the ParMETIS header files are located"
)
mark_as_advanced(ParMETIS_INCLUDE_DIR)
set(ParMETIS_INCLUDE_DIRS "${ParMETIS_INCLUDE_DIR}")

find_library(ParMETIS_LIBRARY
  NAMES parmetis
  DOC "Directory where the ParMETIS library is located"
)
mark_as_advanced(ParMETIS_LIBRARY)
set(ParMETIS_LIBRARIES "${ParMETIS_LIBRARY}")

# Get ParMETIS version
if(NOT PARMETIS_VERSION_STRING AND PARMETIS_INCLUDE_DIR AND EXISTS "${PARMETIS_INCLUDE_DIR}/parmetis.h")
  set(version_pattern "^#define[\t ]+PARMETIS_(MAJOR|MINOR)_VERSION[\t ]+([0-9\\.]+)$")
  file(STRINGS "${PARMETIS_INCLUDE_DIR}/parmetis.h" parmetis_version REGEX ${version_pattern})

  foreach(match ${parmetis_version})
    if(PARMETIS_VERSION_STRING)
      set(PARMETIS_VERSION_STRING "${PARMETIS_VERSION_STRING}.")
    endif()
    string(REGEX REPLACE ${version_pattern} "${PARMETIS_VERSION_STRING}\\2" PARMETIS_VERSION_STRING ${match})
    set(PARMETIS_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  endforeach()
  unset(parmetis_version)
  unset(version_pattern)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParMETIS
  REQUIRED_VARS ParMETIS_LIBRARY ParMETIS_INCLUDE_DIR
  VERSION_VAR PARMETIS_VERSION_STRING
)

# Dependencies
include(CMakeFindDependencyMacro)
#find_dependency(MPI)
find_dependency(METIS)

if(ParMETIS_FOUND)
  if(NOT TARGET ParMETIS::ParMETIS)
    add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
  endif()
  set_property(TARGET ParMETIS::ParMETIS PROPERTY IMPORTED_LOCATION "${ParMETIS_LIBRARY}")
  set_property(TARGET ParMETIS::ParMETIS PROPERTY INTERFACE_LINK_LIBRARIES METIS::METIS)
  set_property(TARGET ParMETIS::ParMETIS PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${ParMETIS_INCLUDE_DIRS}")
endif()
