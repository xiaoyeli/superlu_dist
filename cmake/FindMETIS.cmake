# - Try to find METIS
# Once done this will define
#
#  METIS_FOUND        - system has METIS
#  METIS_INCLUDE_DIRS - include directories for METIS
#  METIS_LIBRARIES    - libraries for METIS
#
# and the imported target
#
#  METIS::METIS

find_path(METIS_INCLUDE_DIR metis.h
  DOC "Directory where the METIS header files are located"
)
mark_as_advanced(METIS_INCLUDE_DIR)
set(METIS_INCLUDE_DIRS "${METIS_INCLUDE_DIR}")

find_library(METIS_LIBRARY
  NAMES metis
  DOC "Directory where the METIS library is located"
)
mark_as_advanced(METIS_LIBRARY)
set(METIS_LIBRARIES "${METIS_LIBRARY}")

# Get METIS version
if(NOT METIS_VERSION_STRING AND METIS_INCLUDE_DIR AND EXISTS "${METIS_INCLUDE_DIR}/metis.h")
  set(version_pattern "^#define[\t ]+METIS_(MAJOR|MINOR)_VERSION[\t ]+([0-9\\.]+)$")
  file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" metis_version REGEX ${version_pattern})

  foreach(match ${metis_version})
    if(METIS_VERSION_STRING)
      set(METIS_VERSION_STRING "${METIS_VERSION_STRING}.")
    endif()
    string(REGEX REPLACE ${version_pattern} "${METIS_VERSION_STRING}\\2" METIS_VERSION_STRING ${match})
    set(METIS_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  endforeach()
  unset(metis_version)
  unset(version_pattern)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
  REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR
  VERSION_VAR METIS_VERSION_STRING
)

if(METIS_FOUND)
  if(NOT TARGET METIS::METIS)
    add_library(METIS::METIS UNKNOWN IMPORTED)
  endif()
  set_property(TARGET METIS::METIS PROPERTY IMPORTED_LOCATION "${METIS_LIBRARY}")
  set_property(TARGET METIS::METIS PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}")
endif()
