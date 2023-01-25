# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /global/common/software/nersc/pm-2022q4/spack/linux-sles15-zen/cmake-3.24.3-k5msymx/bin/cmake

# The command to remove a file.
RM = /global/common/software/nersc/pm-2022q4/spack/linux-sles15-zen/cmake-3.24.3-k5msymx/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120

# Include any dependencies generated for this target.
include FORTRAN/CMakeFiles/f_5x5.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include FORTRAN/CMakeFiles/f_5x5.dir/compiler_depend.make

# Include the progress variables for this target.
include FORTRAN/CMakeFiles/f_5x5.dir/progress.make

# Include the compile flags for this target's objects.
include FORTRAN/CMakeFiles/f_5x5.dir/flags.make

FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.o: FORTRAN/CMakeFiles/f_5x5.dir/flags.make
FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/f_5x5.F90
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/f_5x5.F90 -o CMakeFiles/f_5x5.dir/f_5x5.F90.o

FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/f_5x5.dir/f_5x5.F90.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/f_5x5.F90 > CMakeFiles/f_5x5.dir/f_5x5.F90.i

FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/f_5x5.dir/f_5x5.F90.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/f_5x5.F90 -o CMakeFiles/f_5x5.dir/f_5x5.F90.s

FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o: FORTRAN/CMakeFiles/f_5x5.dir/flags.make
FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/sp_ienv.c
FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o: FORTRAN/CMakeFiles/f_5x5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o -MF CMakeFiles/f_5x5.dir/sp_ienv.c.o.d -o CMakeFiles/f_5x5.dir/sp_ienv.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/sp_ienv.c

FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/f_5x5.dir/sp_ienv.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/sp_ienv.c > CMakeFiles/f_5x5.dir/sp_ienv.c.i

FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/f_5x5.dir/sp_ienv.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/sp_ienv.c -o CMakeFiles/f_5x5.dir/sp_ienv.c.s

# Object files for target f_5x5
f_5x5_OBJECTS = \
"CMakeFiles/f_5x5.dir/f_5x5.F90.o" \
"CMakeFiles/f_5x5.dir/sp_ienv.c.o"

# External object files for target f_5x5
f_5x5_EXTERNAL_OBJECTS =

FORTRAN/f_5x5: FORTRAN/CMakeFiles/f_5x5.dir/f_5x5.F90.o
FORTRAN/f_5x5: FORTRAN/CMakeFiles/f_5x5.dir/sp_ienv.c.o
FORTRAN/f_5x5: FORTRAN/CMakeFiles/f_5x5.dir/build.make
FORTRAN/f_5x5: FORTRAN/libsuperlu_dist_fortran.a
FORTRAN/f_5x5: /opt/cray/pe/libsci/22.11.1.2/nvidia/20.7/x86_64/lib/libsci_nvidia_mp.so
FORTRAN/f_5x5: SRC/libsuperlu_dist.so.8.1.2
FORTRAN/f_5x5: /opt/cray/pe/libsci/22.11.1.2/nvidia/20.7/x86_64/lib/libsci_nvidia_mp.so
FORTRAN/f_5x5: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libparmetis/libparmetis.so
FORTRAN/f_5x5: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libmetis/libmetis.so
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libcudart.so
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusolver.so
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublas.so
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libculibos.a
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublasLt.so
FORTRAN/f_5x5: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusparse.so
FORTRAN/f_5x5: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mpi_mp.so
FORTRAN/f_5x5: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mp.so
FORTRAN/f_5x5: FORTRAN/CMakeFiles/f_5x5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking Fortran executable f_5x5"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/f_5x5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FORTRAN/CMakeFiles/f_5x5.dir/build: FORTRAN/f_5x5
.PHONY : FORTRAN/CMakeFiles/f_5x5.dir/build

FORTRAN/CMakeFiles/f_5x5.dir/clean:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN && $(CMAKE_COMMAND) -P CMakeFiles/f_5x5.dir/cmake_clean.cmake
.PHONY : FORTRAN/CMakeFiles/f_5x5.dir/clean

FORTRAN/CMakeFiles/f_5x5.dir/depend:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120 /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/FORTRAN/CMakeFiles/f_5x5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : FORTRAN/CMakeFiles/f_5x5.dir/depend

