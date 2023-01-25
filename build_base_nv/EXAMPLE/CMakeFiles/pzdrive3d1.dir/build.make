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
CMAKE_BINARY_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv

# Include any dependencies generated for this target.
include EXAMPLE/CMakeFiles/pzdrive3d1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include EXAMPLE/CMakeFiles/pzdrive3d1.dir/compiler_depend.make

# Include the progress variables for this target.
include EXAMPLE/CMakeFiles/pzdrive3d1.dir/progress.make

# Include the compile flags for this target's objects.
include EXAMPLE/CMakeFiles/pzdrive3d1.dir/flags.make

EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/flags.make
EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive3d1.c
EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o -MF CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o.d -o CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive3d1.c

EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive3d1.c > CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.i

EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive3d1.c -o CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.s

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/flags.make
EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix.c
EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o -MF CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o.d -o CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix.c

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix.c > CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.i

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix.c -o CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.s

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/flags.make
EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix3d.c
EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o: EXAMPLE/CMakeFiles/pzdrive3d1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o -MF CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o.d -o CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix3d.c

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix3d.c > CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.i

EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/zcreate_matrix3d.c -o CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.s

# Object files for target pzdrive3d1
pzdrive3d1_OBJECTS = \
"CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o" \
"CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o" \
"CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o"

# External object files for target pzdrive3d1
pzdrive3d1_EXTERNAL_OBJECTS =

EXAMPLE/pzdrive3d1: EXAMPLE/CMakeFiles/pzdrive3d1.dir/pzdrive3d1.c.o
EXAMPLE/pzdrive3d1: EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix.c.o
EXAMPLE/pzdrive3d1: EXAMPLE/CMakeFiles/pzdrive3d1.dir/zcreate_matrix3d.c.o
EXAMPLE/pzdrive3d1: EXAMPLE/CMakeFiles/pzdrive3d1.dir/build.make
EXAMPLE/pzdrive3d1: SRC/libsuperlu_dist.so.8.1.2
EXAMPLE/pzdrive3d1: /opt/cray/pe/libsci/22.11.1.2/nvidia/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/pzdrive3d1: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libparmetis/libparmetis.so
EXAMPLE/pzdrive3d1: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libmetis/libmetis.so
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libcudart.so
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusolver.so
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublas.so
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libculibos.a
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublasLt.so
EXAMPLE/pzdrive3d1: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusparse.so
EXAMPLE/pzdrive3d1: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mpi_mp.so
EXAMPLE/pzdrive3d1: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/pzdrive3d1: EXAMPLE/CMakeFiles/pzdrive3d1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable pzdrive3d1"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pzdrive3d1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
EXAMPLE/CMakeFiles/pzdrive3d1.dir/build: EXAMPLE/pzdrive3d1
.PHONY : EXAMPLE/CMakeFiles/pzdrive3d1.dir/build

EXAMPLE/CMakeFiles/pzdrive3d1.dir/clean:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE && $(CMAKE_COMMAND) -P CMakeFiles/pzdrive3d1.dir/cmake_clean.cmake
.PHONY : EXAMPLE/CMakeFiles/pzdrive3d1.dir/clean

EXAMPLE/CMakeFiles/pzdrive3d1.dir/depend:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/EXAMPLE/CMakeFiles/pzdrive3d1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : EXAMPLE/CMakeFiles/pzdrive3d1.dir/depend

