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
include EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/compiler_depend.make

# Include the progress variables for this target.
include EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/progress.make

# Include the compile flags for this target's objects.
include EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/flags.make

EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o: EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/flags.make
EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive2_ABglobal.c
EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o: EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o -MF CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o.d -o CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive2_ABglobal.c

EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive2_ABglobal.c > CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.i

EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/pzdrive2_ABglobal.c -o CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.s

# Object files for target pzdrive2_ABglobal
pzdrive2_ABglobal_OBJECTS = \
"CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o"

# External object files for target pzdrive2_ABglobal
pzdrive2_ABglobal_EXTERNAL_OBJECTS =

EXAMPLE/pzdrive2_ABglobal: EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/pzdrive2_ABglobal.c.o
EXAMPLE/pzdrive2_ABglobal: EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/build.make
EXAMPLE/pzdrive2_ABglobal: SRC/libsuperlu_dist.so.8.1.2
EXAMPLE/pzdrive2_ABglobal: /opt/cray/pe/libsci/22.11.1.2/nvidia/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/pzdrive2_ABglobal: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libparmetis/libparmetis.so
EXAMPLE/pzdrive2_ABglobal: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libmetis/libmetis.so
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libcudart.so
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusolver.so
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublas.so
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libculibos.a
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublasLt.so
EXAMPLE/pzdrive2_ABglobal: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusparse.so
EXAMPLE/pzdrive2_ABglobal: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mpi_mp.so
EXAMPLE/pzdrive2_ABglobal: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/pzdrive2_ABglobal: EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable pzdrive2_ABglobal"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pzdrive2_ABglobal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/build: EXAMPLE/pzdrive2_ABglobal
.PHONY : EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/build

EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/clean:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE && $(CMAKE_COMMAND) -P CMakeFiles/pzdrive2_ABglobal.dir/cmake_clean.cmake
.PHONY : EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/clean

EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/depend:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120 /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_nv_0120/EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : EXAMPLE/CMakeFiles/pzdrive2_ABglobal.dir/depend

