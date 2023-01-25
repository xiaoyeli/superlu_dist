# CMake generated Testfile for 
# Source directory: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST
# Build directory: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pdtest_1x1_1_2_8_20_SP "/usr/bin/srun" "-n" "1" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x1_3_2_8_20_SP "/usr/bin/srun" "-n" "1" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x2_1_2_8_20_SP "/usr/bin/srun" "-n" "2" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x2_3_2_8_20_SP "/usr/bin/srun" "-n" "2" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x3_1_2_8_20_SP "/usr/bin/srun" "-n" "3" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x3_3_2_8_20_SP "/usr/bin/srun" "-n" "3" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "1" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x1_1_2_8_20_SP "/usr/bin/srun" "-n" "2" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x1_3_2_8_20_SP "/usr/bin/srun" "-n" "2" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x2_1_2_8_20_SP "/usr/bin/srun" "-n" "4" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x2_3_2_8_20_SP "/usr/bin/srun" "-n" "4" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x3_1_2_8_20_SP "/usr/bin/srun" "-n" "6" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x3_3_2_8_20_SP "/usr/bin/srun" "-n" "6" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "2" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x1_1_2_8_20_SP "/usr/bin/srun" "-n" "5" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x1_3_2_8_20_SP "/usr/bin/srun" "-n" "5" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x2_1_2_8_20_SP "/usr/bin/srun" "-n" "10" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x2_3_2_8_20_SP "/usr/bin/srun" "-n" "10" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x3_1_2_8_20_SP "/usr/bin/srun" "-n" "15" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x3_3_2_8_20_SP "/usr/bin/srun" "-n" "15" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/TEST/pdtest" "-r" "5" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;47;add_test;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/TEST/CMakeLists.txt;0;")
