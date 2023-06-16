#!/bin/sh
set -e

export RED="\033[31;1m"
export BLUE="\033[34;1m"
printf "${BLUE} GC; Entered tests file:\n"

export DATA_FOLDER=$TRAVIS_BUILD_DIR/EXAMPLE
export EXAMPLE_FOLDER=$TRAVIS_BUILD_DIR/build/EXAMPLE
export TEST_FOLDER=$TRAVIS_BUILD_DIR/build/TEST

case "${TEST_NUMBER}" in
1)  mpirun "-n" "1" "$TEST_FOLDER/pdtest" "-r" "1" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
2)  mpirun "-n" "1" "$TEST_FOLDER/pdtest" "-r" "1" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
3)  mpirun "-n" "3" "$TEST_FOLDER/pdtest" "-r" "1" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
4)  mpirun "-n" "3" "$TEST_FOLDER/pdtest" "-r" "1" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
5)  mpirun "-n" "2" "$TEST_FOLDER/pdtest" "-r" "2" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
6)  mpirun "-n" "2" "$TEST_FOLDER/pdtest" "-r" "2" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
7)  mpirun "-n" "6" "$TEST_FOLDER/pdtest" "-r" "2" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
8)  mpirun "-n" "6" "$TEST_FOLDER/pdtest" "-r" "2" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "$DATA_FOLDER/g20.rua" ;;
9)  mpirun "-n" "4" "$EXAMPLE_FOLDER/pddrive1" "-r" "2" "-c" "2" "$DATA_FOLDER/big.rua" ;;
10) mpirun "-n" "4" "$EXAMPLE_FOLDER/pddrive2" "-r" "2" "-c" "2" "$DATA_FOLDER/big.rua" ;;
11) mpirun "-n" "4" "$EXAMPLE_FOLDER/pddrive3" "-r" "2" "-c" "2" "$DATA_FOLDER/big.rua" ;;
12) mpirun "-n" "4" "$EXAMPLE_FOLDER/pzdrive1" "-r" "2" "-c" "2" "$DATA_FOLDER/cg20.cua" ;;
13) mpirun "-n" "4" "$EXAMPLE_FOLDER/pzdrive2" "-r" "2" "-c" "2" "$DATA_FOLDER/cg20.cua" ;;
14) mpirun "-n" "4" "$EXAMPLE_FOLDER/pzdrive3" "-r" "2" "-c" "2" "$DATA_FOLDER/cg20.cua" ;;
15) mpirun "-n" "4" "$EXAMPLE_FOLDER/pddrive_ABglobal" "-r" "2" "-c" "2" "$DATA_FOLDER/big.rua" ;;
*) printf "${RED} ###GC: Unknown test\n" ;;
esac
