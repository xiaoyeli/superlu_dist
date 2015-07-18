##PBS -q regular
###PBS -q premium
#PBS -q debug
#PBS -l mppwidth=8
#PBS -l mppnppn=8
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o test.out
#PBS -e test.err
#PBS -m e

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

aprun -n 8 -N 8 -S 4 ./pddrive -r 2 -c 4  /scratch1/scratchdirs/xiaoye/kkt_power.mtx


