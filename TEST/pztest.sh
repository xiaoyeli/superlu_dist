#!/bin/bash

# bash hint: == is for string comparisons, -eq is for numeric ones.

ofile=pztest.out			# output file
if [ -e $ofile ]; then
    rm -f $ofile
fi
echo "Double-complex testing output" > $ofile

MATRICES=(../EXAMPLE/cg20.cua)
NPROWS="1 2"
NPCOLS="1 3"
NVAL="9 19"
NRHS="1 3"
FILLRATIO="2 6"
# following are blocking parameters, see sp_ienv.c
RELAX="4 8"
SUPERSIZE="10 20"
MINGEMM="10000"

##
# Loop through all matrices ...
#
for mat in $MATRICES; do

  #--------------------------------------------
  # Test matrix types generated in LAPACK-style
  #--------------------------------------------
  if  [ "$mat" == "LAPACK" ]; then
      echo '== LAPACK test matrices' >> $ofile
      for n in $NVAL ; do
        for s in $NRHS ; do
	    echo '' >> $ofile
            echo 'n='$n 'nrhs='$s >> $ofile
	      mpiexec -n 2 pztest -r 1 -c 2 -x 4 -m 10 -b 5 -s 1 >> $ofile
        done
      done
  #--------------------------------------------
  # Test a specified sparse matrix
  #--------------------------------------------
  else
    echo '' >> $ofile
    echo '== sparse matrix:' $m >> $ofile
    for s in $NRHS; do
      for r in $NPROWS; do
	for c in $NPCOLS; do
	  np=$(($r*$c))
	  for b in $FILLRATIO; do
	    for x in $RELAX; do
	      for m in $SUPERSIZE; do
		echo '' >> $ofile
   	        echo "**-- nrhs = $s, process grid = $r X $c, fill $b, relax $x, max-super $m"
   	        echo "**-- nrhs = $s, process grid = $r X $c, fill $b, relax $x, max-super $m" >> $ofile
		mpiexec -n $np pztest -r $r -c $c -x $x -m $m -b $b -s 1 -f $mat >> $ofile
	      done
	    done
	  done
	done
      done
    done
  fi
done

