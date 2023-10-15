#!/bin/bash

NFREQUENCY=50
NTIME=128
NPOL=2
NSTATION=64

# remove previous results
rm results/*.csv

for NTIME in {16..256..16} 
do
    make NSTATION=$NSTATION NFREQUENCY=$NFREQUENCY NTIME=$NTIME NPOL=$NPOL
    ./main -n $NSTATION -f $NFREQUENCY -t $NTIME -p $NPOL -c -v
done