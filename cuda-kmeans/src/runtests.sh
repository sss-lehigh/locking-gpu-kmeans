#!/bin/bash

export LD_LIBRARY_PATH=../lib/

ALGO=0
THREADS=1024
ITERS=3

#for j in `seq 1 8`
#do 
CLUSTERS=2
for i in `seq 1 15`
do
#	./cuda_kmeans -f $1 -c $CLUSTERS -x 0 -t $THREADS -i$ITERS
#	./cuda_kmeans -f $1 -c $CLUSTERS -x 1 -t $THREADS -i$ITERS
#	./cuda_kmeans -f $1 -c $CLUSTERS -x 2 -t $THREADS -i$ITERS
#	./cuda_kmeans -f $1 -c $CLUSTERS -x 3 -t $THREADS -i$ITERS
	./cuda_kmeans -f $1 -c $CLUSTERS -x 4 -t $THREADS -i$ITERS
	CLUSTERS=$(($CLUSTERS * 2))
done
#ALGO=$(($ALGO + 1))
#done
