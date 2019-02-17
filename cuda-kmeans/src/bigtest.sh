export LD_LIBRARY_PATH=../lib/

ALGO=0
THREADS=1024

for i in `seq 1 9`
do
./cuda_kmeans -c5000 -f input/random-n300000-d408-c5000.txt -x $ALGO -t $THREADS
ALGO=$(($ALGO+1))
done
