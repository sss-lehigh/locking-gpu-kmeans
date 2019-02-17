# locking-gpu-kmeans
The source code corresponding to our paper at PMAM '19 titled "Don't Forget About Synchronization! A Case Study of K-means on GPU"

Before running KMCUDA must be built. The source code has been altered to allow for timing of the update phase. It requires cmake.
After building the library file, move it to the lib folder under cuda-kmeans then update the library path. Build cuda-kmeans
using the make file in the source directory. There are some example dataset is the dataset folder, along with a script to
generate new synthetic datasets. 


