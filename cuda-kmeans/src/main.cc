#include <iostream>
#include <string>
#include <unistd.h>
#include "kmeans.h"
#include "kmcuda.h"

using namespace std;

void test_kmcuda(data_t*, int, int);

int main(int argc, char** argv) {  
  char* filename = nullptr;
  int c = -1;
  int x = -1;
  int t = -1; 
  int s = -1;
  int iters = 1;
  int o;
  while ((o = getopt(argc, argv, "c:f:x:t:s:i:")) != -1) {
    switch (o) {  
      case 'c':
        c = atoi(optarg);
        break;
      case 'f':
        filename = optarg;
        break;
      case 'x':
        x = atoi(optarg);
        break;
      case 't':
        t = atoi(optarg);
        break;
      case 's':
        s = atoi(optarg);
        break;
      case 'i':
        iters = atoi(optarg);
        break;
      case '?':
        if (optopt == 'c' || optopt == 'f' || optopt == 'x')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
    }
  }

  // validate input
  if(filename == nullptr || c == -1 || x == -1 || t == -1) {
    cout << "Invalid program arguments" << endl;
    return 1;
  }

  for(int i = 0; i < iters; ++i) {
    data_t* data = extract_data(filename);
    if(x > 0) {
    	cuda_kmeans(x, c, data, t);
    }
    else {
    	test_kmcuda(data, c, s);
    }
  }
}

void test_kmcuda(data_t * d, int k, int s) { 
  float tolerance = 0.0001;
  KMCUDAInitMethod init = kmcudaInitMethodRandom;
  KMCUDADistanceMetric L2 = kmcudaDistanceMetricL2; 
  uint32_t samples_size = (uint32_t) d->numPoints;
  uint16_t features_size = (uint32_t) d->numAttrs;
  uint32_t clusters_size = (uint32_t) k;
  if (s == -1) {
    s = rand();
  }
  uint32_t seed = s;
  uint32_t device = 1;
  int32_t device_ptrs = -1;
  int32_t fp16x2 = 0;
  int32_t verbosity = 0;
  const float * samples = *(d->data);
  float * centroids = new float[k * d->numAttrs];
  uint32_t * assignments = new uint32_t[d->numPoints];

  printf("%d\t%d\t", 0, k);
  KMCUDAResult results = kmeans_cuda(init, nullptr, tolerance, 0.0, L2, samples_size, features_size, clusters_size, seed, device, device_ptrs, fp16x2, verbosity, samples, centroids, assignments, nullptr);
  printf("NA\tNA\tNA\n");
}
