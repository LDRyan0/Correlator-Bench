#include <iostream>
#include <complex>

#include "bench_tcc.cu"
#include "bench_xgpu.cu"

#include "util.h"

inline void checkCudaCall(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}

// fill N complex samples into std::complex<float> array
void createSamples(std::complex<float>* samples, size_t N) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 5.0);
    for(int i=0; i<N; ++i) { 
        samples[i] = {distribution(generator), distribution(generator)};
    }
}

int main () {
    Parameters params;
    params.npol = 2;
    params.nstation = 64;
    params.nsample = 16;
    params.nfrequency = 50;
    params.nbaseline = (params.nstation * (params.nstation + 1)) / 2;
    params.input_size = params.nstation * params.nsample * params.nfrequency * params.npol;
    params.output_size = params.nbaseline * params.nfrequency * params.npol * params.npol;

    std::cout << "Initialising CUDA...\n";
    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));

    std::cout << "Generating complex samples...\n";
    // not very good C++, but we're using C libraries so keep raw pointers
    std::complex<float> samples[INPUT_SIZE];
    createSamples(samples, INPUT_SIZE);
    
    runTCC(params);
    runXGPU(params);
}

