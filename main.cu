#include <iostream>
#include <complex>
#include <random>

#include "bench_tcc.h"
#include "bench_xgpu.h"
#include "util.h"

#define checkCudaCall(function, ...) { \
    cudaError_t error = function; \
    if (error != cudaSuccess) { \
        std::cerr  << __FILE__ << "(" << __LINE__ << ") CUDA ERROR: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// fill N complex samples into std::complex<float> array
void createRandomSamples(std::complex<float>* samples, size_t N) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 5.0);
    for(int i=0; i<N; ++i) { 
        samples[i] = {distribution(generator), distribution(generator)};
    }
}

void createTestVector(std::complex<float>* samples, size_t N) { 
    memset(samples, 0, N * sizeof(std::complex<float>));
    samples[0] = {1, 2};
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
    cudaSetDevice(0); // combine the CUDA runtime API and CUDA driver API
    cudaFree(0);

    std::cout << "Generating random complex samples...\n";
    // not very good C++, but we want easy compatibility with C libraries so keep raw pointers
    std::complex<float>* samples_h = new std::complex<float>[params.input_size];
    std::complex<float>* visibilities_h = new std::complex<float>[params.output_size];
    // createRandomSamples(samples_h, params.input_size);
    createTestVector(samples_h, params.input_size);

    std::cout << "First 10 input samples:\n";
    for(int i = 0; i < 10; ++i) { 
        std::cout << samples_h[i] << "\n";
    }


    // Results xgpu_result = runXGPU(params, samples_h, visibilities_h);
    Results tcc_result = runTCC(params, samples_h, visibilities_h);


    std::cout << "First 10 output samples:\n";
    for(int i = 0; i < 10; ++i) { 
        std::cout << visibilities_h[i] << "\n";
    }    

    delete samples_h;
    delete visibilities_h;
}

