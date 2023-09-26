#include <iostream>
#include <iomanip>
#include <complex>
#include <random>

#include "bench_tcc.h"
#include "bench_xgpu.h"
#include "bench_serial.h"
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

// [station][polarisation][time][frequency]
void createTestVector(std::complex<float>* samples, size_t N) { 
    memset(samples, 0, N * sizeof(std::complex<float>));
    samples[0] = {1, 1};
    samples[1] = {1, 2};
}

void printOutputSnapshot(Parameters params, std::complex<float>* data) {
    int idx = 0;
    std::cout.precision(3);
    for(int b = 0; b < params.nbaseline; b++) { 
        int ant1 = (int)(std::floor(-0.5+std::sqrt(0.25+2*b)));
        int ant2 = (int)(b - ant1*(ant1+1)/2);
        std::printf("Baseline: %d (%d,%d)\n", b, ant1, ant2);
        std::printf("\t    pol |           xx          |           xy          |           yx          |           yy          |\n");
        for(int f = 0; f < params.nfrequency; f++) {
            std::printf("\tch %4d |", f);
            for(int p = 0; p < params.npol*params.npol; p++) {
                std::printf("(%+.3e,%+.3e)|", std::real(data[idx]), std::imag(data[idx]));
                idx++;
            }
            std::printf("\n");
            if(f==1) {
                std::printf("\t...\n");
                f=params.nfrequency-2;
            }
        }
        if(b==3) {
            std::printf("...\n");
            b=params.nbaseline-2;
        }
    }  
    std::cout << "\n\n";
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

    std::cout << "Generating random complex samples...\n";
    // not very good C++, but we want easy compatibility with C libraries so keep raw pointers
    std::complex<float>* samples = new std::complex<float>[params.input_size];
    std::complex<float>* visibilities = new std::complex<float>[params.output_size];
    
    // data in [antenna][polarisation][time][channel]
    createRandomSamples(samples, params.input_size);
    // createTestVector(samples, params.input_size);

    std::cout << "First 10 input samples:\n";
    for(int i = 0; i < 10; ++i) { 
        std::cout << samples[i] << "\n";
    }

    Results serial_result = runSerial(params, samples, visibilities);
    printOutputSnapshot(params, visibilities);
    memset(visibilities, 0, params.output_size * sizeof(std::complex<float>));

    Results xgpu_result = runXGPU(params, samples, visibilities);

    printOutputSnapshot(params, visibilities);

    memset(visibilities, 0, params.output_size * sizeof(std::complex<float>));
    Results tcc_result = runTCC(params, samples, visibilities);

    printOutputSnapshot(params, visibilities);

    delete samples;
    delete visibilities;
}

