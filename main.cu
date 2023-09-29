#include <iostream>
#include <iomanip>
#include <complex>
#include <random>
#include <chrono>
#include <cstring>

#include "bench_tcc.h"
#include "bench_mwax_tcc.h"
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
    // generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    generator.seed(2);
    std::normal_distribution<float> distribution(0.0, 5.0);
    for(int i=0; i<N; ++i) { 
        samples[i] = {distribution(generator), distribution(generator)};
    }
}

void printOutputSnapshot(Parameters params, std::complex<float>* data) {
    #ifdef DEBUG
    int idx = 0;
    std::cout.precision(3);
    for(int b = 0; b < params.nbaseline; b++) { 
        // int ant2 = (int)(std::floor(-0.5+std::sqrt(0.25+2*b)));
        // int ant1 = (int)(b - ant2*(ant2+1)/2);
        // std::printf("Baseline: %d (%d,%d)\n", b, ant1, ant2);
        std::printf("Baseline: %d \n", b);
        std::printf("       idx |  ch |           xx          |           xy          |           yx          |           yy          |\n");
        for(int f = 0; f < params.nfrequency; f++) {
            std::printf("%10d |%4d |", idx, f);
            for(int p = 0; p < params.npol*params.npol; p++) {
                std::printf("(%+.3e,%+.3e)|", std::real(data[idx]), std::imag(data[idx]));
                idx++;
            }
            std::printf("\n");
            if(f==2) {
                std::printf("        ...\n");
                f=params.nfrequency-3;
                idx = (b*params.nfrequency + (f+1))*params.npol*params.npol;
            }
        }
        if(b==2) {
            std::printf("...\n");
            b=params.nbaseline-2;
            // b=2045;
            idx = (b+1)*params.nfrequency*params.npol*params.npol;
        }
    }  
    std::cout << "\n\n";
    #endif
}

float rmsError(std::complex<float>* a1, std::complex<float>* a2, size_t N) {
    float abs_error;
    float rel_error;
    float sum = 0;
    int errors = 0;
    for(size_t i = 0; i < N; i++) {
        abs_error = std::abs(a1[i] - a2[i]);
        rel_error = abs_error / (std::max(std::abs(a1[i]), std::abs(a2[i]))); // case where one visibility is 0 but other is small
        if(abs_error > 1 && rel_error > 0.1) { 
            #ifdef DEBUG
            // std::cout << "Error: " << a1[i] << " != " << a2[i] << " (" << i << ")\n";
            // return 0;
            #else
            errors++;
            #endif
        }
        sum += abs_error*abs_error;
    }
    std::cout << N - errors << " / " << N << " values correct\n";
    return std::sqrt(sum / N);
}

// TODO: add support for .csv output
void report(Parameters params, Results result) {
    float total_time = result.in_reorder_time + result.compute_time + result.out_reorder_time;
    std::cout << "\t Input reordering: " << result.in_reorder_time * 1000 << " ms\n";
    std::cout << "\t          Compute: " << result.compute_time * 1000 << " ms\n";
    std::cout << "\tOutput reordering: " << result.out_reorder_time * 1000 << " ms\n";
    std::cout << "\t            Total: " << total_time * 1000 << " ms\n";

    if(result.compute_time != 0)
        std::cout << "\t    Compute FLOPS: " << (params.flop / result.compute_time) / 1000000000 << " GFLOP/s\n";

    if(total_time != 0)
        std::cout << "\t      Total FLOPS: " << (params.flop / total_time) / 1000000000 << " GFLOP/s\n";

}

// [station][polarisation][time][frequency]
void createTestVector(std::complex<float>* samples, size_t N) { 
    memset(samples, 0, N * sizeof(std::complex<float>));
    samples[0] = {1, 1};
    // samples[0*(50*16*2) + 49*(16*2)] = {1, 2};
    samples[49] = {1, 2};
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
    params.flop = 8 * params.nstation * params.nstation / 2 * params.npol * params.npol * params.nfrequency * params.nsample;

    float error_xgpu = 0;
    float error_tcc = 0;
    float error_mwax_tcc = 0;

    std::cout << "Initialising CUDA...\n";

    std::cout << "Generating random complex samples...\n";
    std::complex<float>* samples = new std::complex<float>[params.input_size];
    std::complex<float>* visibilities_serial = new std::complex<float>[params.output_size];
    std::complex<float>* visibilities_gpu = new std::complex<float>[params.output_size];
    
    // input data arranged in [station][polarisation][time][channel]
    createRandomSamples(samples, params.input_size);
    // createTestVector(samples, params.input_size);

    std::cout << "First 10 input samples:\n";
    for(int i = 0; i < 10; ++i) { 
        std::cout << samples[i] << "\n";
    }

    Results serial_result = runSerial(params, samples, visibilities_serial);
    printOutputSnapshot(params, visibilities_serial);
    report(params, serial_result);

    Results xgpu_result = runXGPU(params, samples, visibilities_gpu);
    error_xgpu = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
    printOutputSnapshot(params, visibilities_gpu);
    report(params, xgpu_result);


    memset(visibilities_gpu, 0, params.output_size * sizeof(std::complex<float>));
    Results tcc_result = runTCC(params, samples, visibilities_gpu);
    error_tcc = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
    printOutputSnapshot(params, visibilities_gpu);
    report(params, tcc_result);

    memset(visibilities_gpu, 0, params.output_size * sizeof(std::complex<float>));
    Results mwax_tcc_result = runMWAXTCC(params, samples, visibilities_gpu);
    std::cout << mwax_tcc_result.in_reorder_time << std::endl;
    error_mwax_tcc = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
    printOutputSnapshot(params, visibilities_gpu);
    report(params, mwax_tcc_result);

    std::cout << "XGPU error = " << std::scientific << error_xgpu << "\n";
    std::cout << "TCC error = " << std::scientific << error_tcc << "\n";
    std::cout << "MWAX_TCC error = " << std::scientific << error_mwax_tcc << "\n";

    delete samples;
    delete visibilities_serial;
    delete visibilities_gpu;
}

