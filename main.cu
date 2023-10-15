#include <iostream>
#include <iomanip>
#include <complex>
#include <random>
#include <chrono>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include "bench_tcc.h"
#include "bench_mwax_tcc.h"
#include "bench_xgpu.h"
#include "bench_serial.h"
#include "util.h"

// #define DEBUG

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
    int idx = 0;
    std::cout.precision(3);
    for(int b = 0; b < params.nbaseline; b++) { 
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
            idx = (b+1)*params.nfrequency*params.npol*params.npol;
        }
    }  
    std::cout << "\n\n";
}

float rmsError(std::complex<float>* a1, std::complex<float>* a2, size_t N) {
    float abs_error;
    float rel_error;
    float sum = 0;
    int errors = 0;
    for(size_t i = 0; i < N; i++) {
        abs_error = std::abs(a1[i] - a2[i]);
        rel_error = abs_error / (std::max(std::abs(a1[i]), std::abs(a2[i]))); // case where one visibility is 0 but other is small
        if(abs_error > 0.5 && rel_error > 0.01) { 
            #ifdef DEBUG
            std::cout << "Error: " << a1[i] << " != " << a2[i] << " (" << i << ")\n";
            #endif
            errors++;
        }
        sum += abs_error*abs_error;
    }
    if(errors)
        std::cout << "\033[31m"; // red 
    else
        std::cout << "\033[32m"; // green
    std::cout << N - errors << " / " << N << " values correct\n";
    std::cout << "\033[0m"; // reset 

    return std::sqrt(sum / N);
}

void report(Parameters params, Results result) {
    float total_time = result.in_reorder_time + result.compute_time + result.tri_reorder_time
        + result.channel_avg_time + result.mwax_time;
    std::cout << "      Input reordering: " << result.in_reorder_time * 1000 << " ms\n";
    std::cout << "               Compute: " << result.compute_time * 1000 << " ms\n";
    std::cout << " Triangular reordering: " << result.tri_reorder_time * 1000 << " ms\n";
    // std::cout << "     Channel averaging: " << result.channel_avg_time * 1000 << " ms\n";
    std::cout << "       MWAX reordering: " << result.mwax_time * 1000 << " ms\n";
    std::cout << "                 Total: " << total_time * 1000 << " ms\n";

    if(result.compute_time != 0)
        std::cout << "         Compute FLOPS: " << (params.flop / result.compute_time) / 1e12 << " TOP/s\n";

    if(total_time != 0)
        std::cout << "           Total FLOPS: " << (params.flop / total_time) / 1e12 << " TOP/s\n";

}

void reportCSV(Parameters params, Results result, std::string filename) {
    std::ofstream file;
    file.open(filename, std::ios_base::app); // append 
    
    // .csv header
    // file << "nstation, nfrequency, ntime, npol, input reorder (ms),compute (ms),tri reorder (ms),"
        // << "channel avg (ms),mwax reorder (ms) total (ms),compute (TOPS),total (TOPS)\n";
    
    float total_time = result.in_reorder_time + result.compute_time + result.tri_reorder_time
        + result.channel_avg_time + result.mwax_time;

    file << params.nstation << ",";
    file << params.nfrequency << ",";
    file << params.nsample << ",";
    file << params.npol << ",";
    
    file << result.in_reorder_time * 1000 << ",";
    file << result.compute_time * 1000 << ",";
    file << result.tri_reorder_time * 1000 << ",";
    file << result.channel_avg_time * 1000 << ",";
    file << result.mwax_time * 1000 << ",";
    file << total_time * 1000 << "";

    if(result.compute_time != 0) {
        file << ((params.flop / result.compute_time) / 1e12) << ","; 
    } else  {
        file << "0,";
    }

    if(total_time != 0) {
        file << (params.flop / total_time) / 1e12;
    } else {
        file << "0";
    }
    
    file << "\n";
    file.close();
}

Parameters getParams(int argc, char *argv[]) {
    Parameters params;

    // defaults
    params.npol = 2;
    params.nfrequency = 50;
    params.nstation = 64;
    params.nsample = 16;
    params.npol = 2;
    params.verify = false;
    params.write_csv = false;
    params.snapshot = false;

    for(int opt; (opt = getopt(argc, argv, "f:n:t:p:vcs")) >= 0;) {
        switch (opt) {
            case 'f':   
                params.nfrequency = atoi(optarg);
                break;  
            case 'n':   
                params.nstation = atoi(optarg);
                break;  
            case 't':   
                params.nsample = atoi(optarg);
                break;  
            case 'p':
                params.npol = atoi(optarg);
                break;
            case 'v':
                params.verify = true;
                break;
            case 'c':
                params.write_csv = true;
                break;
            case 's':
                params.snapshot = true;
                break;
        }
    }    
    params.nbaseline = (params.nstation * (params.nstation + 1)) / 2;
    params.input_size = params.nstation * params.nsample * params.nfrequency * params.npol;
    params.output_size = params.nbaseline * params.nfrequency * params.npol * params.npol;
    params.flop = 8ULL * params.nstation * (params.nstation / 2) * params.npol * params.npol * params.nfrequency * params.nsample;

    return params;
}

// [station][polarisation][time][frequency]
void createTestVector(std::complex<float>* samples, size_t N) { 
    memset(samples, 0, N * sizeof(std::complex<float>));
    samples[0] = {1, 1};
}


int main (int argc, char *argv[]) {
    Parameters params = getParams(argc, argv);

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

    if(params.verify) {
        Results serial_result = runSerial(params, samples, visibilities_serial);
        if (params.snapshot) printOutputSnapshot(params, visibilities_serial);
        report(params, serial_result);
        if(params.write_csv) reportCSV(params, serial_result, "results/serial.csv");
    }

    memset(visibilities_gpu, 0, params.output_size * sizeof(std::complex<float>));
    Results xgpu_result = runXGPU(params, samples, visibilities_gpu);
    if(params.verify) {
        error_xgpu = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
        std::cout << "Total XGPU error (rms): " << error_xgpu << "\n";
    }
    if(params.snapshot) printOutputSnapshot(params, visibilities_gpu);
    if(params.write_csv) reportCSV(params, xgpu_result, "results/mwax.csv");
    report(params, xgpu_result);

    memset(visibilities_gpu, 0, params.output_size * sizeof(std::complex<float>));
    Results tcc_result = runTCC(params, samples, visibilities_gpu);
    if(params.verify) { 
        error_tcc = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
        std::cout << " Total TCC error (rms): " << error_tcc << "\n";
    }
    if(params.snapshot) printOutputSnapshot(params, visibilities_gpu);
    if(params.write_csv) reportCSV(params, tcc_result, "results/tcc_v1.csv");
    report(params, tcc_result);

    memset(visibilities_gpu, 0, params.output_size * sizeof(std::complex<float>));
    Results mwax_tcc_result = runMWAXTCC(params, samples, visibilities_gpu);
    if(params.verify) {
        error_mwax_tcc = rmsError(visibilities_serial, visibilities_gpu, params.output_size);
        std::cout << "Total TCC2 error (rms): " << error_mwax_tcc << "\n";
    }
    if(params.snapshot) printOutputSnapshot(params, visibilities_gpu);
    if(params.write_csv) reportCSV(params, mwax_tcc_result, "results/tcc_v2.csv");
    report(params, mwax_tcc_result);


    delete samples;
    delete visibilities_serial;
    delete visibilities_gpu;
}

