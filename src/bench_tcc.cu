#include "bench_tcc.h"

#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cuda_fp16.h>
#include <complex>
#include <chrono>

#include "libtcc/Correlator.h"
#include "util.h"

// parameters required by TCC that are customizable
// NR_BITS can be 4, 8, 16
// only have support for 16 (FP16->FP32) so far
#define NR_BITS 16
#define NR_RECEIVERS_PER_BLOCK 32
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

#define TRANSPOSE_BLOCK_SIZE 512

#define checkCudaCall(function, ...) { \
    cudaError_t error = function; \
    if (error != cudaSuccess) { \
        std::cerr  << __FILE__ << "(" << __LINE__ << ") CUDA ERROR: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// [station][polarisation][time][frequency] -> [frequency][time / tpb][station][polarisation][tpb]
// need to cast input and output as float, as float-to-half conversion is not supported for complex types
// TODO: change so that number of channels can be > 1024 ... very important for MWAX with 6400 channels
__global__ void transpose_to_TCC_kernel(Parameters params, const float* input, __half* output) {
    int t, f, s, p;
    f = blockIdx.x*blockDim.x + threadIdx.x;
    t = blockIdx.y;
    p = blockIdx.z % params.npol;
    s = blockIdx.z / params.npol;
    
    // split into two time axes
    if(f < params.nfrequency) {
        int t0, t1;
        t0 = t / NR_TIMES_PER_BLOCK;
        t1 = t % NR_TIMES_PER_BLOCK;
        
        int in_idx = 2*(s*params.npol*params.nsample*params.nfrequency + p*params.nsample*params.nfrequency + t*params.nfrequency + f);
        int out_idx = 2*(f*params.nsample*params.nstation*params.npol + t0*params.nstation*params.npol*NR_TIMES_PER_BLOCK 
            + s*params.npol*NR_TIMES_PER_BLOCK + p*NR_TIMES_PER_BLOCK + t1);
        
        output[out_idx] = __float2half(input[in_idx]);     // real 
        output[out_idx+1] = __float2half(input[in_idx+1]); // complex
    }
}

inline void transpose_to_TCC(Parameters params, const std::complex<float>* input, std::complex<__half>* output, cudaStream_t stream) {
    // need to support > 1024 channels but still want to keep as minor axis
    if(params.nfrequency <= 1024) {
        dim3 block(params.nfrequency, 1, 1);
        dim3 grid(1, params.nsample, params.nstation*params.npol);
        transpose_to_TCC_kernel<<<grid, block, 0, stream>>>(params, (float*)input, (__half*)output);
    } else {
        dim3 block(TRANSPOSE_BLOCK_SIZE, 1, 1);
        dim3 grid(params.nfrequency / TRANSPOSE_BLOCK_SIZE, params.nsample, params.nstation*params.npol);
        transpose_to_TCC_kernel<<<grid, block, 0, stream>>>(params, (float*)input, (__half*)output);
    }
}

// [frequency][baseline][polarisation][polarisation] -> [baseline][frequency][polarisation*4]
// MWAX polarisation order: 
//      xx_real, xx_imag, yx_real, yx_imag, xy_real, xy_imag, yy_real, yy_imag
// TODO: test this against 3D kernel where polarisation is it's own thread index
__global__ void tcc_to_tri_kernel(Parameters params, const std::complex<float>* input, std::complex<float>* output) {
    int f, b, p;
    b = blockIdx.x*blockDim.x + threadIdx.x;
    f = blockIdx.y;
    int in_idx = f*params.nbaseline*params.npol*params.npol + b*params.npol*params.npol;
    int out_idx = b*params.nfrequency*params.npol*params.npol + f*params.npol*params.npol;

    if(in_idx < params.output_size && out_idx < params.output_size) {
    
        #pragma unroll
        for(p=0; p<params.npol*params.npol; ++p) {
            output[out_idx+p] = input[in_idx+p];
        }
    }
}

inline void tcc_to_tri(Parameters params, const std::complex<float>* input, std::complex<float>* output, cudaStream_t stream) { 
    dim3 block(1024);
    dim3 grid(params.nbaseline / 1024 + 1, params.nfrequency);
    tcc_to_tri_kernel<<<grid, block, 0, stream>>>(params, input, output);
}

void showTccInfo(Parameters params) {
    std::cout << "\t================ TCC INFO ================\n";
    std::cout << "\tnpol:                 " << params.npol << "\n";
    std::cout << "\tnstation:             " << params.nstation << "\n";
    std::cout << "\tnbaseline:            " << params.nbaseline << "\n";
    std::cout << "\tnfrequency:           " << params.nfrequency << "\n";
    std::cout << "\tnsamples:             " << params.nsample << "\n";
    std::cout << "\tcompute_type:         ";
    switch(NR_BITS) { 
        case 4:  std::cout << "INT4 multiply, INT32 accumulate\n"; break;
        case 8:  std::cout << "INT8 multiply, INT32 accumulate\n"; break;
        case 16: std::cout << "FP16 multiply,  FP32 accumulate\n"; break;
    }
    std::cout << "\t=============== EXTRA INFO ===============\n";
    std::cout << "\tinput_size:           " << params.input_size<< " (" << byteToMB(params.input_size*NR_BITS/8*sizeof(half)) << " MB)\n";
    std::cout << "\toutput_size:          " << params.output_size << " (" << byteToMB(params.output_size*NR_BITS/8*sizeof(float)) << " MB)\n";
    std::cout << "\tnreceivers_per_block: " << NR_RECEIVERS_PER_BLOCK << "\n";
    std::cout << "\tntime_per_block:      " << NR_TIMES_PER_BLOCK << "\n";
}

Results runTCC(Parameters params, const std::complex<float>* input_h, std::complex<float>* visibilities_h) {
    Results result = {0};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms;

    std::cout << "Initialising & compiling TCC kernel with NVRTC...\n";

    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));

    cudaStream_t stream;
    std::complex<float>* input_d; // store fp32 input
    std::complex<__half> *tcc_in_d; // typecast down to fp16
    std::complex<float> *tcc_out_d;
    std::complex<float> *tcc_reordered_d;
    std::complex<float> *tcc_reordered_h = (std::complex<float>*)malloc(params.output_size * sizeof(std::complex<float>)); 

    try {
        tcc::Correlator correlator(NR_BITS, params.nstation, params.nfrequency, params.nsample, params.npol, NR_RECEIVERS_PER_BLOCK);
        // showTccInfo(params);

        checkCudaCall(cudaStreamCreate(&stream));
        checkCudaCall(cudaMalloc(&input_d, params.input_size * sizeof(std::complex<float>)));
        checkCudaCall(cudaMalloc(&tcc_in_d, params.input_size * sizeof(std::complex<__half>)));
        checkCudaCall(cudaMalloc(&tcc_out_d, params.output_size * sizeof(std::complex<float>)));
        checkCudaCall(cudaMalloc(&tcc_reordered_d, params.output_size * sizeof(std::complex<float>)));
        checkCudaCall(cudaMemcpy(input_d, input_h, params.input_size * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

        cudaEventRecord(start);
        transpose_to_TCC(params, input_d, tcc_in_d, stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        result.in_reorder_time = time_ms / 1000;

        cudaEventRecord(start);
        correlator.launchAsync((CUstream) stream, (CUdeviceptr) tcc_out_d, (CUdeviceptr) tcc_in_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        result.compute_time = time_ms / 1000;

        checkCudaCall(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        tcc_to_tri(params, tcc_out_d, tcc_reordered_d, stream); // swap baseline and frequency with vanilla TCC
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        result.tri_reorder_time = time_ms / 1000;

        // don't include transfer in reorder time
        checkCudaCall(cudaMemcpy(tcc_reordered_h, tcc_reordered_d, params.output_size * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));
        
        typedef std::chrono::high_resolution_clock Clock;
        auto t0 = Clock::now();
        tri_to_mwax(params, tcc_reordered_h, visibilities_h);
        auto t1 = Clock::now();
        std::chrono::duration<float> elapsed = t1 - t0;
        result.mwax_time += elapsed.count();

        // Free allocated buffers
        checkCudaCall(cudaFree(input_d));
        checkCudaCall(cudaFree(tcc_in_d));
        checkCudaCall(cudaFree(tcc_out_d));
        checkCudaCall(cudaFree(tcc_reordered_d));
        free(tcc_reordered_h);

        checkCudaCall(cudaStreamDestroy(stream));
    } catch(std::exception &error) { 
        std::cerr << error.what() << std::endl;
    }

    return result;
}

