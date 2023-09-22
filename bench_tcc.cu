#include "bench_tcc.h"

#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cuda_fp16.h>
#include <complex>

#include "libtcc/Correlator.h"

// parameters required by TCC that are customizable
// NR_BITS can be 4, 8, 16
// only have support for 16 (FP16->FP32) so far
#define NR_BITS 16
#define NR_RECEIVERS_PER_BLOCK 32
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

#define checkCudaCall(function, ...) { \
    cudaError_t error = function; \
    if (error != cudaSuccess) { \
        std::cerr  << __FILE__ << "(" << __LINE__ << ") CUDA ERROR: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void float_to_half_kernel(const float* input, __half* output, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size) { 
        output[idx] = __float2half(input[idx]);
    }
}

void float_to_half(const float* input,  __half* output, size_t size, cudaStream_t stream)
{
    int nthreads;
    int nblocks;
    if (size < 1024)
        nthreads = size;
    else
        nthreads = 1024;
    nblocks = (size + nthreads - 1) / nthreads;
    
    float_to_half_kernel<<<nblocks,nthreads,0,stream>>>(input, output, size);
}

void showTccInfo(Parameters params) {
    std::cout << "\t================ TCC INFO ================\n";
    std::cout << "\tnpol:                 " << NR_BITS << "\n";
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
    Results result;
    std::cout << "Initialising & compiling TCC kernel with NVRTC...\n";

    cudaStream_t stream;
    std::complex<float>* input_d; // store fp32 input
    std::complex<__half> *samples_d; // typecast down to fp16
    std::complex<float> *visibilities_d;
    try {
        tcc::Correlator correlator(NR_BITS, params.nstation, params.nfrequency, params.nsample, params.npol, NR_RECEIVERS_PER_BLOCK);
        showTccInfo(params);

        checkCudaCall(cudaStreamCreate(&stream));
        checkCudaCall(cudaMalloc(&input_d, params.input_size * sizeof(std::complex<float>)));

        checkCudaCall(cudaMalloc(&samples_d, params.input_size * sizeof(__half)));
        checkCudaCall(cudaMalloc(&visibilities_d, params.output_size * sizeof(std::complex<float>)));
        checkCudaCall(cudaMemcpy(input_d, input_h, params.input_size * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

        // need to recast pointers as CUDA can't do std::complex<float> to std::complex<__half> on device
        float_to_half((float *)input_d, (__half *)samples_d, params.input_size * 2, stream);

        correlator.launchAsync((CUstream) stream, (CUdeviceptr) visibilities_d, (CUdeviceptr) samples_d);

        checkCudaCall(cudaDeviceSynchronize());

        checkCudaCall(cudaMemcpy(visibilities_h, visibilities_d, params.output_size * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));


        checkCudaCall(cudaFree(visibilities_d));
        checkCudaCall(cudaFree(samples_d));
        checkCudaCall(cudaStreamDestroy(stream));
    } catch(std::exception &error) { 
        std::cerr << error.what() << std::endl;
    }



    result.in_reorder_time = 0;
    result.compute_time = 0;
    result.out_reorder_time = 0;

    return result;
}

