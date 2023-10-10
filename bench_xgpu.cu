#include "bench_xgpu.h"

#include <iostream>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <cuComplex.h>

#include "xgpu.h"
#include "xgpu_info.h"
#include "util.h"


#define NR_BITS 16
#define NR_CHANNELS 50
#define NR_POLARIZATIONS 2
#define NR_SAMPLES 16
#define NR_RECEIVERS 64
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 32
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

#define INPUT_SIZE (NR_RECEIVERS * NR_CHANNELS * NR_SAMPLES * NR_POLARIZATIONS)
#define OUTPUT_SIZE (NR_BASELINES * NR_CHANNELS * NR_POLARIZATIONS * NR_POLARIZATIONS)

typedef struct XGPUInternalContextPartStruct {
  int device;
  ComplexInput *array_d[2];
  Complex *matrix_d;
} XGPUInternalContextPart;

#define checkCudaCall(function, ...) { \
    cudaError_t error = function; \
    if (error != cudaSuccess) { \
        std::cerr  << __FILE__ << "(" << __LINE__ << ") CUDA ERROR: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

inline void checkXGPUCall(int xgpu_error) { 
// void checkXGPUCall(int xgpu_error) { 
    if(xgpu_error != XGPU_OK) {
        std::cerr << __FILE__ << "(" << __LINE__ << ") xGPU error (code " << xgpu_error << ")\n";
        // xgpuFree(xgpu_context);
        exit(1);
    }
}


// [station][polarisation][time][channel] -> [time][channel][station][polarization](complexity)
__global__ void transpose_to_xGPU_kernel(const cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns)
{
    // blockIdx.x is column (input time-freq)
    // threadIdx.x is row (input signal path (station-pol))
  
    // (time-freq * (nstation*npol)) + station-pol        (time-freq) + (station-pol * (nsamples*nfrequency))
    output[(blockIdx.x * rows) + threadIdx.x] = input[blockIdx.x + (threadIdx.x * columns)];

    return;
}

/* called with:
 * mwax_transpose_to_xGPU((float complex *)ctx->d_chan, (float complex *)out_pointer, (unsigned)ctx->num_xgpu_signal_paths, 
 *     (unsigned)ctx->num_retained_samps_per_block_per_ant, ctx->stream1);
 * 
 * rows = nstation * npol
 * columns = nsamples * nfrequency
 */ 
int transpose_to_xGPU(cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns) {
    int nblocks = (int)columns;
    int nthreads = (int)rows;

    std::cout << "nblocks = " << nblocks << "\n";
    std::cout << "nthreads = " << nthreads << "\n";

    transpose_to_xGPU_kernel<<<nblocks,nthreads,0>>>(input,output,rows,columns);

    return 0;
}

/************************************************************************************/
/* Re-order an xGPU triangular order visibility set to MWAX order                   */
/* It's hard to explain my algorithm - it's based on visual inspcection of xGPU's   */
/* output triangle versus what we want for MWAX order - including swapping from     */
/* A vs B to B vs A.                                                                */
/************************************************************************************/
// "MWAX order" is [time][baseline][channel][polarisation]
void xgpu_tri_to_mwax(Parameters *params, void * tri_buffer)
{
    int i, j, f;

    int num_tiles = params->nstation;
    
    int column_length = num_tiles;
    int first_baseline_step = 1;
    int baseline_step = first_baseline_step;
    int first_baseline_index = 0;
    int baseline_index = first_baseline_index;
    int visibility_index;
    float *float_mwax_buffer = (float*)malloc(params->output_size * sizeof(Complex));

    // this is bad. very bad. the pointer is manually incremented and in this implementation we want to 
    // free at the end. so keep a copy of the original pointer
    // how did we get here...
    float *float_mwax_buffer_orig = float_mwax_buffer;
    Complex *complex_tri_buffer = (Complex*)tri_buffer;

    for (i=0; i<num_tiles; i++)  // all columns of xGPU's upper triangle
    {
        for (j=0; j<column_length; j++)   // number of rows in each output column decrements (lower triangle) (j=0 is the auto)
        {
            for (f=0; f<params->nfrequency; f++)
            {
                // visibility_index = 4*baseline_index + f*ctx->num_xgpu_visibilities_per_chan;  // x4 for 4 xpols
                visibility_index = params->npol*params->npol*baseline_index + f*params->nbaseline*params->npol*params->npol;
                // if(float_mwax_buffer[1] == 5)  {
                //     std::printf("\txgpu_tri_to_mwax | i: %d  j: %d  v: %d  b: %d  f: %d \n", i, j, visibility_index, baseline_index, f);
                // }
                // conjugate and swap xy/yx because swapping tile A / tile B in going from xGPU triangular format to MWAX format
                *float_mwax_buffer++ = complex_tri_buffer[visibility_index].real;      // xx real
                *float_mwax_buffer++ = -complex_tri_buffer[visibility_index].imag;     // xx imag - conjugate
                *float_mwax_buffer++ = complex_tri_buffer[visibility_index + 2].real;  // yx real
                *float_mwax_buffer++ = -complex_tri_buffer[visibility_index + 2].imag; // yx imag - conjugate
                *float_mwax_buffer++ = complex_tri_buffer[visibility_index + 1].real;  // xy real
                *float_mwax_buffer++ = -complex_tri_buffer[visibility_index + 1].imag; // xy imag - conjugate
                *float_mwax_buffer++ = complex_tri_buffer[visibility_index + 3].real;  // yy real
                *float_mwax_buffer++ = -complex_tri_buffer[visibility_index + 3].imag; // yy imag - conjugate
            }
            baseline_index += baseline_step;
            baseline_step++;
        }
        column_length--;
        first_baseline_step++;
        baseline_step = first_baseline_step;
        first_baseline_index += first_baseline_step;
        baseline_index = first_baseline_index;
    }
    memcpy(complex_tri_buffer, float_mwax_buffer_orig, params->output_size*sizeof(Complex));
    free(float_mwax_buffer_orig);

    return;
}
    
void showxgpuInfo(XGPUInfo xgpu_info) {
    std::cout << "\t=============== XGPU INFO ================\n";
    std::cout << "\tnpol:               " << xgpu_info.npol << "\n";
    std::cout << "\tnstation:           " << xgpu_info.nstation << "\n";
    std::cout << "\tnbaseline:          " << xgpu_info.nbaseline << "\n";
    std::cout << "\tnfrequency:         " << xgpu_info.nfrequency << "\n";
    std::cout << "\tntime:              " << xgpu_info.ntime << "\n";
    std::cout << "\tntimepipe:          " << xgpu_info.ntimepipe << "\n";

    std::cout << "\tcompute_type        ";
    switch(xgpu_info.compute_type) {
        case XGPU_INT8:    std::cout << "INT8 multiply, INT32 accumulate\n"; break;
        case XGPU_FLOAT32: std::cout << "FP32 mulitply,  FP32 accumulate\n"; break;
        default:           std::cout << "<unknown type code: " << xgpu_info.compute_type << ">\n";
    }
    std::cout << "\t=============== EXTRA INFO ===============\n";
    std::cout << "\tinput_type          ";
    switch(xgpu_info.input_type) {
        case XGPU_INT8:    std::cout << "INT8\n"; break;
        case XGPU_INT32:   std::cout << "INT32\n"; break;
        case XGPU_FLOAT32: std::cout << "FP32\n"; break;
        default:           std::cout << "<unknown type code: " << xgpu_info.input_type << ">\n";
    }
    std::cout << "\tvecLength:          " << xgpu_info.vecLength << " (" << byteToMB(xgpu_info.vecLength*sizeof(ComplexInput)) << " MB)\n";
    std::cout << "\tvecLengthPipe:      " << xgpu_info.vecLengthPipe << "\n";
    std::cout << "\tmatLength:          " << xgpu_info.matLength << " (" << byteToMB(xgpu_info.matLength*sizeof(Complex)) << " MB)\n";
    std::cout << "\ttriLength:          " << xgpu_info.triLength << " (" << byteToMB(xgpu_info.triLength*sizeof(Complex)) << " MB)\n";
    std::cout << "\tmatrix_order:       ";
    switch(xgpu_info.matrix_order) {
        case TRIANGULAR_ORDER:               std::cout << "triangular\n"; break;
        case REAL_IMAG_TRIANGULAR_ORDER:     std::cout << "real imaginary triangular\n"; break;
        case REGISTER_TILE_TRIANGULAR_ORDER: std::cout << "register tile triangular\n"; break;
        default: printf("<unknown order code: %d>\n", xgpu_info.matrix_order);
    }

    std::cout << "\tshared_atomic_size: " << xgpu_info.shared_atomic_size << "\n";
    std::cout << "\tcomplex_block_size: " << xgpu_info.complex_block_size << "\n";
}

Results runXGPU(Parameters params, std::complex<float>* samples_h, std::complex<float>* visibilities_h) {
    Results result = {0, 0, 0};
    int device = 0;
    typedef std::chrono::high_resolution_clock Clock;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms;


    // xGPU has two input buffers, assume this is for read/write ping pong style?
    ComplexInput *input_d;  // initial copy of host samples (before reorder)
    ComplexInput *array_d0; // xGPU buffer holding 1st half of input data

    Complex *matrix_d;
    Complex *matrix_h = (Complex*)malloc(params.output_size * sizeof(Complex)); 

    // allocate GPU X-engine memory
    std::cout << "Initialising XGPU...\n";

    XGPUInfo xgpu_info;
    xgpuInfo(&xgpu_info); // get xGPU info from library
    showxgpuInfo(xgpu_info);

    // check that compiled parameters are equal to the target runtime parameters
    assert(xgpu_info.npol == params.npol &&  "xGPU npol does not match");
    assert(xgpu_info.nstation == params.nstation &&  "xGPU nstation does not match");
    assert(xgpu_info.nfrequency == params.nfrequency && "xGPU nfrequency does not match");
    assert(xgpu_info.ntime == params.nsample &&       "xGPU npol does not match");
    assert(xgpu_info.nbaseline == params.nbaseline && "xGPU npol does not match");
    assert(xgpu_info.vecLength == params.input_size &&   "xGPU vecLength does not match");
    assert(xgpu_info.triLength == params.output_size &&  "xGPU triLength does not match");
    // xgpu_info.matLength will be different because of REGISTER_TILE_TRIANGULAR_ORDER

    checkCudaCall(cudaMalloc(&input_d, params.input_size * sizeof(ComplexInput)));
    checkCudaCall(cudaMemcpy(input_d, samples_h, params.input_size * sizeof(ComplexInput), cudaMemcpyHostToDevice));

    XGPUContext xgpu_ctx;
    xgpu_ctx.array_h = NULL; // NOT USED IN MWAX: host input array
    xgpu_ctx.matrix_h = NULL; // USED IN MWAX: results from channel averaging, largely reduced size
    checkXGPUCall(xgpuInit(&xgpu_ctx, device)); // allocates all internal buffers
    XGPUInternalContextPart *xgpuInternalPointer = (XGPUInternalContextPart *)xgpu_ctx.internal;

    // set device pointers equal to buffers created by xGPU
    array_d0 = xgpuInternalPointer->array_d[0]; // location of the 1st xGPU input array, for telling the pre-correlation code where to write results
    matrix_d = xgpuInternalPointer->matrix_d;   // the xGPU output matrix, for use in frequency averaging and fetching of visibilities
    
    xgpuClearDeviceIntegrationBuffer(&xgpu_ctx);

    // device function!
    cudaEventRecord(start);
    transpose_to_xGPU((cuFloatComplex*) input_d, (cuFloatComplex*) array_d0 , params.nstation*params.npol, params.nfrequency*params.nsample);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    result.in_reorder_time = time_ms / 1000;

    checkXGPUCall(xgpuCudaXengine(&(xgpu_ctx), SYNCOP_SYNC_COMPUTE));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    result.compute_time = time_ms / 1000;

    checkCudaCall(cudaMemcpy(visibilities_h, matrix_d, params.output_size * sizeof(Complex), cudaMemcpyDeviceToHost));

    xgpu_tri_to_mwax(&params, visibilities_h);

    xgpuFree(&xgpu_ctx);

    return result;
}
