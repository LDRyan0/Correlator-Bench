#include "bench_xgpu.h"

#include <iostream>
#include <cassert>
#include <cuda.h>
#include "cuComplex.h"

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
extern "C"
int transpose_to_xGPU(cuFloatComplex* input, cuFloatComplex* output, unsigned rows, unsigned columns) {
    int nblocks = (int)columns;
    int nthreads = (int)rows;

    std::cout << "nblocks = " << nblocks << "\n";
    std::cout << "nthreads = " << nthreads << "\n";

    transpose_to_xGPU_kernel<<<nblocks,nthreads,0>>>(input,output,rows,columns);

    return 0;
}

/******************************************************************************/
/* Re-order a register-tile order visibility set to triangular order          */
/* (duplicate of xgpuReorderMatrix() from xGPU cpu_util.c but using shortened */
/*  matLength from channel averaging)                                         */
    /******************************************************************************/
__host__ void xgpu_reg_to_tri(Parameters *params, void *reg_buffer) {
    int f, i, rx, j, ry, pol1, pol2;
    int reg_index;
    int tri_index;
    float *float_reg_buffer = (float *)reg_buffer;
    
    Complex *complex_tri_buffer = (Complex *)malloc(params->output_size * sizeof(Complex));
    memset(complex_tri_buffer, '0', params->output_size);

    int matLength = params->nfrequency * ((params->nstation/2+1)*(params->nstation/4)*params->npol*params->npol*4);

    for(f=0; f<params->nfrequency; f++) {
        for(i=0; i<params->nstation/2; i++) {
            for (rx=0; rx<2; rx++) {
                for (j=0; j<=i; j++) {
                    for (ry=0; ry<2; ry++) {
                        int k = f*(params->nstation+1)*(params->nstation/2) + (2*i+rx)*(2*i+rx+1)/2 + 2*j+ry;
                        int l = f*4*(params->nstation/2+1)*(params->nstation/4) + (2*ry+rx)*(params->nstation/2+1)*(params->nstation/4) + i*(i+1)/2 + j;
                        for (pol1=0; pol1<params->npol; pol1++) {
                            for (pol2=0; pol2<params->npol; pol2++) {
                                // [frequency][baseline][polarisation][polarisation]
                                // k*npol*npol + pol1*npol + pol2
                                // [k][pol1][pol2]
                                tri_index = (k*params->npol+pol1)*params->npol+pol2; 
                                reg_index = (l*params->npol+pol1)*params->npol+pol2;
                                complex_tri_buffer[tri_index].real = float_reg_buffer[reg_index];
                                complex_tri_buffer[tri_index].imag = float_reg_buffer[reg_index+matLength];
                                // if(complex_tri_buffer[tri_index].imag == 5 || complex_tri_buffer[tri_index].imag == -5) {
                                //     std::printf("\txgpu_reg_to_tri  | f: %d  tri_index: %d  k: %d  r1: %d  r2:  %d\n", f, tri_index, k, 2*i+rx, 2*j+ry);
                                // }
                                int s1 = 2*i+rx;
                                int s2 = 2*j+ry;
                                // if(s1 == 0 && s2 == 0) {
                                //     std::cout << "TAG | f: " << f << "  pol1: " << pol1 << "  pol2: " << pol2 << "  tri_index: " << tri_index << "\n";
                                // }
                                if(tri_index == 407680) {
                                    std::cout << "ACCESS | " << s1 << "," << s2 << "\n";
                                    std::cout << "\t" << float_reg_buffer[reg_index] << "," << float_reg_buffer[reg_index+matLength] << "\n"; 
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // TODO: replace this by changing pointers instead of redundant copy
    memcpy(float_reg_buffer, complex_tri_buffer, params->output_size*sizeof(Complex));

    free(complex_tri_buffer);

    return;
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
    
    // device function!
    transpose_to_xGPU((cuFloatComplex*) input_d, (cuFloatComplex*) array_d0 , params.nstation*params.npol, params.nfrequency*params.nsample);

    checkXGPUCall(xgpuCudaXengine(&(xgpu_ctx), SYNCOP_SYNC_COMPUTE));

    checkCudaCall(cudaMemcpy(visibilities_h, matrix_d, params.output_size * sizeof(Complex), cudaMemcpyDeviceToHost));

    for(int i = 0; i < params.output_size; i++) {
        if(std::imag(visibilities_h[i]) == 5) { 
            std::cout << "ERROR in xgpuCudaXengine: 5i at " << i << "\n";
        } else if (std::imag(visibilities_h[i]) == -5) {
            std::cout << "ERROR in xgpuCudaXengine: -5i at " << i << "\n";
        }
    }

    // host functions!!
    xgpu_reg_to_tri(&params, visibilities_h);
    for(int i = 0; i < params.output_size; i++) {
        if(std::imag(visibilities_h[i]) == 5) { 
            std::cout << "ERROR in xgpu_reg_to_tri: 5i at " << i << "\n";
        } else if (std::imag(visibilities_h[i]) == -5) {
            std::cout << "ERROR in xgpu_reg_to_tris: -5i at " << i << "\n";
        }
    }

    xgpu_tri_to_mwax(&params, visibilities_h);

    for(int i = 0; i < params.output_size; i++) {
        if(std::imag(visibilities_h[i]) == 5) { 
            std::cout << "ERROR in xgpu_tri_to_mwax: 5i at " << i << "\n";
        } else if (std::imag(visibilities_h[i]) == -5) {
            std::cout << "ERROR in xgpu_tri_to_mwax: -5i at " << i << "\n";
        }
    }


    xgpuFree(&xgpu_ctx);

    return result;
}
