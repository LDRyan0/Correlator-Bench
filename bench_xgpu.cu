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

#include <iostream>
#include <cassert>
#include <cuda.h>

#include "bench_xgpu.h"
#include "util.h"


/* Data ordering for xGPU input vectors is (running from slowest to fastest)
 * [time][channel][station][polarization][complexity]
 * Output matrix has ordering
 * [channel][station][station][polarization][polarization][complexity]
 */

/* Define MATRIX_ORDER based on which MATRIX_ORDER_XXX is defined.
 * There are three matrix packing options:
 *
 * TRIANGULAR_ORDER
 * REAL_IMAG_TRIANGULAR_ORDER
 * REGISTER_TILE_TRIANGULAR_ORDER (default)
 *
 * To specify the matrix ordering scheme at library compile time, use one of
 * these options to the compiler:
 *
 * -DMATRIX_ORDER_TRIANGULAR
 * -DMATRIX_ORDER_REAL_IMAG
 * -DMATRIX_ORDER_REGISTER_TILE
 */

/* Return values from xgpuCudaXengine()
#define XGPU_OK                          (0)
#define XGPU_OUT_OF_MEMORY               (1)
#define XGPU_CUDA_ERROR                  (2)
#define XGPU_INSUFFICIENT_TEXTURE_MEMORY (3)
#define XGPU_NOT_INITIALIZED             (4)
#define XGPU_HOST_BUFFER_NOT_SET         (5)
*/


inline void checkXGPUCall(int xgpu_error) { 
// void checkXGPUCall(int xgpu_error) { 
    if(xgpu_error != XGPU_OK) {
        std::cerr << __FILE__ << "(" << __LINE__ << ") xGPU error (code " << xgpu_error << ")\n";
        // xgpuFree(xgpu_context);
        exit(1);
    }
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

void runXGPU(Parameters params) {
    int device = 0;
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

    XGPUContext xgpu_context;
    xgpu_context.array_h = NULL; // NOT USED IN MWAX: host input array
    xgpu_context.matrix_h = NULL; // USED IN MWAX: results from channel averaging, largely reduced size
    checkXGPUCall(xgpuInit(&xgpu_context, device));

    /* xGPU structs (not using DP4A):
     * typedef float ReImInput;
     * 
     * typedef struct ComplexInputStruct {
     *   ReImInput real;
     *   ReImInput imag;
     * } ComplexInput;
     * 
     * typedef struct ComplexStruct {
     *   float real;
     *   float imag;
     * } Complex;
     */
    ComplexInput *array_h = xgpu_context.array_h;
    Complex *cuda_matrix = xgpu_context.matrix_h;
}
