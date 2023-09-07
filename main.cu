#define NR_BITS 16
#define NR_CHANNELS 6400
#define NR_POLARIZATIONS 2
#define NR_SAMPLES_PER_CHANNEL 64
#define NR_RECEIVERS 144
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 32
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <complex>

// TCC
#include "libtcc/Correlator.h"

/*
typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
*/

/* Correlator::Correlator(unsigned nrBits,
            unsigned nrReceivers,
		    unsigned nrChannels,
		    unsigned nrSamplesPerChannel,
		    unsigned nrPolarizations,
		    unsigned nrReceiversPerBlock,
		    const std::string &customStoreVisibility
		)
*/

// xGPU
#include "xgpu.h"
#include "xgpu_info.h"

/*
 * Data ordering for input vectors is (running from slowest to fastest)
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

// inline void checkXGPUCall(int xgpu_error, XGPUContext* xgpu_context) { 
inline void checkXGPUCall(int xgpu_error) { 
    if(xgpu_error != XGPU_OK) {
        std::cerr << __FILE__ << "(" << __LINE__ <<") xGPU error (code " << xgpu_error << ")\n";
        // xgpuFree(xgpu_context);
        exit(1);
    }
}

inline void checkCudaCall(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}

void showTccInfo() {
    std::cout << "\t================ TCC INFO ================\n";
    std::cout << "\tnpol:                 " << NR_POLARIZATIONS << "\n";
    std::cout << "\tnstation:             " << NR_RECEIVERS << "\n";
    std::cout << "\tnbaseline:            " << NR_BASELINES << "\n";
    std::cout << "\tnfrequency:           " << NR_CHANNELS << "\n";
    std::cout << "\tnsamples:             " << NR_SAMPLES_PER_CHANNEL << "\n";
    std::cout << "\tcompute_type:         ";
    switch(NR_BITS) { 
        case 4:  std::cout << "INT4 multiply, INT32 accumulate\n"; break;
        case 8:  std::cout << "INT8 multiply, INT32 accumulate\n"; break;
        case 16: std::cout << "FP16 multiply,  FP32 accumulate\n"; break;
    }
    std::cout << "\t=============== EXTRA INFO ===============\n";
    std::cout << "\tnreceivers_per_block: " << NR_RECEIVERS_PER_BLOCK << "\n";
    std::cout << "\tntime_per_block:      " << NR_TIMES_PER_BLOCK << "\n";
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
    std::cout << "\tvecLength:          " << xgpu_info.vecLength << "\n";
    std::cout << "\tvecLengthPipe:      " << xgpu_info.vecLengthPipe << "\n";
    std::cout << "\tmatLength:          " << xgpu_info.matLength << "\n";
    std::cout << "\ttriLength:          " << xgpu_info.triLength << "\n";
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

int main () {
    int device = 0;

    std::cout << "Initialising CUDA...\n";
    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));

    std::cout << "Initialising & compiling TCC kernel with NVRTC...\n";
    tcc::Correlator correlator(NR_BITS, NR_RECEIVERS, NR_CHANNELS, NR_SAMPLES_PER_CHANNEL, NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);
    showTccInfo();

    // allocate GPU X-engine memory
    // 
    std::cout << "Initialising XGPU...\n";

    XGPUInfo xgpu_info;
    xgpuInfo(&xgpu_info); // get xGPU info from library
    showxgpuInfo(xgpu_info);

    XGPUContext xgpu_context;
    xgpu_context.array_h = NULL; // NOT USED IN MWAX: host input array
    xgpu_context.matrix_h = NULL; // USED IN MWAX: results from channel averaging, largely reduced size
    checkXGPUCall(xgpuInit(&xgpu_context, device));
    





}

