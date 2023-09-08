#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cuda_fp16.h>
#include <complex>

#include "bench_tcc.h"

#include "libtcc/Correlator.h"
#include "util.h"

// parameters required by TCC that are customizable
// NR_BITS can be 4, 8, 16
// only have support for 16 (FP16->FP32) so far
#define NR_BITS 16
#define NR_RECEIVERS_PER_BLOCK 32
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

/*
typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
*/

/* Correlator::Correlator(unsigned nrBits
        unsigned nrReceivers,
		    unsigned nrChannels,
		    unsigned nrSamplesPerChannel,
		    unsigned nrPolarizations,
		    unsigned nrReceiversPerBlock,
		    const std::string &customStoreVisibility
		)
*/


inline float byteToMB(long bytes) {
    return (float)bytes/(1024.0*1024.0);
}

// tcc::Correlator holds no internal state of its parameters 
// have to infer from how we originally instantiated tcc::Correlator
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

void runTCC(Parameters params) {
    std::cout << "Initialising & compiling TCC kernel with NVRTC...\n";
    tcc::Correlator correlator(NR_BITS, params.nstation, params.nfrequency, params.nsample, params.npol, NR_RECEIVERS_PER_BLOCK);
    showTccInfo(params);
}

