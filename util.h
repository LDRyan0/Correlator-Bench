#ifndef UTIL_H
#define UTIL_H

#include <complex>

// X-engine parameters
typedef struct {
    unsigned int npol;         // number of polarisations (2)
    unsigned int nstation;     // number of antennas/receivers/tiles
    unsigned int nsample;      // number of time samples
    unsigned int nfrequency;   // number of frequency channels
    unsigned int nbaseline;    // number of baselines
    unsigned int input_size;   // number of input elements
    unsigned int output_size;  // number of output elements
    unsigned int flop;         // total floating point operations executed
} Parameters;

// struct for the return type of each call to benchmark
typedef struct { 
    float in_reorder_time;
    float compute_time;
    float out_reorder_time;
} Results;

float byteToMB(long bytes);
void tri_to_mwax(Parameters params, std::complex<float>*, std::complex<float>*);

#endif
