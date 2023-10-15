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
    unsigned long long input_size;   // number of input elements
    unsigned long long output_size;  // number of output elements
    unsigned long long flop;         // total floating point operations executed
    bool verify;
    bool write_csv;
    bool snapshot;
} Parameters;

// struct for the return type of each call to benchmark
typedef struct { 
    float in_reorder_time;
    float compute_time;
    float tri_reorder_time;
    float channel_avg_time;
    float mwax_time;
} Results;

float byteToMB(long bytes);
void tri_to_mwax(Parameters params, std::complex<float>*, std::complex<float>*);

#endif
