#include <complex>
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "util.h"

// [station][polarisation][time][frequency] -> [baseline][frequency][polarisation*4]
Results runSerial(Parameters params, const std::complex<float>* in, std::complex<float>* out) { 
    Results result = {0};
    typedef std::chrono::high_resolution_clock Clock;

    int ns = params.nstation;
    int np = params.npol;
    int nf = params.nfrequency;
    int nt = params.nsample;

    int samp1_idx, samp2_idx, vis_idx;
    int b_idx = 0; // "triangular" baseline index

    auto t0 = Clock::now();
    for(int s1 = 0; s1 < ns; s1++) {
        for(int s2 = s1; s2 < ns; s2++) {
            for(int p1 = 0; p1 < np; p1++) { 
                for(int p2 = 0; p2 < np; p2++) { 
                    for(int t = 0; t < nt; t++) {
                        for(int f = 0; f < nf; f++) { 
                            samp1_idx = s1*np*nt*nf + p1*nt*nf + t*nf + f;
                            samp2_idx = s2*np*nt*nf + p2*nt*nf + t*nf + f;
                            vis_idx = b_idx*nf*np*np + f*np*np + p1*np + p2;
                            out[vis_idx] += in[samp1_idx] * std::conj(in[samp2_idx]);
                        }
                    }
                }
            }
            b_idx++;
        }
    }

    auto t1 = Clock::now();
    std::chrono::duration<float> elapsed = t1 - t0;
    result.compute_time = elapsed.count();
    return result;
}