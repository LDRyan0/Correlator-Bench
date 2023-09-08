#ifndef UTIL_H
#define UTIL_H

// common interface to convery X-engine sizing parameters
typedef struct {
    unsigned long npol;
    unsigned long nstation;
    unsigned long nbaseline;
    unsigned long nfrequency;
    unsigned long nsample;
    unsigned long input_size;
    unsigned long output_size;
} Parameters;

// struct for the return type of each call to benchmark
typedef struct { 
    float in_reorder_time;
    float compute_time;
    float out_reorder_time;
} BenchResult;

float byteToMB(long bytes);

#endif
