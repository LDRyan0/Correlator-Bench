#ifndef UTIL_H
#define UTIL_H

// common interface to convery X-engine sizing parameters
typedef struct {
    unsigned int npol;
    unsigned int nstation;
    unsigned int nsample;
    unsigned int nfrequency;
    unsigned int nbaseline;
    unsigned int input_size;
    unsigned int output_size;
} Parameters;

// struct for the return type of each call to benchmark
typedef struct { 
    float in_reorder_time;
    float compute_time;
    float out_reorder_time;
} Results;

float byteToMB(long bytes);

#endif
