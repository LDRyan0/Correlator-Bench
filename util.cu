#include <iostream>
#include <cuda.h>

float byteToMB(long bytes) {
    return (float)bytes/(1024.0*1024.0);
}	