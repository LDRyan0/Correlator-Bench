#ifndef XGPU_BENCH_H
#define XGPU_BENCH_H

#include <complex>
#include "util.h"

namespace xgpu {
    Results run(Parameters, std::complex<float>* samples, std::complex<float>* visibilities);
}
#endif

