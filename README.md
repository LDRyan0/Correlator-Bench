# Correlator-Bench
A benchmarking and validation framework for the X-engine of FX telescope arrays.

Early development. Only targeting [TCC](https://git.astron.nl/RD/tensor-core-correlator) and [mwax-xGPU](https://github.com/MWATelescope/mwax-xGPU/tree/master/src) so far.

# Build
- `git clone --recursive` to obtain xGPU and TCC
- follow instructions to build xGPU and TCC
- ensure `LD_LIBRARY_PATH` points to shared libraries
    - `tensor-core-correlator/libtcc/libtcc.so`
    - `tensor-core-correlator/external/cuda-wrappers/libcu.so`
    - `mwax-xGPU/src/libxgpu.so`
- `make`

| Software    | Minimum version |
| ----------- | ----------- |
| CUDA        | 10.0 or later |
| CMake       | 3.17 or later |
| gcc         | 9.3 or later  |
| OS          | Linux distro (amd64) |

# Roadmap
- ~complex normally distributed input values~
- ~xGPU~
  - ~input reorder to xGPU~
  - ~execution of xGPU~
  - ~output reorder xGPU to MWAX~
- TCC
  - input reorder to TCC
  - ~execution of TCC~
  - output reorder xGPU to MWAX
- benchmarking analysis
- output validation
- result precision comparison (TCC requires downconversion to FP16)
- generalise interface
  - change main.cu to main.c/main.cpp
  - pass arrays as void* and let user typecast to custom complex format
  - store size of input in parameters for full generalisability
