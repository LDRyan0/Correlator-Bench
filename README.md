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
## Critical
- ~complex normally distributed input values~
- ~xGPU~
  - ~input reorder to xGPU~
  - ~execution of xGPU~
  - ~output reorder xGPU to MWAX~
- ~TCC~
  - ~float to half conversion~
  - ~input reorder to TCC~
  - ~execution of TCC~
  - ~output reorder xGPU to MWAX~
- ~Serial~
- ~benchmarking analysis~
- ~output validation~
- ~result precision comparison (TCC requires downconversion to FP16)~
- ~change tcc reorder kernel to support > 1024 channels~

## Non-critical
- use `-Wl` and `-rpath` in Makefile so user doesn't have to manually change `LD_LIBRARY_PATH`
- generalise interface
  - change main.cu to main.c/main.cpp
  - pass arrays as void* and let user typecast to custom complex format
  - store size of input/output type in parameters 
  - compile flags for AMD/CUDA
  - save output to disk to compare accuracy and performance across AMD/CUDA
