# Correlator-Bench
A benchmarking framework for FX telescope arrays.

Early development.

# Build
- `git clone --recursive` to obtain xGPU and TCC
- follow instructions to build xGPU and TCC
- update `LD_LIBRARY_PATH` to point to shared libraries
- `make`

| Software    | Minimum version |
| ----------- | ----------- |
| CUDA        | 10.0 or later |
| CMake       | 3.17 or later |
| gcc         | 9.3 or later  |
| OS          | Linux distro (amd64) |
