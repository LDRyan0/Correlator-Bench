NVCC=       nvcc
CC=         gcc
CXX=        g++
NVCCFLAGS=  -std=c++14

TCC_LIBDIR=    ./tensor-core-correlator/libtcc/
TCC_INCDIR=    ./tensor-core-correlator/

XGPU_LIBDIR=   ./mwax-xGPU/src/
XGPU_INCDIR=   ./mwax-xGPU/src/

LIBS=          -L$(TCC_LIBDIR) -L$(XGPU_LIBDIR)
INCS=          -I$(TCC_INCDIR) -I$(XGPU_INCDIR)

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(LIBS) $(INCS) -ltcc -lxgpumwax144t_6400 main.cu -o main

clean: 
	rm -rf main