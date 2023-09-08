NVCC=          nvcc
CC=            gcc
CXX=           g++
NVCCFLAGS=     -std=c++14 

TCC_LIBDIR=    ./tensor-core-correlator/libtcc/
TCC_INCDIR=    ./tensor-core-correlator/

XGPU_LIBDIR=   ./mwax-xGPU/src/
XGPU_INCDIR=   ./mwax-xGPU/src/


LIBS=          -L$(XGPU_LIBDIR) -L$(TCC_LIBDIR)
INCS=          -I$(XGPU_INCDIR) -I$(TCC_INCDIR)
LDLIBS=        -ltcc -lxgpumwax64t_50

OBJS=          bench_tcc.o bench_xgpu.o
EXEC=          main

$(EXEC): main.cu bench_tcc.o bench_xgpu.o util.o
	$(NVCC) $(NVCCFLAGS) $(INCS) $(LIBS) $(LDLIBS) main.cu -o main

bench_tcc.o: bench_tcc.cu util.o
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -ltcc -c bench_tcc.cu

bench_xgpu.o: bench_xgpu.cu util.o
	$(NVCC) $(NVCCFLAGS) -I$(XGPU_INCDIR) -lxgpumwax64t_50 -c bench_xgpu.cu

util.o: util.cu
	$(NVCC) $(NVCCFLAGS) -c util.cu

clean: 
	rm $(EXEC) $(OBJS) 
