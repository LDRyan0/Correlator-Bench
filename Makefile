NVCC=          nvcc
NVCCFLAGS=     -std=c++14 -g -G

TCC_LIBDIR=    ./tensor-core-correlator/libtcc/
XGPU_LIBDIR=   ./mwax-xGPU/src/

TCC_INCDIR=    ./tensor-core-correlator/
XGPU_INCDIR=   ./mwax-xGPU/src/

LIBS=          -L$(XGPU_LIBDIR) -lxgpumwax64t_50
LIBS+=         -L$(TCC_LIBDIR) -ltcc

INCS=          -I$(TCC_INCDIR) -I$(XGPU_INCDIR)

OBJS=          main.o bench_tcc.o bench_xgpu.o util.o
EXEC=          main

$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -c main.cu

bench_tcc.o: bench_tcc.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c bench_tcc.cu

bench_xgpu.o: bench_xgpu.cu
	$(NVCC) $(NVCCFLAGS) -I$(XGPU_INCDIR) -c bench_xgpu.cu

util.o: util.cu
	$(NVCC) $(NVCCFLAGS) -c util.cu

clean:
	rm *.o $(EXEC)

