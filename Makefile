NVCC=              nvcc
NVCCFLAGS=         -std=c++14 -g -G
NVCCFLAGS=         -Xcompiler -fopenmp

CXX=		       g++
CXXFLAGS=          -g -std=c++14 -fopenmp

TCC_LIBDIR=        ./tensor-core-correlator/libtcc
XGPU_LIBDIR=       ./mwax-xGPU/src

TCC_INCDIR=        ./tensor-core-correlator
XGPU_INCDIR=       ./mwax-xGPU/src

LIBS=         -L$(TCC_LIBDIR) -ltcc
LIBS+=          -L$(XGPU_LIBDIR) -lxgpumwax$(NSTATION)t_$(NFREQUENCY)
# LIBS+=          -L$(XGPU_LIBDIR) -lxgpumwax64t_50

INCS=          -I$(TCC_INCDIR) -I$(XGPU_INCDIR)

OBJS=          main.o bench_tcc.o bench_mwax_tcc.o bench_xgpu.o bench_serial.o util.o 
EXEC=          main

$(EXEC): $(OBJS) xgpu
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $(EXEC) $(OBJS) -lgomp

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -c main.cu

bench_tcc.o: bench_tcc.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c bench_tcc.cu

bench_mwax_tcc.o: bench_mwax_tcc.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c bench_mwax_tcc.cu

bench_xgpu.o: bench_xgpu.cu
	$(NVCC) $(NVCCFLAGS) -I$(XGPU_INCDIR) -c bench_xgpu.cu

bench_serial.o: bench_serial.cpp
	$(CXX) $(CXXFLAGS) -c bench_serial.cpp

util.o: util.cpp
	$(CXX) $(CXXFLAGS) -c util.cpp

clean:
	rm *.o $(EXEC)
	rm $(XGPU_LIBDIR)/*.so

xgpu:		
	# make -C $(XGPU_INCDIR) CUDA_ARCH=sm_80 NFREQUENCY=$(NFREQUENCY) NSTATION=$(NSTATION) NTIME=$(NTIME) NTIME_PIPE=$(NTIME) 
	make -C $(XGPU_INCDIR) MATRIX_ORDER_TRIANGULAR=1 CUDA_ARCH=sm_80 NFREQUENCY=$(NFREQUENCY) NSTATION=$(NSTATION) NTIME=$(NTIME) NTIME_PIPE=$(NTIME) 