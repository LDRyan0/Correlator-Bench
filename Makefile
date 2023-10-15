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

OBJS=          src/main.o src/bench_tcc.o src/bench_mwax_tcc.o src/bench_xgpu.o src/bench_serial.o src/util.o 
EXEC=          main

$(EXEC): $(OBJS) xgpu
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $(EXEC) $(OBJS) -lgomp

src/main.o: src/main.cu
	$(NVCC) $(NVCCFLAGS) -c src/main.cu

src/bench_tcc.o: src/bench_tcc.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c src/bench_tcc.cu

src/bench_mwax_tcc.o: src/bench_mwax_tcc.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c src/bench_mwax_tcc.cu

src/bench_xgpu.o: src/bench_xgpu.cu
	$(NVCC) $(NVCCFLAGS) -I$(XGPU_INCDIR) -c src/bench_xgpu.cu

src/bench_serial.o: src/bench_serial.cpp
	$(CXX) $(CXXFLAGS) -c src/bench_serial.cpp

src/util.o: src/util.cpp
	$(CXX) $(CXXFLAGS) -c src/util.cpp

clean:
	rm src/*.o $(EXEC)
	rm $(XGPU_LIBDIR)/*.so

xgpu:		
	# make -C $(XGPU_INCDIR) CUDA_ARCH=sm_80 NFREQUENCY=$(NFREQUENCY) NSTATION=$(NSTATION) NTIME=$(NTIME) NTIME_PIPE=$(NTIME) 
	make -C $(XGPU_INCDIR) MATRIX_ORDER_TRIANGULAR=1 CUDA_ARCH=sm_80 NFREQUENCY=$(NFREQUENCY) NSTATION=$(NSTATION) NTIME=$(NTIME) NTIME_PIPE=$(NTIME) 