NVCC=          nvcc
NVCCFLAGS=     -std=c++14
NVCCFLAGS=     -Xcompiler -fopenmp

CXX=		   g++
CXXFLAGS=      -g -std=c++14 -fopenmp
# CXXFLAGS=      -std=c++14 -O3 -fopenmp

TCC_LIBDIR=    ./tensor-core-correlator/libtcc
XGPU_LIBDIR=   ./mwax-xGPU/src

TCC_INCDIR=    ./tensor-core-correlator
XGPU_INCDIR=   ./mwax-xGPU/src

LIBS=          -L$(TCC_LIBDIR) -ltcc
LIBS+=         -L$(XGPU_LIBDIR) -lxgpumwax$(NSTATION)t_$(NFREQUENCY)

INCS=          -I$(TCC_INCDIR) -I$(XGPU_INCDIR)

OBJS=          src/main.o src/tcc1_bench.o src/tcc2_bench.o src/tcc3_bench.o src/xgpu_bench.o src/serial_bench.o src/util.o 
EXEC=          xbench

debug: NVCCFLAGS += -g -G -DDEBUG
debug: $(EXEC)

release: NVCCFLAGS += -O3
release: $(EXEC)

$(EXEC): $(OBJS) xgpu
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $(EXEC) $(OBJS) -lgomp

src/main.o: src/main.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

src/tcc1_bench.o: src/tcc1_bench.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c -o $@ $<

src/tcc2_bench.o: src/tcc2_bench.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c -o $@ $<

src/tcc3_bench.o: src/tcc3_bench.cu
	$(NVCC) $(NVCCFLAGS) -I$(TCC_INCDIR) -c -o $@ $<

src/xgpu_bench.o: src/xgpu_bench.cu
	$(NVCC) $(NVCCFLAGS) -I$(XGPU_INCDIR) -c -o $@ $<

src/serial_bench.o: src/serial_bench.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/util.o: src/util.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm src/*.o $(EXEC)
	rm $(XGPU_LIBDIR)/*.so

xgpu:		
	make -C $(XGPU_INCDIR) MATRIX_ORDER_TRIANGULAR=1 CUDA_ARCH=sm_80 NFREQUENCY=$(NFREQUENCY) NSTATION=$(NSTATION) NTIME=$(NTIME) NTIME_PIPE=$(NTIME)