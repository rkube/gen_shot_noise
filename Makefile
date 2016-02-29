CC	= /usr/bin/g++
CFLAGSOMP	= -Wall -fopenmp -std=c++11 -O0 
IFLAGS	= -I/home/rku000/source/gen_shot_noise/include  -I/usr/local/cuda/include
LFLAGS	= -L/usr/local/cuda/lib64 -I/home/rku000/source/gen_shot_noise/include -lcudart  -lboost_program_options -lboost_system -lboost_filesystem

CUDACC	= /usr/local/cuda/bin/nvcc
#CUDACFLAGS	= --compiler-bindir=/Developer/NVIDIA/CUDA-5.5/bin -O2 --compile --ptxas-options=-v --gpu-architecture sm_30 --machine 64 
CUDACFLAGS	= -O0 --std=c++11 --compile --gpu-architecture sm_50 --machine 64 

gen_signal_cuda: gen_signal_cuda.cu
	$(CUDACC) $(CUDACFLAGS) -o objs/gen_signal_cuda.o gen_signal_cuda.cu 

gen_signal_omp: gen_signal_omp.cpp
	$(CC) $(CFLAGSOMP) -c -o objs/gen_signal_omp.o gen_signal_omp.cpp  $(IFLAGS) 

datatypes: datatypes.cpp
	$(CUDACC) $(CUDACFLAGS) -c -o objs/datatypes.o datatypes.cpp $(IFLAGS)

gen_shot_noise: gen_signal_cuda gen_shot_noise.cpp gen_signal_omp.cpp
	$(CC) $(CFLAGSOMP) -o gen_shot_noise gen_shot_noise.cpp objs/gen_signal_omp.o objs/gen_signal_cuda.o $(IFLAGS) $(LFLAGS)

config: config.cpp include/gen_shot_noise.h
	$(CC) $(CFLAGSOMP) -c -o objs/config.o config.cpp

gen_shot_noise_time: gen_shot_noise_time.cpp config gen_signal_cuda.cu include/gen_shot_noise.h
	$(CC) $(CFLAGSOMP) -o gen_shot_noise_time gen_shot_noise_time.cpp objs/gen_signal_cuda.o objs/config.o $(IFLAGS) $(LFLAGS)

clean:
	rm gen_shot_noise gen_shot_noise_time objs/gen_signal_omp.o objs/gen_signal_cuda.o objs/config.o

