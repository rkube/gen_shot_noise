CC	= /opt/local/bin/g++-mp-4.8
CFLAGSOMP	= -fopenmp -std=c++11 -O2 
IFLAGS	= -I/Users/ralph/source/gen_shot_noise/include  -I/opt/local/include
LFLAGS	= -L/opt/local/lib -L/Developer/NVIDIA/CUDA-5.5/lib -I/Users/ralph/source/gen_shot_noise/include -lcudart -lboost_program_options-mt  -lhdf5_hl_cpp -lhdf5_cpp -lhdf5 -lhdf5_hl


CUDACC	= /Developer/NVIDIA/CUDA-5.5/bin/nvcc
#CUDACFLAGS	= --compiler-bindir=/Developer/NVIDIA/CUDA-5.5/bin -O2 --compile --ptxas-options=-v --gpu-architecture sm_30 --machine 64 
CUDACFLAGS	= -O2 --compile --gpu-architecture sm_30 --machine 64 

gen_shot_noise: gen_signal_cuda.cu gen_shot_noise.cpp gen_signal_omp.cpp
	$(CC) $(CFLAGSOMP) -c -o objs/gen_signal_omp.o gen_signal_omp.cpp  $(IFLAGS) 
	$(CUDACC) $(CUDACFLAGS) -o objs/gen_signal_cuda.o gen_signal_cuda.cu 
	$(CC) $(CFLAGSOMP) -o gen_shot_noise gen_shot_noise.cpp objs/gen_signal_omp.o objs/gen_signal_cuda.o $(IFLAGS) $(LFLAGS)

clean:
	rm gen_shot_noise objs/gen_signal_omp.o objs/gen_signal_cuda.o

