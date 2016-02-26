#ifndef gen_signal_h
#define gen_signal_h

#include <string>
#include <exception>
#include <vector>
#include "datatypes.h"
#include <cuda_runtime.h>

// Threads per block
const int cuda_blocksize = 64;

// Maximum number of blocks per kernel call
const int cuda_num_blocks = 128;

// Maximal length of time series to be computed at once
// Put maximal 256MB on the GPU, thats 2^25 doubles
//const int cuda_max_gpu_mem = 268435456;
const int cuda_max_gpu_mem = 805306368;
//const int cuda_max_gpu_mem = 1073414144;

void generate_ts_omp(int* burst_tidx, double* burst_amplitude, int K,
                     double* signal, int N, int set_nth, double g, double l);

void generate_ts_cuda(int* burst_tidx, double* burst_amplitude, int K,
                      double* signal, int N, double g, double l);

void generate_ts_cuda_v2(std::vector<pulse>& pulses, double* signal, double dt, size_t nelem_signal);

class cuda_error : public std::exception
{
    public:
        cuda_error(cudaError_t err_t_, std::string err_msg_) : err_t(err_t_), err_msg(err_msg_) {};
        cudaError_t get_err_t() const {return (err_t);};
        virtual const char* what() const throw() {return (err_msg.data());};
    private:
        const cudaError_t err_t;
        const std::string err_msg;
};
#endif //gen_signal_h
