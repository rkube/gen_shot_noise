#include <iostream>
#include <vector>
#include "include/gen_signal.h"

using namespace std;

__global__ void add_to_signal(int* d_burst_tidx, 
                              double* d_burst_amplitude, 
                              int K, 
                              double* d_signal, 
                              int N,
                              int g_offset, 
                              double dt, 
                              double l){
// Compute the total offsets this thread has to compute:
    const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = thread_offset + g_offset;

    int k = 0;
    double result = 0.0;
    double bexp = 0.0;
    for(k = 0; k < K; k++){
        if (offset < d_burst_tidx[k]){
            bexp = dt * (offset - d_burst_tidx[k]) / l;
            //result += d_burst_amplitude[k] * exp(dt * (offset - d_burst_tidx[k]) / l);
        } else if(offset >= d_burst_tidx[k]) {
            bexp = dt * (d_burst_tidx[k] - offset) / (1.0 - l);
            //result += d_burst_amplitude[k] * exp(dt * (d_burst_tidx[k] - offset) / (1.0 - l));
        }
        result += d_burst_amplitude[k] * exp(bexp);
    }
    d_signal[thread_offset] = result;
}

void generate_ts_cuda(int* burst_tidx, double* burst_amplitude, int K,
                      double* signal, int N, double dt, double l){
    int i_it = 0;
    int round_t_offset = 0;
    size_t size_K_double = 0; 
    size_t size_K_int = 0;
    size_t size_N = 0;

    // Pointers to device memory 
    int* d_burst_tidx;
    double* d_burst_amplitude;
    double* d_signal;
    cudaError_t err;

    // Burst parameters that fall in round when wrapping is used
    vector<int> round_burst_tidx;
    vector<double> round_burst_amplitude;

    // Crop signal if it is too long
    const size_t signal_size = N * sizeof(double);
    if (signal_size > cuda_max_gpu_mem){
        cerr << "Array does not fit into GPU memory: " << N * sizeof(double) << "bytes requested\n";
        cerr << "Max allowed: " << cuda_max_gpu_mem << " bytes\n";
        cerr << "Truncating: " << N << " -> " << cuda_max_gpu_mem / sizeof(double) << " elements \n";
        N = cuda_max_gpu_mem / sizeof(double);
    }

    // Elements in time series process per kernel call
    const int elem_per_round = cuda_num_blocks * cuda_blocksize;

    // Compute total number of blocks to be computed
    const int total_blocks = N / cuda_blocksize;
    // Number of rounds considering cuda_max_blocks
    const int num_rounds = total_blocks / cuda_num_blocks;
    // Overlap expressed in tidx
    const int round_halo = 300;

    cout << "N = " << N << ", " << cuda_blocksize << "elements per block, ";
    cout << cuda_num_blocks << "blocks per round, " << num_rounds << " rounds\n";

    int round_num_bursts = 0;
    // Limit GPU Kernel to have cuda_max_blocks blocks per call
    for(int round = 0; round < num_rounds; round++){
        // Flush burst parameters for current round
        round_burst_tidx.clear();
        round_burst_amplitude.clear();
        // Compute offset for current round
        round_t_offset = round * elem_per_round;
        // Find burst arrival times falling in the current bounds. Include those burst
        // arrival times falling in the block overlap from the previous round
        for(i_it = 0; i_it < K; i_it++){
            if ((burst_tidx[i_it] > round_t_offset - elem_per_round * round_halo) &&
                 (burst_tidx[i_it] < round_t_offset + elem_per_round))
            {
                //cout << "Burst: t = " << burst_tidx[i_it] << ", Amplitude: " << burst_amplitude[i_it] << "\n";
                round_burst_tidx.push_back(burst_tidx[i_it]);
                round_burst_amplitude.push_back(burst_amplitude[i_it]);
            }
        }
        //cout << round_burst_tidx.size() << " past bursts in round " << round << "/" << num_rounds << "\n";
        // Bursts to treat in this round
        round_num_bursts = round_burst_tidx.size();
        size_K_int = round_num_bursts * sizeof(int);
        size_K_double = round_num_bursts * sizeof(double);
        size_N = elem_per_round * sizeof(double);
        // Allocate device memory and copy burst parameters to device
        if ( (err = cudaMalloc(&d_burst_tidx, size_K_int)) != cudaSuccess ){
            cerr << "cudaMalloc failed for " << size_K_int << "bytes: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }
        if ( (err = cudaMalloc(&d_burst_amplitude, size_K_double)) != cudaSuccess ){
            cerr << "cudaMalloc failed for " << size_K_double << "bytes: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }
        if ( (err = cudaMalloc(&d_signal, size_N)) != cudaSuccess ){
            cerr << "cudaMalloc failed for " << size_N << "bytes: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }
        cudaMemcpy(d_burst_tidx, round_burst_tidx.data(), size_K_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_burst_amplitude, round_burst_amplitude.data(), size_K_double, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_signal, signal + round_t_offset, size_N, cudaMemcpyHostToDevice);

        add_to_signal<<<cuda_num_blocks, cuda_blocksize>>>(d_burst_tidx, d_burst_amplitude, round_num_bursts, d_signal, elem_per_round, round_t_offset, dt, l);
        cudaMemcpy(signal + round_t_offset, d_signal, size_N, cudaMemcpyDeviceToHost);

        cudaFree(d_signal);
        cudaFree(d_burst_amplitude);
        cudaFree(d_burst_tidx);
    }

}

