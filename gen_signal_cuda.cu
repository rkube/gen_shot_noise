#include <iostream>
#include <vector>
#include <sstream>
#include "include/gen_signal.h"

using namespace std;

__global__ void d_add_to_signal(int* d_burst_tidx, 
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

/// double* d_signal: pointer to the entire signal
/// pulse_params*:  d_pulse_params: Pointer to entire array pulse_params that we use to construct the signal
/// size_t g_offset:
/// size_t k_start
/// size_t k_end
/// size_t num_pulses
/// size_t nelem_signal

__global__ void d_add_to_signal_v2(double* d_signal,  
                                     pulse_params* d_pulse_params,
                                     int g_offset, 
                                     int t_0,
                                     double dt, 
                                     int num_pulses, 
                                     int nelem_signal)
{
    // The thread offset, relative to g_offset
    const unsigned int thread_offset{blockIdx.x * blockDim.x + threadIdx.x};
    // The total offset in d_signal
    const unsigned int offset{thread_offset + g_offset};

    // result stores the contribution of each pulse arriving before offset
    double result{0.0};
    for(size_t k = 0; k < num_pulses; k++)
    {
        if (d_pulse_params[k].tidx > t_0 && d_pulse_params[k].tidx < offset)
            result += d_pulse_params[k].amplitude * exp(dt * (double(d_pulse_params[k].tidx) - double(offset)) / d_pulse_params[k].taud); 
    }

    if (offset < nelem_signal)
        d_signal[offset] = result;
}


size_t pulses_in_k_interval(pulse_params* pulse_params_arr, size_t k_start, size_t k_end, size_t K)
{
    size_t pulse_count{0};
    for(size_t k = 0; k < K; k++)
    {
        cout << " (";
        if(pulse_params_arr[k].tidx > k_start && pulse_params_arr[k].tidx < k_end)
        {
            cout << " ***pulse at tidx = " << pulse_params_arr[k].tidx << " ***";
            pulse_count++;   
        }
        cout << ") ";
    }
    return pulse_count;
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

        d_add_to_signal<<<cuda_num_blocks, cuda_blocksize>>>(d_burst_tidx, d_burst_amplitude, round_num_bursts, d_signal, elem_per_round, round_t_offset, dt, l);
        cudaMemcpy(signal + round_t_offset, d_signal, size_N, cudaMemcpyDeviceToHost);

        cudaFree(d_signal);
        cudaFree(d_burst_amplitude);
        cudaFree(d_burst_tidx);
    }

}

/// generate_ts_cuda_v2
/// vector<pulse>& pulses: The pulses we are adding to the signal
/// double* signal       : The output signal
/// double dt            : Time spacing of signal
/// size_t nelem_signal  : Number of elements of the signal
///
/// The idea is that we have a vector of pulses with some parameters.
/// Define a structure of pulse parameters, like A_k, t_k, tau_d_k, etc. and populate it on the host
/// Copy it to the device and use the same block algorithm as in generate_ts_cuda above.
///
void generate_ts_cuda_v2(vector<pulse>& pulses, double* signal, double dt, size_t nelem_signal)
{
    pulse_params* pulse_params_arr = new pulse_params[pulses.size()];
    // Pointers to device memory
    double* d_signal{nullptr};
    pulse_params* d_pulse_params;
    cudaError_t err;

    // Allocate memory for the output signal and for the pulse parameters on the device
    if ((err = cudaMalloc(&d_signal, nelem_signal * sizeof(double))) != cudaSuccess){
        stringstream err_msg;
        err_msg << "cudaMalloc failed for " << nelem_signal * sizeof(double) << "bytes: " << cudaGetErrorString(err) << "\n";
        throw cuda_error(err, err_msg.str());
    }

    if ((err = cudaMalloc(&d_pulse_params, pulses.size() * sizeof(pulse_params))) != cudaSuccess){
        stringstream err_msg;
        err_msg << "cudaMalloc failed for " << pulses.size() * sizeof(pulse_params) << "bytes: " << cudaGetErrorString(err) << "\n";
        throw cuda_error(err, err_msg.str());
    }
    
    // Copy signal to device
    for(size_t n = 0; n < nelem_signal; n++)
        signal[n] = 0.0;
    cudaMemcpy(d_signal, signal, nelem_signal * sizeof(double), cudaMemcpyHostToDevice);

    // Fill the pulse_params array on the host and copy it on device
    for(size_t p = 0; p < pulses.size(); p++)
    {
        pulse_params_arr[p].amplitude = pulses[p].get_A();
        pulse_params_arr[p].taud = pulses[p].get_tau_d();
        pulse_params_arr[p].tidx = size_t(pulses[p].get_t() / dt);
    }
    cudaMemcpy(d_pulse_params, pulse_params_arr, pulses.size() * sizeof(pulse_params), cudaMemcpyHostToDevice);

    // How many elements of signal get processed each round
    constexpr size_t elem_per_round{cuda_num_blocks * cuda_blocksize};
    // The number of blocks we process every round
    const size_t num_rounds{(nelem_signal + (elem_per_round - 1)) / elem_per_round};
    // Consider bursts in the nelem_halo elements when building the signal
    constexpr size_t nelem_halo {10000};

    cout << nelem_signal << " elements, " << elem_per_round << " elements per round, " << num_rounds << "rounds" << endl;
    // Minimal arrival index of pulses to consider in a round
    size_t t_0{0};
    // Maximal arrival index of pulses to consider in a round
    size_t t_1{0};
    for(size_t round = 0; round < num_rounds; round++)
    {
        // Consider only pulses arriving within [t_start:t_end] in this round
        // Here t_start is given by the start point of elements for this interval - the halo
        t_0 = size_t(max( int(round * elem_per_round - nelem_halo), 0));
        t_1 = size_t((round + 1) * elem_per_round);
        //cout << "Round: " << round << "/" << num_rounds << "\tElements " << round * elem_per_round << ".." << (round + 1) * elem_per_round;
        //cout << "-- t_0=" << t_0 << ", t_1=" << t_1 << ", num_pulses=" << pulses_in_k_interval(pulse_params_arr, t_0, t_1, pulses.size());
        //cout << endl;
        d_add_to_signal_v2<<<cuda_num_blocks, cuda_blocksize>>>(d_signal, 
                                                                d_pulse_params, 
                                                                size_t(round * elem_per_round), 
                                                                t_0, 
                                                                dt, 
                                                                pulses.size(), 
                                                                nelem_signal);
    }        

    cudaMemcpy(signal, d_signal, nelem_signal * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cudaFree(d_pulse_params);
    delete pulse_params_arr; 
}
// End of file gen_signal_cuda.cu
