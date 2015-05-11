// Threads per block
const int cuda_blocksize = 192;

// Maximum number of blocks per kernel call
const int cuda_num_blocks = 2;

// Maximal length of time series to be computed at once
// Put maximal 256MB on the GPU, thats 2^25 doubles
//const int cuda_max_gpu_mem = 268435456;
const int cuda_max_gpu_mem = 805306368;
//const int cuda_max_gpu_mem = 1073414144;

void generate_ts_omp(int* burst_tidx, double* burst_amplitude, int K,
                     double* signal, int N, int set_nth, double g, double l);

void generate_ts_cuda(int* burst_tidx, double* burst_amplitude, int K,
                      double* signal, int N, double g, double l);

