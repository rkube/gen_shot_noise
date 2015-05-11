#include <omp.h>
#include <iostream>
#include <cmath>

void generate_ts_omp(int* burst_tidx, 
                     double* burst_amplitude,
                     int K,
                     double* signal,
                     int N,
                     int set_nth, double dt, double l){
// Generate synthetic time series using OpenMP
// Psi(t) = sum_{k=1}^{K} psi_k (t - t_k) where
//          / A_k exp(t / tau) if t < 0
// psi(t) = 
//          \ A_k exp(-t / tau) if t > 0


    int tid;
    double* S_local;
    int i_it, j_it;

    omp_set_num_threads(set_nth);

    std::cout << "Generating " << K << "bursts, dt= " << dt << ", l = " << l << "\n";
#pragma omp parallel shared(burst_tidx, burst_amplitude) private(S_local, i_it, j_it)
{
    // Allocate memory for signal in each thread
    S_local = (double*) calloc(N, sizeof(double));
#pragma omp for 
    // Generate individual bursts on each thread
    for (i_it = 0; i_it < K; i_it++){
        tid = omp_get_thread_num();
        //printf("Thread: %d/%d\t Burst# %d\ttburst=%8f\tAmplitude=%f\n" , tid, nthreads, i_it, burst_time[i_it], burst_amplitude[i_it]);

        // Generate burst rise, t_rise = l
        for (j_it = 0; j_it < burst_tidx[i_it]; j_it++ ){
            S_local[j_it] += burst_amplitude[i_it] * 
                             exp( -1.0 * dt * (burst_tidx[i_it] - j_it) / l);
        }
        // Generate burst decay, t_fall = 1-l
        for (j_it = burst_tidx[i_it]; j_it < N; j_it++){
            S_local[j_it] += burst_amplitude[i_it] * 
                             exp( -1.0 * dt * (j_it - burst_tidx[i_it]) / (1-l) );
        }
    }
    // Add individual bursts to signal
#pragma omp critical
    for(j_it = 0; j_it < N; j_it++){
        signal[j_it] += S_local[j_it];
    }
    // Delete local signal memory 
    free(S_local);
}
}
