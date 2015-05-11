#include <iostream>
#include <cstdio>
#include <random>
#include <vector>
#include <tuple>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <boost/program_options.hpp>
#include <iterator>
#include <omp.h>
#include <H5cpp.h>
#include "gen_signal.h"

using namespace std;


class ts_params{
public:
    ts_params(double g0, double l0, int K0, double A0, double dt0) : g(g0), l(l0), K(K0), A_mean(A0), dt(dt) {};
    inline double get_g() const {return g;};
    inline double get_l() const {return l;};
    inline int get_K() const {return K;};
    inline double get_A() const {return A_mean;};
    inline double get_dt() const {return dt;};
private:
    double g;
    double l;
    int K;
    double A_mean;
    double dt;
};


// Writes output to HDF5 file
void write_output(ts_params tsp, double* signal, int N)
{
    char fname[128];
    //char fname_ascii[128];
    sprintf(fname, "signals/shotnoise_K%010d_g%04.1f_l%04.2f_omp.h5", tsp.get_K(), tsp.get_g(), tsp.get_l());
    //sprintf(fname_ascii, "signals/shotnoise_K%010d_g%04.1f_l%04.2f_omp.dat", tsp.get_K(), tsp.get_g(), tsp.get_l());
    cout << "Writing output to " << fname << "\n";
    H5::H5File file (fname, H5F_ACC_TRUNC);

    hsize_t N_t(N);

    H5::DataSpace dspace(1, &N_t);
    H5::DataSet dset = file.createDataSet("Time Series", H5::PredType::NATIVE_DOUBLE, dspace);
    dset.write(signal, H5::PredType::NATIVE_DOUBLE);

    // Add parameters used to generate time series as attributes
    double g0 = tsp.get_g();
    double l0 = tsp.get_l();
    double A0 = tsp.get_A();
    int K0 = tsp.get_K();

    H5::FloatType double_type(H5::PredType::NATIVE_DOUBLE);
    H5::IntType int_type(H5::PredType::NATIVE_INT);

    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att_g = dset.createAttribute("gamma", double_type, att_space);
    H5::Attribute att_l = dset.createAttribute("l", double_type, att_space);
    H5::Attribute att_A = dset.createAttribute("A_mean", double_type, att_space);
    H5::Attribute att_K = dset.createAttribute("K", double_type, att_space);
    att_g.write(double_type, &g0);
    att_l.write(double_type, &l0);
    att_A.write(double_type, &A0);
    att_K.write(int_type, &K0);

    /*
    ofstream outfile;
    outfile.open(fname_ascii);
    for(int i_it = 0; i_it < N; i_it++)
    {
        outfile << std::fixed << std::setprecision(10) << signal[i_it] << "\n";
    } 
    outfile.close();
    */
}



// Define tuple for burst events
// burst number, burst time index, burst amplitude
typedef tuple<int, double, double> burst_tuple;

// Sort function for tuples, comparing the first number
// Note to self: We compare double values with this function.
// Do not use epsilons for comparison because we only need alomst
// greater than
bool burst_cmp(const burst_tuple &lhs, const burst_tuple &rhs){
    return( get<1>(lhs) < get<1>(rhs) );
}

int main(int argc, char* argv[]){
    // Shot noise parameters:
    // g: Intermittency parameter: tau_d/tau_w
    // l: l = tau_rise, tau_fall = (1-l)
    // N: Number of bursts
    // A_mean: Amplitude mean_value
    double g = 1.0;
    double l = 0.05;
    int K = 10;
    double A_mean = 1.0;                  // Amplitude mean value
    const double dt = 0.01;

    bool use_cuda = false;
    bool use_omp = false;
    // Filename of the log file
    char log_filename[128];
    ofstream logfile;

    int i_it, j_it;
    // OpenMP variables
    //int tid;
    int set_nth = 1;
    //tid = omp_get_thread_num();

    cout << "Creating a synthetic shot noise signal\n";
    cout << "Parameters:\n";
    /* Parse command line options */
    try{
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("K", boost::program_options::value<int>(), "Number of bursts, default = 10000")
            ("g", boost::program_options::value<double>(), "Intermittency parameter tau_d/tau_w, default = 1.0")
            ("l", boost::program_options::value<double>(), "tau_rise, tau_decay = 1-l, default = 0.05")
            ("A", boost::program_options::value<double>(), "Expected burst amplitude, default = 1.0")
            ("n", boost::program_options::value<int>(), "Number of threads")
            ("cuda", boost::program_options::value<bool>(), "Use CUDA to generate time series")
            ("omp", boost::program_options::value<bool>(), "Use OpenMP to generate time series");
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")){
            cout << desc << "\n";
            return(1);
        }
        if ( vm.count("K") ){
            K = vm["K"].as<int>();
            cout << "From command line: K = " << K << "\n";
        } else {
            K = 10000;
            cout << "Using default: K = " << K << "\n";
        }
        if ( vm.count("g") ){
            g = vm["g"].as<double>();
            cout << "From command line: g = " << g << "\n";
        } else {
            g = 1.0;
            cout << "Using default: g = " << g << "\n";
        }
        if ( vm.count("l") ){
            l = vm["l"].as<double>();
            cout << "From command line: l = " << l << "\n";
        } else {
            l = 0.05;
            cout << "Using default: l = " << l << "\n";
        } 
        if ( vm.count("A") ) {
            A_mean = vm["A"].as<double>();
            cout << "From command line: A = " << A_mean << "\n";
        } else {
            A_mean = 1.0;
            cout << "Using default: A = " << A_mean << "\n";
        }
        if ( vm.count("n") ){
            set_nth = vm["n"].as<int>();
            cout << "Setting number of threads" << set_nth << "\n";
        } 
        if (vm.count("omp")){
            use_omp = true;
            cout << "Using OMP\n";
        } 
        if (vm.count("cuda")){
            use_cuda = true;
            cout << "Using CUDA\n";
        }

    } catch (exception& e){
        cerr << "error: " << e.what() << "\n";
        return 1;
    } catch(...) {
        cerr << "Exception of unknown type\n";
        return 1;
    }

    // If we use CUDA to generate the series, pad it to a multiple of the blocksize
    /*if (use_cuda){
        K = K + (K % cuda_blocksize);
        cout << "Padding number of bursts to " << K < "\n";
    }*/

    if (use_omp & use_cuda){
        cout << "Command line specified usage of both OMP and CUDA, defaulting to OMP\n";
        use_cuda = false;
    }


    //
    // Set up filenames for file output
    // Write one file with the time series and one file with the amplitude 
    // distribution / burst times
    // out_filename: Timeseries
    // log_filename: amplitude distribution and burst time
    //

    sprintf(log_filename, "signals/shotnoise_K%010d_g%04.1f_l%04.2f_omp.log", K, g, l);
    cout << "Logfile: " << log_filename << "\n\n";

    // Derived parameters for library calls:
    // exp_dist lambda: Rate of occurance, see 
    // http://www.cplusplus.com/reference/random/exponential_distribution/
    const double exp_dist_lambda = 1 / A_mean;  // E[x] = 1/lambda 

    const double t_end = float(K)/g;
    const size_t N = K/(g*dt);
   
    // Vector containing bursts as they are drawn from the RNG
    vector<burst_tuple> bursts_vector(K);

    // Vectors containing the burst values, sorted by arrival time
    vector<double> burst_time(K);
    vector<int> burst_tidx(K);
    vector<double> burst_amplitude(K); 

    //double* time_range = (double*) calloc(N, sizeof(double));
    //double* signal = (double*) calloc(N, sizeof(double));
    double* time_range = new double[N];
    double* signal = new double[N];

    for( i_it = 0; i_it < N; i_it++){
        time_range[i_it] = float(i_it) * dt;
        signal[i_it] = -1.0;
    }

    // Random number generator
    random_device rd;
    // Use a Mersenne twister to generate random numbers
    mt19937 rnd_gen( rd() );
    // Create exponentially distributed random numbers
    exponential_distribution<double> dist_exp(exp_dist_lambda);
    // Create uniformly distributed random numbers
    uniform_real_distribution<double> dist_uni(0.0, 1.0);

    // Draw K uniformly and K exponentially distributed random numbers and
    // create burst tuples
    i_it = 0;
    for(auto v_it = bursts_vector.begin(); v_it != bursts_vector.end(); v_it++){
        *v_it = make_tuple(i_it, dist_uni(rnd_gen), dist_exp(rnd_gen)); 
        i_it++;
    }

    // Sort vector and fill burst_time, burst_tidx, burst_amplitude
    sort(bursts_vector.begin(), bursts_vector.end(), burst_cmp);
    i_it = 0;
    for(auto v_it = bursts_vector.begin(); v_it != bursts_vector.end(); v_it++){
        burst_time[i_it] = get<1>(*v_it) * t_end;
        burst_tidx[i_it] = floor(get<1>(*v_it) * t_end / dt);
        burst_amplitude[i_it] = get<2>(*v_it);
        i_it++;
    }

    // Write log file: Amplitude and burst arrival time
    cout << "Writing logfile\n";
    logfile.open(log_filename, ios_base::out | ios_base::trunc);
    for ( i_it = 0; i_it < K; i_it++){
        logfile << std::fixed << std::setprecision(6);
        logfile << burst_amplitude[i_it] << "\t" << burst_time[i_it] << "\n";
    }
    logfile.close();

    cout << "Creating shot noise time series with " << N << " elements\n";

    if (use_omp){
        cerr << "Generating timeseries with OMP\n";
        generate_ts_omp(burst_tidx.data(), burst_amplitude.data(), K, signal, N, set_nth, dt, l);
    } else if(use_cuda){
        cerr << "Generating timeseries with CUDA\n";
        generate_ts_cuda(burst_tidx.data(), burst_amplitude.data(), K, signal, N, dt, l);
    }
    cout << "...done\n Writing output\n";

    // Write file output.
    ts_params my_ts(g, l, K, A_mean, dt);
    write_output(my_ts, signal, N);

    delete [] signal;
    delete [] time_range;
    // Tada!
    return(0);
}



