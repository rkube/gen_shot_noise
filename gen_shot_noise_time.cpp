/*
 * Generate a shot noise signal where the bursts have individual amplitude,
 * radial velocity, decay time, as well as size distributions
 *
 *
 * Characteristic scales used: Length: 1cm, time: 1 microsecond
 */

#include <iostream>
#include <fstream>
//#include <boost/program_options>
#include <random>
#include <vector>
#include <array>
#include <algorithm>
#include "include/gen_signal.h"
#include "include/gen_shot_noise.h"

using namespace std;

/// Class to handle output
/// Constructor creates the output file
/// write_output_xi writes the output for a given xi
//

//output_t :: output_t(string filename_, 
//                     unsigned int K_, 
//                     double T_end_, 
//                     double dt_, 
//                     double L_par_, 
//                     double_C_s,
//                     double L_sol) : K(K_), T_end(T_end_), dt(dt_), L_par(L_par_), C_s(C_s_)
//{
//    H5::F5File file(filename.str(), H5F_ACC_TRUNC);
//
//}

// Template specialization of exponentially distributed variable because
// it takes only one parameter instead of two
template<>
rand_functor<std::exponential_distribution<double> > :: rand_functor(double shape_, double scale_) : shape(shape_), scale(scale_)
{
    exponential_distribution<double> random_distribution(shape);
}

template<>
rand_functor<std::exponential_distribution<double> > :: rand_functor(vector<double> params) : rand_functor(params[0], params[1]){}


int main(int argc, char* argv[])
{
    config sn_config("shotnoise.cfg");
    constexpr size_t num_xi{10};
    constexpr array<double, 10> xi_range{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    double* signal_xi = new double[sn_config.get_nelem()];


    switch (sn_config.get_dist_arrival_type())
    {
        //case dist_t::expon_t: rand_functor<exponential_distribution<double> > A_k0(sn_config.get_dist_amp_params()); break;
        //case dist_t::normal_t: rand_functor<normal_distribution<double> > A_k0(sn_config.get_dist_amp_params()); break;
        //case dist_t::uniform_t: rand_functor<uniform_distribution<double> > A_k0(sn_config.get_dist_amp_params()); break;
        default:
            cerr <<"Fucked up\n";
    }
    
    rand_functor<double>* t_k0;

    //rand_functor<sn_config.get_dist_arrival_type> t_k0(sn_config.get_dist_arrival_params());
    //rand_functor<exponential_distribution<double> > A_k0(sn_config.get_dist_amp_params());
    //rand_functor<normal_distribution<double> > l_k(sn_config.get_dist_length_params());
    //rand_functor<normal_distribution<double> > v_k0(sn_config.get_dist_vrad_params());

    //uniform_real_distribution<double> t_k0(0.0, T_end);
    // Amplitude distribution, scale is 1.0
    //exponential_distribution<double> A_k0(1.0);
    // Cross-field size distribution, normally distributed. Mean = 0.5cm, std=0.1cm
    //normal_distribution<double> l_k(0.5, 0.1);
    // Radial velocity distribution, normally distributed: Mean=500m/s, std=50m/s
    //normal_distribution<double> v_k0(0.05, 0.005);

    // Draw bursts from the distribution
    vector<pulse> pulses_xi0;
    double gamma = 0.0;
    for(size_t k = 0; k < sn_config.get_num_bursts(); k++)
    {
        pulses_xi0.push_back(pulse(A_k0(), l_k(), v_k0(), t_k0(), sn_config.get_tau_par()));
        gamma += pulses_xi0[k].get_tau_d();
    }

    gamma = gamma / double(sn_config.get_num_bursts());
    double xi{0.0}; // Radial position at which we compute the signal
    vector<pulse> pulses_xi; // Vector of pulses propagated to position xi
    double t_k{0.0}; // Arrival time of a pulse at xi, intermediate variable
    double A_k{0.0}; // Amplitude of a pulse at xi, intermediate variable
    //for(size_t xi_idx = 0; xi_idx < num_xi; xi_idx++)
    for(size_t xi_idx = 0; xi_idx < 1; xi_idx++)
    {
        xi = xi_range[xi_idx];
        // Create vector of pulses, propagated to xi.
        pulses_xi.clear();
        for(auto it : pulses_xi0)
        {
            t_k = xi / it.get_v() + it.get_t();
            // Propagate the original pulses to position xi only if it is within time interval
            if(t_k < sn_config.get_Tend())
            {
                A_k = it.get_A() * exp(-xi / (it.get_v() * sn_config.get_tau_par()));
                pulses_xi.push_back(pulse(A_k, it.get_l(), it.get_v(), t_k, sn_config.get_tau_par()));
            }
        }
        // Sort pulses after they are propagated in time
        sort(pulses_xi.begin(), pulses_xi.end(),
                [](const pulse p1, const pulse p2) -> bool {
                    return((p1.get_t() - p2.get_t()) < 1e-8);
                    });

        // Generate signal with the vector of propagated pulses
        try{
            generate_ts_cuda_v2(pulses_xi, signal_xi, sn_config.get_dt(), sn_config.get_nelem());
        } catch (cuda_error err)
        {
            cerr << "Cuda error occured when generating signal" << endl;
            cerr << err.what() << endl;

            delete [] signal_xi;
            return(1);
        }
    }

//    ofstream myfile ("out.txt");
//    if (myfile.is_open())
//    {
//        for(size_t n = 0; n < nelem; n++)
//            myfile << signal_xi[n] << endl;
//        myfile.close();
//    }
//    else 
//    {
//        cerr << "Unable to open file";
//    }


    delete [] signal_xi;
    return(0);
}
