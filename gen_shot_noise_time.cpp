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
#include <cassert>
#include "include/config.h"
#include "include/output.h"
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
rand_functor<std::exponential_distribution<double> > :: rand_functor(double shape_, double scale_) : base_functor(shape_, scale_), dist(get_shape()) {}

template<>
rand_functor<std::exponential_distribution<double> > :: rand_functor(vector<double> params_) : base_functor(params_), dist(get_shape()) {}


int main(int argc, char* argv[])
{
    config_t sn_config("shotnoise.cfg");
    output_t sn_output(sn_config);
    const double delta_xi{sn_config.get_Lsol() / double(sn_config.get_num_xi())};

    double* signal_xi = new double[sn_config.get_nelem()];

    base_functor* p_t_k0{nullptr};
    base_functor* p_A_k0{nullptr};
    base_functor* p_l_k{nullptr};
    base_functor* p_v_k0{nullptr};

    switch(sn_config.get_dist_arrival_type())
    {
        case dist_t::uniform_t: p_t_k0 = new rand_functor<std::uniform_real_distribution<double> >(sn_config.get_dist_arrival_params()); break;
        case dist_t::expon_t: p_t_k0 = new rand_functor<std::exponential_distribution<double> >(sn_config.get_dist_arrival_params()); break;
        case dist_t::normal_t: p_t_k0 = new rand_functor<std::normal_distribution<double> >(sn_config.get_dist_arrival_params()); break;
    }

    switch(sn_config.get_dist_amp_type())
    {
        case dist_t::uniform_t: p_A_k0 = new rand_functor<std::uniform_real_distribution<double> >(sn_config.get_dist_amp_params()); break;
        case dist_t::expon_t: p_A_k0 = new rand_functor<std::exponential_distribution<double> >(sn_config.get_dist_amp_params()); break;
        case dist_t::normal_t: p_A_k0 = new rand_functor<std::normal_distribution<double> >(sn_config.get_dist_amp_params()); break;
    }

    switch(sn_config.get_dist_length_type())
    {
        case dist_t::uniform_t: p_l_k = new rand_functor<std::uniform_real_distribution<double> >(sn_config.get_dist_length_params()); break;
        case dist_t::expon_t: p_l_k = new rand_functor<std::exponential_distribution<double> >(sn_config.get_dist_length_params()); break;
        case dist_t::normal_t: p_l_k = new rand_functor<std::normal_distribution<double> >(sn_config.get_dist_length_params()); break;
    }

    switch(sn_config.get_dist_vrad_type())
    {
        case dist_t::uniform_t: p_v_k0 = new rand_functor<std::uniform_real_distribution<double> >(sn_config.get_dist_length_params()); break;
        case dist_t::expon_t: p_v_k0 = new rand_functor<std::exponential_distribution<double> >(sn_config.get_dist_vrad_params()); break;
        case dist_t::normal_t: p_v_k0 = new rand_functor<std::normal_distribution<double> >(sn_config.get_dist_vrad_params()); break;
    }

    assert(p_t_k0 != nullptr);
    assert(p_A_k0 != nullptr);
    assert(p_l_k != nullptr);
    assert(p_v_k0 != nullptr);

    // Draw bursts from the distribution
    vector<pulse> pulses_xi0;
    double gamma = 0.0;
    for(size_t k = 0; k < sn_config.get_num_bursts(); k++)
    {
        pulses_xi0.push_back(pulse((*p_A_k0)(), (*p_l_k)(), (*p_v_k0)(), (*p_t_k0)(), sn_config.get_tau_par()));
        gamma += pulses_xi0[k].get_tau_d();
    }

    gamma = gamma / double(sn_config.get_num_bursts());
    double xi{0.0}; // Radial position at which we compute the signal
    vector<pulse> pulses_xi; // Vector of pulses propagated to position xi
    double t_k{0.0}; // Arrival time of a pulse at xi, intermediate variable
    double A_k{0.0}; // Amplitude of a pulse at xi, intermediate variable


    for(auto it : pulses_xi0)
        cout << it;

    for(size_t xi_idx = 0; xi_idx < sn_config.get_num_xi(); xi_idx++)
    {
        xi = xi_idx * delta_xi;
        cout << xi_idx << " / " << sn_config.get_num_xi() << ", xi = " << xi << endl;
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
            } else {
                cout << "ignoring burst at t_k = " << t_k << "(t_k0 = " << it.get_t() << ", v = " << it.get_v() << ")" << endl;
            }
        }
        // Sort pulses after they are propagated in time
        sort(pulses_xi.begin(), pulses_xi.end(),
                [](const pulse p1, const pulse p2) -> bool {
                    return((p1.get_t() - p2.get_t()) < 1e-8);
                    });

        for(auto it : pulses_xi)
            cout << it;

        // Generate signal with the vector of propagated pulses
        try{
            generate_ts_cuda_v2(pulses_xi, signal_xi, sn_config.get_dt(), sn_config.get_nelem());

            sn_output.write_to_tlev(xi_idx, signal_xi);
        } 
        catch (cuda_error err)
        {
            cerr << "Cuda error occured when generating signal" << endl;
            cerr << err.what() << endl;

            delete [] signal_xi;
            return(1);
        }

        catch(output_error err)
        {
            cerr << "Output error occured while writing for xi = " << xi_idx << endl;
            cerr << err.what() << endl;

            delete[] signal_xi;
            return(1);
        }
    }

    ofstream myfile ("out.txt");
    if (myfile.is_open())
    {
        for(size_t n = 0; n < sn_config.get_nelem(); n++)
            myfile << signal_xi[n] << endl;
        myfile.close();
    }
    else 
    {
        cerr << "Unable to open file";
    }


    delete [] signal_xi;
    return(0);
}
