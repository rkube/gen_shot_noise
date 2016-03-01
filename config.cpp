#include "include/config.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Split at whitespace and put values in double vector
void split_at_whitespace(const string params, vector<double>& result)
{
    vector<string> split_vector;
    boost::split(split_vector, params, boost::is_any_of(" \t"), boost::token_compress_on);
    for(auto it: split_vector)
    {
        try{
        result.push_back(boost::lexical_cast<double> (it));
        } catch (boost::bad_lexical_cast&) {
            cerr << "Could not cast model parameter " << it << " to double\n";
        }
    }
}



const map<string, dist_t> distribution_map{
    {"exponential", dist_t::expon_t},
    {"normal", dist_t::normal_t},
    {"uniform", dist_t::uniform_t},
};

config_t :: config_t(string cfg_file) : dist_vrad_type(dist_t::normal_t), 
                                        dist_amp_type(dist_t::expon_t), 
                                        dist_length_type(dist_t::normal_t), 
                                        dist_arrival_type(dist_t::uniform_t)
{

    po::options_description cf_opts("Allowed options");
    po::variables_map vm;

    fs::path dir(string("."));
    fs::path file(string("shotnoise.cfg"));
    fs::path full_path = dir / file;
    cout << "\tfs::path = " << full_path << endl;

    string dist_vrad_str;
    string dist_vrad_params_str;
    string dist_amp_str;
    string dist_amp_params_str;
    string dist_length_str;
    string dist_length_params_str;
    string dist_arrival_str;
    string dist_arrival_params_str;

    try{
        cf_opts.add_options()
            ("K", po::value<size_t>(&num_bursts), "Number of bursts")
            ("dt", po::value<double>(&dt), "time step")
            ("Tend", po::value<double>(&T_end), "Final time")
            ("Lpar", po::value<double>(&L_par), "Parallel length scale")
            ("Cs", po::value<double>(&C_s), "Acoustic velocity")
            ("Lsol", po::value<double>(&L_sol), "SOL width")
            ("vrad_dist", po::value<string>(&dist_vrad_str), "Distribution of pulse velocities")
            ("vrad_dist_params", po::value<string>(&dist_vrad_params_str), "Parameters for distribution of pulse velocities")
            ("amp_dist", po::value<string>(&dist_amp_str), "Distribution of pulse amplitudes")
            ("amp_dist_params", po::value<string>(&dist_amp_params_str), "Parameters for distribution of pulse amplitudes")
            ("length_dist", po::value<string>(&dist_length_str), "Distribution of pulse lengths")
            ("length_dist_params", po::value<string>(&dist_length_params_str), "Parameters for distribution of pulse lengths")
            ("arrival_dist", po::value<string>(&dist_arrival_str), "Distribution of pulse arrival times ")
            ("arrival_dist_params", po::value<string>(&dist_arrival_params_str), "Parameters for distribution of pulse arrival times")
            ("num_xi", po::value<size_t>(&num_xi), "Number of radial positions")
            ("outfile_name", po::value<string>(&outfile_name), "Name of the output file");
    } catch(exception &e)
    {
        cerr << e.what() << endl;
    }

    // Parse configuration file
    ifstream cf_stream(full_path.string());
    if (!cf_stream)
    {
        cerr << "slab_config::slab_config(string): Could not open config file: " << full_path.string() << "\n";
        exit(1);
    }
    else
    {
        po::store(po::parse_config_file(cf_stream, cf_opts), vm);
        po::notify(vm);
    }

    dist_amp_type = distribution_map.at(dist_amp_str);
    dist_amp_params.clear();
    split_at_whitespace(dist_amp_params_str, dist_amp_params);

    dist_vrad_type = distribution_map.at(dist_vrad_str);
    dist_vrad_params.clear();
    split_at_whitespace(dist_vrad_params_str, dist_vrad_params);

    dist_length_type = distribution_map.at(dist_length_str);
    dist_length_params.clear();
    split_at_whitespace(dist_length_params_str, dist_length_params);

    dist_arrival_type = distribution_map.at(dist_arrival_str);
    dist_arrival_params.clear();
    split_at_whitespace(dist_arrival_params_str, dist_arrival_params);
}


// End of file config.hpp
