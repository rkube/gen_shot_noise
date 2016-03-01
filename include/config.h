#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "gen_shot_noise.h"

class config_t{
    public:
        config_t(std::string);

        inline size_t get_num_bursts() const {return(num_bursts);};
        inline double get_dt() const {return(dt);};
        inline double get_Tend() const {return(T_end);};
        inline double get_Lpar() const {return(L_par);};
        inline double get_Cs() const {return(C_s);};
        inline double get_Lsol() const {return(L_sol);}; 

        inline size_t get_nelem() const {return(size_t(T_end / dt));};
        inline double get_tau_par() const {return(L_par / C_s);};

        inline size_t get_num_xi() const {return(num_xi);};

        inline dist_t get_dist_vrad_type() const {return (dist_vrad_type);};
        inline std::vector<double> get_dist_vrad_params() const {return(dist_vrad_params);};

        inline dist_t get_dist_amp_type() const {return (dist_amp_type);};
        inline std::vector<double> get_dist_amp_params() const {return(dist_amp_params);};
        
        inline dist_t get_dist_length_type() const {return (dist_length_type);};
        inline std::vector<double> get_dist_length_params() const {return(dist_length_params);};
        
        inline dist_t get_dist_arrival_type() const {return (dist_arrival_type);};
        inline std::vector<double> get_dist_arrival_params() const {return(dist_arrival_params);};

        inline  std::string get_outfile_name() const {return(outfile_name);};
    private:
        size_t num_bursts;  // Number of bursts
        double dt;          // Time stepping
        double T_end;       // End time
        double L_par;       // Parallel length
        double C_s;         // Acoustic velocity
        double L_sol;       // SOL width
        size_t num_xi;      // Number of radial positions

        // Vrad distribution and parameters
        dist_t dist_vrad_type;
        std::vector<double> dist_vrad_params;

        // Amplitude distribution and parameters
        dist_t dist_amp_type;
        std::vector<double> dist_amp_params;

        // Size distribution and parameters
        dist_t dist_length_type;
        std::vector<double> dist_length_params;

        // Arrival time distribution and parameters
        dist_t dist_arrival_type;
        std::vector<double> dist_arrival_params;
    
        // Filename where we write output to
        string outfile_name;
};

#endif //CONFIG_H
