#ifndef GEN_SHOT_NOISE
#define GEN_SHOT_NOISE

#include <string>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

using namespace std;

enum class dist_t {expon_t, normal_t, uniform_t};

class config{
    public:
        config(string);

        inline size_t get_num_bursts() const {return(num_bursts);};
        inline double get_dt() const {return(dt);};
        inline double get_Tend() const {return(T_end);};
        inline double get_Lpar() const {return(L_par);};
        inline double get_Cs() const {return(C_s);};
        inline double get_Lsol() const {return(L_sol);}; 

        inline size_t get_nelem() const {return(size_t(T_end / dt));};
        inline double get_tau_par() const {return(L_par / C_s);};

        inline dist_t get_dist_vrad_type() const {return (dist_vrad_type);};
        inline vector<double> get_dist_vrad_params() const {return(dist_vrad_params);};

        inline dist_t get_dist_amp_type() const {return (dist_amp_type);};
        inline vector<double> get_dist_amp_params() const {return(dist_amp_params);};
        
        inline dist_t get_dist_length_type() const {return (dist_length_type);};
        inline vector<double> get_dist_length_params() const {return(dist_length_params);};
        
        inline dist_t get_dist_arrival_type() const {return (dist_arrival_type);};
        inline vector<double> get_dist_arrival_params() const {return(dist_arrival_params);};

    private:
        size_t num_bursts;   // Number of bursts
        double dt;  // Time stepping
        double T_end;    // End time
        double L_par;    // Parallel length
        double C_s;      // Acoustic velocity
        double L_sol;   // SOL width

        // Vrad distribution and parameters
        dist_t dist_vrad_type;
        vector<double> dist_vrad_params;

        // Amplitude distribution and parameters
        dist_t dist_amp_type;
        vector<double> dist_amp_params;

        // Size distribution and parameters
        dist_t dist_length_type;
        vector<double> dist_length_params;

        // Arrival time distribution and parameters
        dist_t dist_arrival_type;
        vector<double> dist_arrival_params;
};


// Functor object for random number generation
class rand_functor{
    public:
        typedef void (rand_functor *dist_fun_ptr)();
        // vector<double> has two parameters: shape and scale.
        rand_functor(vector<double>, dist_t);
        rand_functor(double, double, dist_t); 
        inline double operator() ();

        inline double rng_expon() const {return(rng_expon(generator));};
        inline double rng_uniform() const {return(rng_uniform(generator));};
        inline double rng_normal() const {return(rng_normal(generator));};

    private:
        const double shape;
        const double scale;
        const dist_t type;
        default_random_engine generator;

        exponential_distribution<double> dist_exp;
        normal_distribution<double> dist_norm;
        uniform_distribution<double> dist_uni;

        static map<dist_t, dist_fun_ptr> create_rng_map {
            map<dist_t, dist_fun_ptr> my_map;
            my_map[dist_t::expon_t] = &rng_expon();
            my_map[dist_t::normal_t] = &rng_normal();
            my_map[dist_t::uniform_t] = &rng_uniform();
            return(my_map);
        };
};

rand_functor :: rand_functor(double shape_, double scale_, dist_t type_) : shape(shape_), scale(scale_), type(type_),
    dist_exp(shape),
    dist_norm(shape, scale),
    dist_uni(shape, scale)
{
}

#endif //GEN_SHOT_NOISE
