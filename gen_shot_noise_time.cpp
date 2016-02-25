/*
 * Generate a shot noise signal where the bursts have individual amplitude,
 * radial velocity, decay time, as well as size distributions
 *
 *
 * Characteristic scales used: Length: 1cm, time: 1 microsecond
 */

#include <iostream>
//#include <boost/program_options>
#include <random>
#include <vector>
#include <array>
#include <algorithm>

using namespace std;


class pulse{
    public:
        ///*
        // double A: Amplitude of the pulse
        // double l: Cross-field size of the pulse
        // double v: Radial velocity of the pulse
        // double t: Arrival time of the pulse
        // double tau: Parallel decay time of the pulse
        ///*
        pulse(double A_, double l_, double v_, double t_, double tau_) : A(A_), l(l_), v(v_), t(t_), tau_par(tau_) {};
        pulse(pulse& rhs) : A(rhs.A), l(rhs.l), v(rhs.v), t(rhs.t), tau_par(rhs.tau_par) {}; 
        pulse(pulse&& rhs) : A(move(rhs.A)), l(move(rhs.l)), v(move(rhs.v)), t(move(rhs.t)), tau_par(rhs.tau_par) {}; 
        /// Returns the pulse amplitude
        double get_A() const {return A;};
        double set_A(double A_) {A = A_;};

        double get_l() const {return l;};
        double set_l(double l_) {l = l_;};

        double get_v() const {return v;};
        double set_v(double v_) {v = v_;};

        double get_t() const {return t;};
        double set_t(double t_) {t = t_;};

        double get_tau_par() const {return tau_par;};
        double set_tau_par(double tau_par_) {tau_par = tau_par_;};

        // tau_d = tau_par * tau_perp / (tau_par + tau_perp), with tau_perp = l/v
        double get_tau_d() const {return (tau_par * (get_l() / get_v()) / (tau_par + (get_l() / get_v())));};

        pulse& operator=(const pulse&);

        friend std::ostream& operator<<(std::ostream& os, pulse& p)
        {
            cout << "A = " << p.get_A() << ", l = " << p.get_l() << ", v = " << p.get_v() << ", t = " << p.get_t() << ", tau_par = " << p.get_tau_par() << endl;
        }

    private:
        double A;
        double l;
        double v;
        double t;
        double tau_par;
};


pulse& pulse::operator=(const pulse& rhs)
{
    A = rhs.get_A();
    l = rhs.get_l();
    v = rhs.get_v();
    t = rhs.get_t();
    tau_par = rhs.get_tau_par();    
}


/// Returns a vector of indices that sorts input by arrival time (get_t())
/// vector<pulse>& input: The vector to sort
/// 
/// See: http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
vector<int> sort_pulse_vector(vector<pulse> &input)
{
    vector<int> idx(input.size());
    for(size_t i = 0; i < input.size(); i++)
        idx[i] = i;

    sort(idx.begin(), idx.end(), [&input](size_t i1, size_t i2) {return (input[i1].get_t() < input[i2].get_t()); });
    return idx;
}


int main(int argc, char* argv[])
{
    // Number of bursts
    constexpr unsigned int K{10000};
    // Time step
    constexpr double dt{0.01};
    // Length of time series
    constexpr double T_end{5e3};
    // Paralll connecion length
    constexpr double L_par{1e4};
    // Ion acoustic velocity
    constexpr double C_s{10};
    // Parallel transit time
    constexpr double tau_par{L_par / C_s};
    // SOL width
    constexpr double L_sol{30.0}; 
    // Number of discretization points of SOL domain
    constexpr size_t num_xi{10};
    constexpr array<double, 10> xi_range{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // Generate distributions objects
    random_device rd;
    mt19937 rndgen(rd());
    // Arrival time distribution, uniformly distributed on [0.0, T_end]
    uniform_real_distribution<double> t_k0(0.0, T_end);
    // Amplitude distribution, scale is 1.0
    exponential_distribution<double> A_k0(1.0);
    // Cross-field size distribution, normally distributed. Mean = 0.5cm, std=0.1cm
    normal_distribution<double> l_k(0.5, 0.1);
    // Radial velocity distribution, normally distributed: Mean=500m/s, std=50m/s
    normal_distribution<double> v_k0(0.05, 0.005);

    // Draw bursts from the distribution
    vector<pulse> pulses_xi0;
    double gamma = 0.0;
    for(size_t k = 0; k < K; k++)
    {
        pulses_xi0.push_back(pulse(A_k0(rd), l_k(rd), v_k0(rd), t_k0(rd), tau_par));
        gamma += pulses_xi0[k].get_tau_d();
    }
    // Sort the bursts by arrival time
    sort(pulses_xi0.begin(), pulses_xi0.end(), [](const pulse &p1, const pulse &p2) -> bool { return(p1.get_t() < p2.get_t());});
    gamma = gamma / double(K);
    cout << "gamma = " << gamma << endl;

    for(auto it: pulses_xi0)
        cout << "t = " << it.get_t() << ", A = " << it.get_A() << ", l = " << it.get_l() << ", v = " << it.get_v() << ", tau_d = " << it.get_tau_d() << endl;

    double xi{0.0}; // The xi
    double t_k{0.0}; // Arrival time of the pulse at xi
    double A_k{0.0}; // Amplitude of pulse at xi
    for(size_t xi_idx = 0; xi_idx < num_xi; xi_idx++)
    {
        xi = xi_range[xi_idx];
        cout << "xi = " << xi << endl;
        vector<pulse> pulse_xi;
        for(auto it : pulses_xi0)
        {
            t_k = xi / it.get_v() + it.get_t();
            if(t_k < T_end)
            {
                A_k = it.get_A() * exp(-xi / (it.get_v() * tau_par));
                pulse_xi.push_back(pulse(A_k, it.get_l(), it.get_v(), t_k, tau_par));
            }
            else{
                cout << "Ignoring pulse " << it << ": t_k = " << t_k <<  endl;
            }
        }

    //for(auto it: pulses_xi0)
    //    cout << "t = " << it.get_t() << ", A = " << it.get_A() << ", l = " << it.get_l() << ", v = " << it.get_v() << ", tau_d = " << it.get_tau_d() << endl;


    }
    return(0);
}
