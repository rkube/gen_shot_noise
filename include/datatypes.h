#ifndef datatypes_h
#define datatypes_h

#include <iostream>

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
        //pulse(pulse& rhs) : A(rhs.A), l(rhs.l), v(rhs.v), t(rhs.t), tau_par(rhs.tau_par) {}; 
        //pulse(pulse&& rhs) : A(std::move(rhs.A)), l(std::move(rhs.l)), v(std::move(rhs.v)), t(std::move(rhs.t)), tau_par(rhs.tau_par) {}; 
        
        /// Returns the pulse amplitude
        double get_A() const {return A;};
        void set_A(double A_) {A = A_;};

        double get_l() const {return l;};
        void set_l(double l_) {l = l_;};

        double get_v() const {return v;};
        void set_v(double v_) {v = v_;};

        double get_t() const {return t;};
        void set_t(double t_) {t = t_;};

        double get_tau_par() const {return tau_par;};
        void set_tau_par(double tau_par_) {tau_par = tau_par_;};

        // tau_d = tau_par * tau_perp / (tau_par + tau_perp), with tau_perp = l/v
        double get_tau_d() const {return (tau_par * (get_l() / get_v()) / (tau_par + (get_l() / get_v())));};

        //pulse& operator=(const pulse&);

        friend std::ostream& operator<<(std::ostream& os, const pulse& p)
        {
            os << "A = " << p.get_A() << ", l = " << p.get_l() << ", v = " << p.get_v() << ", t = " << p.get_t() << ", tau_par = " << p.get_tau_par() << std::endl;
            return(os);
        }

    private:
        double A;
        double l;
        double v;
        double t;
        double tau_par;
};


/// This class stores the pulse parameters as used by the cuda routine 
/// to add to a signal
#ifdef CUDACC
__host__ __device__
#endif // CUDACC
struct pulse_params{
    double amplitude;
    double taud;
    size_t tidx;
};

#endif // datatypes_h
