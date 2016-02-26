#ifndef datatypes_h
#define datatypes_h

#include <iostream>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#endif

#ifndef __CUDACC__
#define CUDA_CALLABLE
#endif
class pulse{
    public:
        ///*
        // double A: Amplitude of the pulse
        // double l: Cross-field size of the pulse
        // double v: Radial velocity of the pulse
        // double t: Arrival time of the pulse
        // double tau: Parallel decay time of the pulse
        ///*
        CUDA_CALLABLE
        pulse(double A_, double l_, double v_, double t_, double tau_) : A(A_), l(l_), v(v_), t(t_), tau_par(tau_) {};
        pulse() : A(1.0), l(1.0), v(1.0), t(1.0), tau_par(0.1) {};
        //pulse(pulse& rhs) : A(rhs.A), l(rhs.l), v(rhs.v), t(rhs.t), tau_par(rhs.tau_par) {}; 
        //pulse(pulse&& rhs) : A(std::move(rhs.A)), l(std::move(rhs.l)), v(std::move(rhs.v)), t(std::move(rhs.t)), tau_par(rhs.tau_par) {}; 
        
        /// Returns the pulse amplitude
        CUDA_CALLABLE
        double get_A() const {return A;};
        CUDA_CALLABLE
        void set_A(double A_) {A = A_;};

        CUDA_CALLABLE
        double get_l() const {return l;};
        CUDA_CALLABLE
        void set_l(double l_) {l = l_;};

        CUDA_CALLABLE
        double get_v() const {return v;};
        CUDA_CALLABLE
        void set_v(double v_) {v = v_;};

        CUDA_CALLABLE
        double get_t() const {return t;};
        CUDA_CALLABLE
        void set_t(double t_) {t = t_;};

        CUDA_CALLABLE
        double get_tau_par() const {return tau_par;};
        CUDA_CALLABLE
        void set_tau_par(double tau_par_) {tau_par = tau_par_;};

        // tau_d = tau_par * tau_perp / (tau_par + tau_perp), with tau_perp = l/v
        CUDA_CALLABLE
        double get_tau_d() const {return (tau_par * (get_l() / get_v()) / (tau_par + (get_l() / get_v())));};

        // Returns the time index when the burst occurs, assuming sampling rate is dt and t_start = 0
        CUDA_CALLABLE
        size_t get_tidx(double dt) const{ return(size_t(t /dt));};
      
        CUDA_CALLABLE
        double evaluate(double tau) const {return (get_A() * exp((get_t() - tau) / get_tau_d()));};

        friend std::ostream& operator<<(std::ostream& os, const pulse& p)
        {
            os << "A = " << p.get_A() << ", t = " << p.get_t() << ", v = " << p.get_v() << ", l = " << p.get_l() << ", tau_d = " << p.get_tau_d() << std::endl;
            return(os);
        }

    private:
        // The pulse amplitude
        double A;
        // Pulse cross-field size
        double l;
        // Pulse radial velocity
        double v;
        // Pulse arrival time
        double t;
        // Pulse parallel time scale
        double tau_par;
};

#endif // datatypes_h
