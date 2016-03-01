#ifndef gen_shot_noise
#define gen_shot_noise

#include <string>
#include <exception>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

using namespace std;

enum class dist_t {expon_t, normal_t, uniform_t};

// Functor object for random number generation
// Use a base functor to define operator() and ctor routines
class base_functor{
    public:
        base_functor(double shape_, double scale_) : shape(shape_), scale(scale_) {};
        base_functor(vector<double> params_) : shape(params_[0]), scale(params_[1]) {};

        inline double get_shape() const {return(shape);};
        inline double get_scale() const {return(scale);};

        virtual inline double operator() () = 0;
    private:
        const double shape;
        const double scale;
};

// Use a derived class so we can choose a distribution of a rand_functor instance
// at runtime. F.ex.:
//
// base_functor* b_ptr;
// p_btr = new rand_functor<std::normal_distribution<double> >(5.0, 1.5);

template <class DIST>
class rand_functor : public base_functor{
    public:
        rand_functor(vector<double> params_) : base_functor(params_), generator(chrono::system_clock::now().time_since_epoch().count()), dist(get_shape(), get_scale())  {};
        rand_functor(double shape_, double scale_) : base_functor(shape_, scale_), generator(chrono::system_clock::now().time_since_epoch().count()), dist(get_shape(), get_scale()) {};

        inline double operator() () {return dist(generator);};

    private:
        default_random_engine generator;
        DIST dist;
};

#endif //GEN_SHOT_NOISE
