#include <iostream>
#include "include/gen_shot_noise.h"

using namespace std;

template <>
rand_functor<std::exponential_distribution<double> > :: rand_functor(double shape_, double scale_) : base_functor(shape_, scale_), dist(get_shape()) {}

template <>
rand_functor<std::exponential_distribution<double> > :: rand_functor(vector<double> params_) : base_functor(params_), dist(get_shape()) {}


// See examples on cplusplus.com
// http://www.cplusplus.com/reference/random/uniform_real_distribution/
int main(void)
{
    constexpr int nrolls{10000};
    constexpr int nstars{95};
    constexpr int nintervals{10};

    vector<int> p_exp(nintervals, 0);
    vector<int> p_uni(nintervals, 0);
    vector<int> p_nor(nintervals, 0);


    base_functor* expon_ptr;
    expon_ptr = new rand_functor<std::exponential_distribution<double> >(3.5, 1.0);
    base_functor* uni_ptr;
    uni_ptr = new rand_functor<std::uniform_real_distribution<double> >(0.0, 1.0);
    base_functor* nor_ptr;
    nor_ptr = new rand_functor<std::normal_distribution<double> >(5.0, 1.5);

    double number{0.0};
    int idx;
    for(int t = 0; t < nrolls; t++)
    {
        number = (*expon_ptr)();
        idx = int(nintervals * number);
        if (idx < nintervals)
            p_exp[idx]++;

        number = (*uni_ptr)();
        idx = int(nintervals * number);
        if (idx < nintervals)
            p_uni[idx]++;

        number = (*nor_ptr)();
        if (number > 0.0 && number < double(nintervals))
            p_nor[int(floor(number))]++;
    }
    
    cout << "Exponential distribution: (3.5)" << endl;
    for(int t = 0; t < nintervals; t++)
    {
        cout << double(t) / nintervals << "-" << double(t + 1) / nintervals << ": ";
        cout << string(p_exp[t] * nstars / nrolls, '*') << endl;
    }

    cout << "Uniform distribution: (0.0, 1.0)" << endl;
    for(int t = 0; t < nintervals; t++)
    {
        cout << double(t) / nintervals << "-" << double(t + 1) / nintervals << ": ";
        cout << string(p_uni[t] * nstars / nrolls, '*') << endl;
    }

    cout << " Normal distribution: (5.0, 1.5)" << endl;
    for(int t = 0; t < nintervals; t++)
    {
        cout << double(t) << "-" << double(t + 1) << ": ";
        cout << string(p_nor[t] * nstars / nrolls, '*') << endl;
    }
}
