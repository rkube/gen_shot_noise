#include <iostream>
#include "include/gen_shot_noise.h"

using namespace std;

template <>
rand_functor<std::exponential_distribution<double> > :: rand_functor(double shape_, double scale_) : shape(shape_), scale(scale_)
{
    exponential_distribution<double> random_distribution(shape);
}


template <>
rand_functor<std::exponential_distribution<double> > :: rand_functor(vector<double> params) : rand_functor(params[0], params[1]) {}
// End of file config.hpp

int main(void)
{
    rand_functor<uniform_real_distribution<double> > uni(0.0, 1.0);
    rand_functor<exponential_distribution<double> > mexp(1.0, 1.0);
    rand_functor<normal_distribution<double> > norm(0.0, 1.0);
    for(size_t t = 0; t < 10; t++)
        cout << uni() << "\t"  << "\t" << norm() << "\t" << mexp() << endl;
}
