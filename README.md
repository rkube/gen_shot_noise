# gen_shot_noise
Generate shot noise time series

Time Series is given by 

Psi(t) = \sum_k A_k \phi(t-t_k)

where the distribution of A_k and t_k can be configured in the code.

See the commemnts in the code for setting the distribution parameters and
the length of the time series


Use gen_shot_noise_time for the newest version. It features
* text file configuration, see shotnoise.cfg
* hdf5 output
