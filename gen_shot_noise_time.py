#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
Use the following dimensional quantities for normalization
L_perp = 1cm
tau = 1 mu s
"""

# Number of bursts
K = 5000
dt = 1e-2 
T_end = 5e3
L_par = 1e4
C_s = 10.
L_sol = 30.0
dt = 1e-1

##################################################################
# Initial distributions of the pulses. Index 0 implies that this
# distribution will be sorted when iterating over xi
##################################################################

# Amplitude distribution A_k0 of the pulses
A_k0 = np.random.exponential(scale=1.0, size=K)
# Initial arrival time distribution of the pulses
t_k0 = np.random.uniform(0.0, T_end, size=K)
t_k0.sort()
# Length distribution of the pulses
l_k = np.random.normal(loc=0.5, scale=1e-2, size=K)
# Velocity distribution of the pulses
v_k0 = np.random.normal(loc=5e-2, scale=5e-3, size=K)

tau_perp = l_k / v_k0
tau_par = L_par / C_s
tau_d_k0 = tau_par * tau_perp / (tau_par + tau_perp)
tau_w = T_end / K
gamma = tau_d_k0.mean() / tau_w
print 'gamma = %f' % gamma
# Position where we evaluate the profile
xi_range = np.arange(0.0, L_sol, 1.0)

# Number of points in time direction
#N = int(K / gamma)
N = int(T_end / dt)
print 'N = %d' % N

signal = np.zeros([xi_range.size, N], dtype='float64')

N_rg = np.arange(N)
t_rg = N_rg * dt

plt.figure()
for xi_idx, xi in enumerate(xi_range):
    print 'xi = %4.2f' % xi
    # Gnerate the arrival time distribution of the bursts
    t_k = xi / v_k0 + t_k0
    # Apply exponential damping to burst amplitudes
    A_k = A_k0 * np.exp(-xi / (v_k0 * tau_par))

    # Remove the bursts that are out of bounds
    good_burst_idx = t_k < T_end
    # print 'bursts in range: %d/%d' % (good_burst_idx.sum(), good_burst_idx.size)
    # Purge and all initial distributions of blob parameters 
    t_k = t_k[good_burst_idx]
    A_k = A_k[good_burst_idx]
    tau_d_k = tau_d_k0[good_burst_idx]
    v_k = v_k0[good_burst_idx]
    # Sort all distributions such that the arrival times are ordered 
    idx_sort = t_k.argsort()
    t_k = t_k[idx_sort]
    A_k = A_k[idx_sort]
    tau_d_k = tau_d_k[idx_sort]
    v_k = v_k[idx_sort]

    # Generate list of indices when the bursts arrive and build the signal
    t_burst_idx = np.floor(t_k / dt).astype('int')
    #print 't_burst_idx = ', t_burst_idx

    num_bursts = t_k.size
    for burst in np.arange(num_bursts):
        burst_tidx = t_burst_idx[burst]
        #print 'burst %d, tidx=%d, A_k = %f, t_k=%f, taud_=%f, v_k=%f' % (burst, burst_tidx, A_k[burst], t_k[burst], tau_d_k[burst], v_k[burst])
        signal[xi_idx, burst_tidx:] += A_k[burst] * np.exp( -(t_rg[burst_tidx:] - t_k[burst]) / tau_d_k[burst])


plt.figure()
plt.plot(tau_d_k, '.')

plt.figure()
for i in np.arange(signal.shape[0]):
    plt.plot(t_rg, signal[i, :] + i + signal[i, :].mean())
plt.xlabel(r"t")
plt.ylabel(r"$\Phi(t)$")

plt.figure()
plt.semilogy(xi_range, signal[:, 4000:40000].mean(axis=1))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\langle \Phi \rangle$")

plt.show()
# End of file gen_shot_noise_time.py
