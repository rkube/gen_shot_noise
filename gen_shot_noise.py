import numpy as np
import matplotlib.pyplot as plt


#def Shot_noise_gen(g=None, l=None, N=None):
#    # This function returns a shot noise signal with intermittency parameter g,
#    # l is the relation between rise and decay time, and N is the number of
#    # bursts. The function returns the signal and its time steps.
#
#
#    t = mslice[1:N]
#    dt = 1e-2
#    K = N /eldiv/ (g *elmul* dt)
#    tend = N / g
#    time = mslice[dt:dt:tend]
#    Am = 1
#    A = exprnd(Am, N, 1)
#    tevent = rand(N, 1) * tend
#    tevent = sort(tevent)
#    kevent = round(tevent /eldiv/ dt)
#    trevent = kevent *elmul* dt
#    twait = zeros(N - 1, 1)
#    for nn in mslice[1:N - 1]:
#        twait(nn).lvalue = trevent(nn + 1) - trevent(nn)
#    end
#
#    sgnl = zeros(K, 1)
#    S = zeros(K, 1)
#    for i in mslice[1:N]:
#        S(mslice[1:kevent(i)]).lvalue = A(i) *elmul* exp((mslice[-trevent(i) + dt:dt:0]) /eldiv/ l)
#        a = mslice[dt:dt:round((tend - trevent(i)) /eldiv/ dt) *elmul* dt]
#        S(mslice[kevent(i) + 1:end]).lvalue = A(i) *elmul* exp(-(a) /eldiv/ (1 - l))
#        sgnl = sgnl + S
#    end



def gen_shot_noise(g=1.0, l=0.5, N=10000, dt = 1e-2):
    """
    Generate a synthetic shotnoise signal
    g: Intermittency parameter, tau_d / tau_w
    l: tau_rise / tau_fall
    N: Number of bursts
    """
    
    t = np.arange(N)
    K = N/(g*dt)
    t_end = float(N)/g
    time = np.arange(dt, t_end, dt)

    #print 'K = %f, t_end = %f' % (K, t_end)

    Am = 1.
    # Generate N exponentially distributed burst amplitudes
    A = np.random.exponential(1., N)
    # Generate N uniformly distributed burst arrival times
    t_burst = np.random.uniform(size = N) * t_end
    # Sort burst arrival times
    t_burst = np.sort(t_burst)
    t_burst_idx = np.floor(t_burst / dt)
    #print 'Burst arrival times:', t_burst
    #print 'Burst arrival indices:', t_burst_idx
    # Compute the waiting times
    t_wait = t_burst[1:] - t_burst[:-1]

    signal = np.zeros(K, dtype='float64')
    S = np.zeros(K, dtype='float64')
    t_range = np.arange(0, t_end, dt)

    plt.figure()
    # Generate waveforms for each burst and add them up in signal
    for i in np.arange(N):
        if ( i % 100 == 0 ):
            print 'Burst %d/%d' % (i, N)
        # time index of current burst
        bi = t_burst_idx[i]
        # Rise of the burst
        S[:bi] = A[i] * np.exp( -(t_burst[i] - t_range[:bi]) / l )
        # Decay of the burst
        S[bi:] = A[i] * np.exp( -( t_range[bi:] - t_burst[i] )/ (1-l) )
        signal = signal + S

        #plt.plot(t_range, S, 'g')

    plt.plot(t_range, signal)
    plt.plot(t_burst, A, 'ko')

    plt.show()


print 'Hello, world'
print 'Testing synthetic shotnoise'

gen_shot_noise(g=1.0, l=0.1, N=1000)


print 'done'


