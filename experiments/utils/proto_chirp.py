

import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft

def ef(X):
    # return X
    # return np.exp(X)
    return np.power(X, 0.8)

methods = ['exp', 'linear', 'quadratic', 'logarithmic', 'hyperbolic'] # , optional
# Kind of frequency sweep.  If not given, `linear` is assumed.  See
# Notes below for more details.

gs = gridspec.GridSpec(2, len(methods))
fig = plt.figure()

N = 10000
T1 = 500
sr = 20
t = np.linspace(0, T1, N + 1)

for i, method in enumerate(methods):
    if method == 'exp':
        method = 'linear'
        t_ = t
        t = (ef(t) / ef(T1)) * T1
    else:
        t_ = t
        
    print("t", t)
    
    chirp = sig.chirp(t = t, f0 = 1e-3, t1 = T1, f1 = sr/2, method = method)
    # CHIRP = fft.rfft(chirp)
    F, T, CHIRP = sig.spectrogram(chirp, sr)
    ax = fig.add_subplot(gs[i])
    ax.set_title("%s" % (method,))
    ax.plot(chirp)
    
    ax = fig.add_subplot(gs[i+len(methods)])
    ax.set_title("Spectrum %s" % (method,))
    ax.pcolormesh(T, F, CHIRP)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')

    t = t_

    

plt.show()
