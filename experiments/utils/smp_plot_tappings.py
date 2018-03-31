import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

tappings = pd.read_csv('dm-imol-pre_l0-tappings.csv')

print(tappings.describe)

fig = plt.figure()


tmin = 0
tmax = 0
vmin = 0
vmax = 0

mks = ['inv', 'fwd']
mvars = {}

for mk in mks:
    for tk in ['X_fit', 'Y_fit', 'X_pre']:
        # tappings
        t_ = tappings[tappings.mk == mk][tappings.tapk == tk]
        tapi = t_.tapi
        tapi__ = np.array([int(tapi_[1:-1]) for tapi_ in tapi.tolist()])
        if np.min(tapi__) < tmin:
            tmin = np.min(tapi__)
        if np.max(tapi__) > tmax:
            tmax = np.max(tapi__)
        # mvars.append(list(t_.ch))
        mvars.update(zip(list(t_.ch), list(t_.ch)))

for i, k in enumerate(mvars):
    mvars[k] = i
    
print('mvars = %s' % mvars)
print('mks = %s' % mks)
print('tmax = %s %s' % (type(tmax), tmax))
print('tmin = %s %s' % (type(tmin), tmin))

numtaps = tmax - tmin
print('numtaps = %s' % numtaps)
        
gs = GridSpec(len(mks), 1)
cs = ['r', 'g', 'b', 'c', 'm', 'y']

for i, mk in enumerate(mks):
    ax = fig.add_subplot(gs[i])
    ax.set_title('tap %s' % mk)
    ax.grid()
    for j, tk in enumerate(['X_fit', 'Y_fit', 'X_pre']):
        # tappings
        t_ = tappings[tappings.mk == mk][tappings.tapk == tk]
        tapi = t_.tapi
        tapi__ = np.array([int(tapi_[1:-1]) for tapi_ in tapi.tolist()])
        xs_ = [mvars[k] for k in list(t_.ch)]
        ax.plot(tapi__, xs_, c=cs[j], linestyle='', marker='o')
fig.show()
plt.show()
