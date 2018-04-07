"""plot tappings with matplotlib from smp_graphs config

tappings plots: inkscape, tikz, mpl
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from smp_base.plot_utils import put_legend_out_right
from smp_base.plot import makefig

# tap spec file
# tapspecfile = 'dm-imol-pre_l0-tappings-3.csv'
tapspecfile = 'dm-actinf-tappings-pre_l0.csv'
tapspecfile_base = tapspecfile[:-4]

# get the data
# columns = ['mk','tapk','ch','tapi']
# tappings = pd.read_csv('dm-imol-pre_l0-tappings.csv', names = columns, dtype = zip(columns, [str, str, str, np.int64]))
# tappings = pd.read_csv('dm-imol-pre_l0-tappings.csv')
tappings = pd.read_csv(tapspecfile)

# debug
print('plot_tappings: tappings = \n%s' % tappings.describe())

# get model keys
# mks = ['inv', 'fwd']
mks = list(tappings.mk.unique())
print('plot_tappings: model keys = %s' % mks)
mtapk = dict(zip(tappings.tapk.unique(), range(len(tappings.tapk.unique()))))
print('plot_tappings: model taps = %s' % mtapk)
mvars = dict(zip(sorted(tappings.ch.unique()), range(len(tappings.ch.unique()))))
print('plot_tappings: model vars = %s' % mvars)

# limits
tmin = 0
tmax = 0
vmin = 0
vmax = 0

print('tappings.tapi.dtype = %s' % tappings.tapi.dtype)
print('tappings.tapi.unqiue = %s' % (tappings.tapi.unique()))
tapi_list = np.array([eval(l) for l in list(tappings.tapi)])
# tapi_list = np.array([l + str(tappings.loc[i].ch).startswith('attr') for i, l in enumerate(tapi_list)])
print('tapi_list = %s' % tapi_list.T)
# tappings.tapi = tapi_list
# print('tapi', tappings.tapi.unique())

# tmin = tappings.tapi.min()
# tmax = tappings.tapi.max()
tmin = np.min(tapi_list)
tmax = np.max(tapi_list)
print('plot_tappings: tap index min/max = %s / %s' % (tmin, tmax))

# tapo = list(tappings.tapo)
# print('plot_tappings: tapo = %s' % (tapo, ))

# for mk in mks:
#     # for tk in ['X_fit', 'Y_fit', 'X_pre']:
#     for tk in mtapk:
#         # tappings
#         # t_ = tappings[tappings.mk == mk][tappings.tapk == tk]
#         t_ = tappings[(tappings.mk == mk) & (tappings.tapk == tk)]
#         tapi = t_.tapi
#         tapi___ = np.array([int(tapi_[1:-1]) for tapi_ in tapi.tolist()])

#         t_index_ = t_.index.values # .tolist()
#         tapi__ = tapi_list[t_index_].ravel()
#         print('t_.index = %s' % t_.index.values)
#         print('tapi__ = %s' % tapi__)
#         print('tapi___ = %s' % tapi___)
#         if np.min(tapi__) < tmin:
#             tmin = np.min(tapi__)
#         if np.max(tapi__) > tmax:
#             tmax = np.max(tapi__)
#         # mvars.append(list(t_.ch))
#         # mvars.update(zip(list(t_.ch), list(t_.ch)))

# for i, k in enumerate(mvars):
#     mvars[k] = i
    
print('mvars = %s' % mvars)
print('mks = %s' % mks)
print('tmax = %s %s' % (type(tmax), tmax))
print('tmin = %s %s' % (type(tmin), tmin))

numtaps = tmax - tmin
print('numtaps = %s' % numtaps)
        
# create figure
# fig = plt.figure()
rows = len(mks)
cols = 1
fig = makefig(rows, cols, wspace=0.2, hspace=0.2)

# gs = GridSpec(len(mks), 1)
cs = ['r', 'g', 'b', 'c', 'm', 'y']

# second pass

# for i, mk in enumerate(mks):
#     ax = fig.add_subplot(gs[i])
#     ax.set_title('tap %s' % mk)
#     ax.grid()
#     for j, tk in enumerate(['X_fit', 'Y_fit', 'X_pre']):
#         # tappings
#         t_ = tappings[tappings.mk == mk][tappings.tapk == tk]
#         tapi = t_.tapi
#         tapi__ = np.array([int(tapi_[1:-1]) for tapi_ in tapi.tolist()])
#         tapi__ = tapi__.astype(np.float) + j * 0.1
#         xs_ = [mvars[k] for k in list(t_.ch)]
#         ax.plot(
#             tapi__, xs_, c=cs[j],
#             linestyle='',
#             marker='o', markersize=10, markeredgecolor='k', markeredgewidth=2,
#             alpha=0.6)
#         # ax.set_yticklabels(list(t_.ch))

altval = [0.1, 0.2]
for i, mk in enumerate(mks):
    # ax = fig.add_subplot(gs[i])
    ax = fig.axes[i]
    ax.set_title('tap %s' % mk)
    ax.grid(b=False)
    ax.set_xticks([])
    for j, tk in enumerate(mtapk):
        # tappings
        t_ = tappings[(tappings.mk == mk) & (tappings.tapk == tk)]
        tapi = t_.tapi
        tapo = list(t_.tapo)
        tapi___ = np.array([int(tapi_[1:-1]) for tapi_ in tapi.tolist()])

        t_index_ = t_.index.values # .tolist()
        tapi__ = tapi_list[t_index_].ravel()
        
        print('t_.index = %s' % t_.index.values)
        print('tapi__ = %s' % tapi__)
        print('tapi___ = %s' % tapi___)
        print('tapo = %s' % tapo)

        ys_ = np.array([mvars[k] for k in list(t_.ch)]) # + (j * 0.1)
        print('ys_ = %s' % ys_)
        tapi__ = tapi__.astype(np.float) + (j * 0.33 - 0.33)
        for _i, ch_i in enumerate(list(t_.ch)):
            # print('ch_i = %s' % ch_i)
            if 'attr.' in ch_i:
                tapi__[_i] += 1.0
        ax.plot(
            tapi__, ys_, c=cs[j],
            linestyle='',
            marker='o', markersize=18, markeredgecolor='k', markeredgewidth=2,
            label=tk,
            alpha=0.6)

        [ax.text(tapi__[i_] - 0.05, ys_[i_] - 0.15, tapo[i_]) for i_ in tapo]
                
    [ax.axhspan(mval - 0.5, mval+ 0.5, color='k', edgecolor=None, alpha=altval[mval%2]) for mval in mvars.values()]
    [ax.axvspan(tval - 0.5, tval+ 0.5, color='k', edgecolor=None, alpha=altval[tval%2]) for tval in range(tmin, tmax+1)]

    ax.set_aspect(0.33)
    xlim = list(ax.get_xlim())
    xlim[1] += 1
    ax.set_xlim(xlim)
    
    # list(t_.ch)
    ax.set_yticks(sorted(mvars.values()))
    ax.set_yticklabels(sorted(mvars, key=mvars.get))
    ax.set_ylabel('variables')
    # ax.legend()
    put_legend_out_right(ax=ax)
    if i == len(mks)-1:
        ax.set_xticks(range(tmin, tmax+1))
        ax.set_xlabel('time steps')

    
    fig.set_size_inches((cols * 3 * 4, rows * 4))
    fig.savefig('%s.pdf' % tapspecfile_base, dpi=300, bbox_inches='tight')
        
fig.show()
plt.show()
