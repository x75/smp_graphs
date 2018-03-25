# coding: utf-8
"""smp_graphs thesis batch learning vs. online learning

Make illustrative plot that describes the transition from standard
batch learning to single time step incremental learning.
"""
import numpy as np
import matplotlib.pyplot as plt

durs = np.logspace(1, 10, 4, base=2)[::-1]
# fig = plt.figure(figsize=(12, 6))
fig = plt.figure(figsize=(6, 12))
episodes = []
for j, dur in enumerate(durs):
    episodes.append(np.random.uniform(-1, 1, (int(dur), 2)))
    episodes[-1][0,:] = 0.
    for i, t_i in enumerate(episodes[-1][1:]):
        episodes[-1][i+1] = episodes[-1][i] + np.random.normal(0, 0.1, episodes[-1][i].shape)
        # np.putmask(episodes[-1][i+1], np.abs(episodes[-1][i+1]) > np.array([1, 1]), episodes[-1][i+1] + 2 * (1 - np.abs(episodes[-1][i+1])))
        res_ = episodes[-1][i+1] - np.clip(episodes[-1][i+1], -1, 1)
        episodes[-1][i+1] -= 2*res_
    # ax = fig.add_subplot(2, 4, j+1)
    ax = fig.add_subplot(4, 2, (j*2)+1)
    ax.plot(episodes[-1][:,0], episodes[-1][:,1], 'b-', marker=',', alpha=0.5)
    ax.plot(episodes[-1][-1,0], episodes[-1][-1,1], 'bo', marker='o', alpha=0.8)
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    # ax.set_title('Episode of %d steps' % (dur-1))
    ax.set_ylabel('Episode of %d steps' % (dur-1), fontsize=8)
    ax.set_aspect(1)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    # ax = fig.add_subplot(2, 4, 4+j+1)
    ax = fig.add_subplot(4, 2, (j*2)+2)
    ax.plot(episodes[-1])
    ax.set_xlim((-10, 1033))
    ax.set_ylim((-1.1, 1.1))
    ax.set_aspect(400)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
# fig.subplots_adjust(wspace=0.1, hspace=0.05)
fig.subplots_adjust(wspace=0.05, hspace=0.2)
fig.show()
fig.savefig('im-batch-to-single.pdf', bbox_inches='tight')

plt.show()

