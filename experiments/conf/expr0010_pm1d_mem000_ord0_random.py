"""smp_graphs configuration

.. moduleauthor:: Oswald Berthold, 2018

id:thesis_smpx0001

Baseline behaviour - open-loop uniform random search in finite
isotropic space. Special case of kinesis with coupling = 0 between
measurement and action.

TODO
1. loop over randseed
2. loop over budget vs. density (space limits, distance threshold), hyperopt
3. loop over randseed with fixed optimized parameters
4. loop over kinesis variants [bin, cont] and system variants ord [0, 1, 2, 3?] and ndim = [1,2,3,4,8,16,...,ndim_max]

TODO low-level
- block groups
- experiment sig, make hash, store config and logfile with that hash
- compute experiment hash: if exists, use logfile, else compute
- compute experiment/model_i hash: if exists, use pickled model i, else train
- pimp smp_graphs graph visualisation
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MomentBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

from smp_graphs.utils_conf import get_systemblock

import functools
from functools import partial

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/5
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
}
    
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

desc = """This is a full length run of a configuration otherwise
identical to experiment expr000, consisting of an episode of {0} time
steps of the baseline behaviour. The length of the episode is greater
than the agent's initial budget. The strategy must statistically be
good enough to let the agent survive for a number of steps larger than
then the initial budget, which would be consumed after the same number
of steps when following a null strategy, that is, do nothing with a
zero or other single constant action.""".format(numsteps)

outputs = {
    'latex': {'type': 'latex',},
}

# configure system block 
systemblock   = get_systemblock['pm'](
    dim_s0 = dim, dim_s1 = dim, lag = 1, order = order)
# systemblock   = get_systemblock['sa'](
#     dim_s0 = dim, dim_s1 = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1 = systemblock['params']['dims']['s0']['dim']
# dim_s_goal   = dim_s1
dim_s_goal    = dim_s0

# graph
graph = OrderedDict([
    # robot
    ('robot1', systemblock),
        
    # brain
    ('braina', {
        # FIXME: idea: this guy needs to pass down its input/output configuration
        #        to save typing / errors on the individual modules
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'braina',
            'nocache': True,
            'graph': OrderedDict([
                # every brain has a budget
                ('budget', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'credit': np.ones((1, 1)) * budget,
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s0), # area of goal
                        'inputs': {
                            # 'credit': {'bus': 'budget/credit', 'shape': (1,1)},
                            's0': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            's0_ref': {'bus': 'pre_l1/pre', 'shape': (dim_s0, 1)},
                            },
                        'outputs': {
                            'credit': {'shape': (1, 1)},
                            'resets': {'shape': (dim_s0, 1)},
                        },
                        'models': {
                            'budget': {'type': 'budget_linear'},
                        },
                        'rate': 1,
                    },
                }),
                # uniformly dist. random goals, triggered when error < goalsize
                ('pre_l1', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'rate': 1,
                        # 'ros': ros,
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s0), # area of goal
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim, 'shape': (dim_s0, 1)},
                            'hi': {'val': lim, 'shape': (dim_s0, 1)},
                            'mdltr': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
                            'd_pre': {'shape': (dim_s0, 1)},
                        },
                        'models': {
                            'goal': {'type': 'random_uniform_modulated'}
                        },
                    },
                }),
                    
                # uniformly distributed random action, no modulation
                ('pre_l0', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim},
                            'hi': {'val': lim}},
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
                            }
                        },
                    }),
            ]),
        }
    }),
    
    # # robot
    # ('robot1', systemblock),

    # measures
    ('measure', {
        'block': MomentBlock2,
        'params': {
            'id': 'measure',
            'blocksize': numsteps,
            'inputs': {
                # 'credit': {'bus': 'pre_l1/credit', 'shape': (1, numsteps)},
                'bcredit': {'bus': 'budget/credit', 'shape': (1, numsteps)},
            },
            'outputs': {
                'bcredit_mu': {'shape': (1, 1)},
                'bcredit_var': {'shape': (1, 1)},
                'bcredit_min': {'shape': (1, 1)},
                'bcredit_max': {'shape': (1, 1)},
            },
        },
    }),
    
    # plotting
    ('plot', {
        'block': PlotBlock2,
        'params': {
            # general
            'id': 'plot',
            'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'savesize': (8, 6),
            'wspace': 0.15,
            'hspace': 0.5,
            'fig_rows': 2,
            'fig_cols': 3,
            'axesspec': [(0, slice(0, 2)), (0, 2), (1, slice(0, 2)), (1, 2)], 
            'desc': """Full episode of the baseline agent behaviour
            covering an episode length of {0} time
            steps. In the top left, the raw sensorimotor timeseries is
            shown, and in the top right the histograms of goal hits is
            plotted on top of the unique goal histogram, showing no
            obvious mismatch. The bottom row contains the same types
            of plots but for the budget variable, which never even
            gets close to a critical value.""".format(numsteps),
            
            # inputs
            'inputs': {
                's_p': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's_e': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'd_pre_l1': {'bus': 'pre_l1/d_pre', 'shape': (dim_s_goal, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'resets_l1': {'bus': 'budget/resets', 'shape': (1, numsteps), 'event': True},
            },

            # subplot configuration
            'subplots': [
                [
                    {
                        # 'input': ['pre_l0', 's_p', 'pre_l1'],
                        'input': ['pre_l1', 'pre_l0', 's_p', 'resets_l1',],
                        'plot': [
                            partial(timeseries, linewidth = 2.0,    alpha = 0.8, xlabel = None, marker = ''),
                            partial(timeseries, linewidth = 0.0,    alpha = 0.35, xlabel = None, marker = '.'),
                            partial(timeseries, linewidth = 0.0,    alpha = 0.35, xlabel = None, marker = '.', xticks = False),
                            partial(timeseries, linestyle = 'none', alpha = 0.5, xlabel = None, marker = 'o', xticks = False, color='r'),
                            # partial(linesegments),
                            # partial(timeseries, linewidth = 1.0, alpha = 1.0, xlabel = None),
                            # partial(timeseries, alpha = 1.0, xlabel = None),
                            # partial(timeseries, linewidth = 2.0, alpha = 1.0, xticks = False, xlabel = None)],
                        ],
                        'event': [False] * 3 +  [True],
                        'title': 'Goal, action, result, goal-hit',
                        'title_pos': 'top_out',
                        'ylabel': 'unit action [a]',
                        'legend_space': 0.75,
                        'legend_loc': 'right',
                        'legend': {
                            'goal $\hat{s}^{l_1}_p$': 0, 'action $\hat{s}^{l_0}_p$': dim_s0, 'measured $s_p$': 2 * dim_s0,
                            'goal hit': 3 * dim_s0},
                    },
                    
                    {
                        # 'input': ['d_pre_l1', 'pre_l0', 's_p', 'resets_l1'],
                        # 'event': [True, False, False, False],
                        'input': ['d_pre_l1', 'resets_l1'],
                        'event': [True, False],
                        'plot': [
                            partial(
                                histogram, histtype = 'stepfilled', bins=21,
                                orientation = 'horizontal',
                                yticks = False, # xticks = False,
                                alpha = 0.5, density = True
                            )
                        ] + [
                            partial(
                                histogram, histtype = 'stepfilled', bins=21,
                                orientation = 'horizontal',
                                yticks = False, # xticks = False,
                                alpha = 0.5, density = True, color='r'
                            )
                        ],
                        'title': None,
                        'title_pos': 'top_out',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        'xlim': None,
                        'xlabel': 'rel. frequency [n/N]',
                        # 'ylabel': 'unit budget [c]',
                        'legend': False,
                        # 'legend': {'unit budget': 1},
                        # 'legend_space': 0.75,
                        # 'mode': 'stack'
                    },
                ],
                
                [
                    {
                        'input': ['credit_l1'], # , 'credit_l1', 'credit_l1'],
                        # 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
                        'plot': [
                            partial(timeseries, alpha = 0.5, linewidth=0.8),
                            # partial(timeseries, alpha = 0.8, linestyle='none', marker='$\$$', color='orange'),
                            # partial(timeseries, alpha = 0.8, linestyle='none', marker='o', fillstyle='none', markersize=10, color='orange'),
                        ],
                        'title': 'Agent budget',
                        'title_pos': 'top_out',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        'xlabel': 'time step [t]',
                        'ylabel': 'unit budget [c]',
                        'legend': {'unit budget': 0},
                        'legend_loc': 'right',
                        'legend_space': 0.75,
                    },
                    
                    {
                        'input': ['credit_l1'],
                        'plot': [
                            partial(
                                histogram, histtype = 'stepfilled',
                                orientation = 'horizontal',
                                yticks = False, bins=21, # ylim = (0, 21),
                                alpha = 0.5, density = True,
                            ),
                        ],
                        'title': None,
                        'title_pos': 'top_out',
                        'xlabel': 'count [n]',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        'legend': False,
                        'legend_space': 0.75,
                    },
                ]
            ],
        },
    })
])
