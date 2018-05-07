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

# global parameters can be overwritten from the commandline
ros = False
numsteps = 30 # 10000/400
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 127

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    # 'outputs': {
    #     'latex': {'type': 'latex'},
    # },
}

dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

# print "expr0000 __doc__", __doc__

desc = """The baseline of agent behaviour is open-loop uniform random
search in finite isotropic space. The minimal function required for
this behaviour is \\emph{{goal recognition}}. This experiment consists
of an episode of short duration with only 100 time steps and is
intended to introduce the style of documentation of experiments in
this thesis. Each row in the plot of
\\autoref{{fig:smp-expr0000-pm1d-mem000-ord0-random-demo-plot}}
corresponds to one selection of variable, with the data plotted as
timeseries on the left column and as a histogram on the right
column. The name of the experiment says, that this is an
smp-experiment with the number 0 (0000), a single pointmass system of
zeroth order with one DoF and zero memory, and a (uniform) random
brain.""".format()

# outputs = lconf['outputs']
outputs = {
    'latex': {'type': 'latex',},
}

# prepare system block
systemblock   = get_systemblock['pm'](
    dim_s0 = dim, dim_s1 = dim, lag = 1, order = order)
# systemblock   = get_systemblock['sa'](
#     dim_s0 = dim, dim_s1 = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1  = systemblock['params']['dims']['s0']['dim']
# dim_s_goal   = dim_s1
dim_s_goal    = dim_s0

# graph
graph = OrderedDict([
    # robot
    ('robot1', systemblock),
        
    # brain
    ('brain', {
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'brain',
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
                            'credit': {'shape': (1,1)},
                            'resets': {'shape': (dim_s0,1)},
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
                        # goal recognition
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s0), # area of goal
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim, 'shape': (dim_s0, 1)},
                            'hi': {'val': lim, 'shape': (dim_s0, 1)},
                            'mdltr': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
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

    # # filter state by d(goal) < goalsize
    # ('filter_s_p', {
    #     'block': FilterBlock2,
    #     'params': {
    #         'inputs': {
    #             's0': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
    #         }
    #     }
    # }),

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
            'id': 'plot',
            'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'savesize': (8, 5),
            'wspace': 0.3,
            'hspace': 0.5,
            'inputs': {
                's_p': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's_e': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'resets_l1': {'bus': 'budget/resets', 'shape': (1, numsteps), 'event': True},
                },
            'desc': 'Baseline agent illustration',
            'subplots': [
                
                [
                    {
                        'input': ['pre_l1', 'pre_l0', 's_p', 'resets_l1', 'pre_l1'],
                        'plot': [
                            partial(timeseries, linewidth = 2.0,    alpha = 0.6, xlabel = None, marker = ''),
                            partial(timeseries, linewidth = 1.0,    alpha = 0.7, xlabel = None, marker = '.'),
                            partial(timeseries, linewidth = 1.0,    alpha = 0.7, xlabel = None, marker = '.', xticks = False),
                            partial(timeseries, linestyle = 'none', alpha = 0.8, xlabel = None, marker = 'o', xticks = False, color='r'),
                            partial(linesegments),
                        ],
                        'lineseg_val': [None] * 4 + [('pre_l0', 's_p')],
                        'lineseg_idx': [None] * 4 + [[(18, 20), (19, 21), (20, 22)]],
                        'event': [False] * 3 +  [True] + [False],
                        'title': 'Goal, action, result, goal-hit',
                        'title_pos': 'top_out',
                        'ylabel': 'Acceleration [a]',
                        'legend_space': 0.75,
                    },
                    
                    # {
                    #     'input': ['pre_l1', 'pre_l0', 's_p'],
                    #     'plot': [partial(
                    #         histogram, orientation = 'horizontal', histtype = 'stepfilled',
                    #         yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(3)],
                    #     'title': '',
                    #     'title_pos': 'top_out',
                    #     'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    #     'legend_space': 0.75,
                    #     # 'mode': 'stack'
                    # },
                    
                ],
                [
                    {
                        'input': ['credit_l1', 'credit_l1'],
                        # 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
                        'plot': [
                            partial(timeseries, alpha = 0.8, linestyle='none', marker='$\$$', color='orange'),
                            partial(timeseries, alpha = 0.8, linestyle='none', marker='o', fillstyle='none', markersize=10, color='orange'),
                        ],
                        'title': 'Agent budget',
                        'title_pos': 'top_out',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        'xlabel': 'time step [t]',
                        'ylabel': 'unit budget [c]',
                        'legend': {'unit budget': 0},
                        'legend_space': 0.75,
                    },
                    
                    # {
                    #     'input': 'credit_l1', 'plot': partial(
                    #         histogram, orientation = 'horizontal', histtype = 'stepfilled',
                    #         yticks = False, ylim = (0, 1000), alpha = 1.0, normed = False),
                    #     'title': 'agent budget (histogram)',
                    #     'title_pos': 'top_out',
                    #     'xlabel': 'count [n]',
                    #     'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    #     'legend_space': 0.75,
                    # },
                    
                ]
            ],
        },
    })
])
