"""smp_graphs configuration

baseline behaviour - open-loop uniform random search in finite isotropic space

id:thesis_smpx0001

Oswald Berthold 2017

special case of kinesis with coupling = 0 between measurement and action
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MomentBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/100
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
    'outputs': {
        'latex': {'type': 'text'},
    },
}

dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

outputs = {
    'latex': {'type': 'text'},
}

# prepare system block
systemblock   = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = 1, order = order)
# systemblock   = get_systemblock['sa'](
#     dim_s_proprio = dim, dim_s_extero = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio

# print "sysblock", systemblock['params']['dim_s_proprio']

# TODO
# 1. loop over randseed
# 2. loop over budget vs. density (space limits, distance threshold), hyperopt
# 3. loop over randseed with fixed optimized parameters
# 4. loop over kinesis variants [bin, cont] and system variants ord [0, 1, 2, 3?] and ndim = [1,2,3,4,8,16,...,ndim_max]

# TODO low-level
# block groups
# experiment sig, make hash, store config and logfile with that hash
# compute experiment hash: if exists, use logfile, else compute
# compute experiment/model_i hash: if exists, use pickled model i, else train
# pimp smp_graphs graph visualisation

# graph
graph = OrderedDict([
    # robot
    ('robot1', systemblock),
        
    # brain
    ('brain', {
        # FIXME: idea: this guy needs to pass down its input/output configuration
        #        to save typing / errors on the individual modules
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
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {
                            # 'credit': {'bus': 'budget/credit', 'shape': (1,1)},
                            's0': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            's0_ref': {'bus': 'pre_l1/pre', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'credit': {'shape': (1,1)},
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
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': lim, 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
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
                            'pre': {'shape': (dim_s_proprio, 1)},
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
            'id': 'plot',
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.15,
            'hspace': 0.15,
            'inputs': {
                's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                },
            'desc': 'Baseline agent illustration',
            'subplots': [
                [
                    {
                        'input': ['pre_l0', 's_p', 'pre_l1'],
                        'plot': [
                            partial(timeseries, linewidth = 1.0, alpha = 1.0, xlabel = None),
                            partial(timeseries, alpha = 1.0, xlabel = None),
                            partial(timeseries, linewidth = 2.0, alpha = 1.0, xticks = False, xlabel = None)],
                        'title': 'two-level prediction and measurement (timeseries)',
                    },
                    {
                        'input': ['pre_l0', 's_p', 'pre_l1'],
                        'plot': [partial(
                            histogram, orientation = 'horizontal', histtype = 'stepfilled',
                            yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(3)],
                        'title': 'two-level prediction and measurement (histogram)',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        # 'mode': 'stack'
                    },
                ],
                [
                    {'input': 'credit_l1', 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
                         'title': 'agent budget (timeseries)',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    },
                    {'input': 'credit_l1', 'plot': partial(
                        histogram, orientation = 'horizontal', histtype = 'stepfilled',
                        yticks = False, ylim = (0, 1000), alpha = 1.0, normed = False),
                        'title': 'agent budget (histogram)',
                        'xlabel': 'count [n]',
                        'desc': 'Single episode pm1d baseline \\autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    },
                ]
            ],
        },
    })
])
