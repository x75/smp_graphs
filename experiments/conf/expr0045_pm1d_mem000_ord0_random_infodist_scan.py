"""smp_graphs configuration

baseline behaviour - open-loop uniform random search in finite isotropic space

id:thesis_smp_expr0050

Oswald Berthold 2017

special case of kinesis with coupling = 0 between measurement and action
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/10
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
    'd_i': 0.0,
}
    
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

desc = """The purpose of this experiment is to illustrate the effect
of projection distortion and external entropy on the information
distance between two sensory modalities. This is done using a
parameterized model of these effects and a few examples of different
configurations are shown in the plot."""

outputs = {
    'latex': {'type': 'latex',},
}

# configure system block 
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
                            # 's0': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            's0': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, 1)},
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

                # new modality m2 with distance parameter
                ('pre_l2', {
                    'block': ModelBlock2,
                    'params': {
                        'models': {
                            'infodistgen': {
                                'type': 'random_lookup',
                                'd': 0.5,
                                's_a': 0.5,
                                's_f': 1.0,
                                'e': 0.0,
                            }
                        },
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                        },
                        'outputs': {
                            'y': {'shape': (dim_s_proprio, 1)},
                        },
                    }
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
                            # 'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, 1)},
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
                
                # measure mutual information I(m1;m2)
                ('mi', {
                    'block': MIBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'shift': (0, 1),
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            # 'y': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                        },
                        'outputs': {
                            'infodist': {'shape': (1, 1, 1)},
                        }
                    }
                }),
                # measure information distance d(m1, m2)
                ('infodist', {
                    'block': InfoDistBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'shift': (0, 1),
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            # 'y': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                        },
                        'outputs': {
                            'infodist': {'shape': (1, 1, 1)},
                        }
                    }
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

    # plotting random_lookup influence
    # one configuration plot grid:
    # | transfer func h | horizontal output | horziontal histogram |
    # | vertical input  | information meas  | -                    |
    # | vertical histo  | -                 | - (here the model)   |
    ('plot_infodist', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.15,
            'hspace': 0.15,
            'xlim_share': True,
            'inputs': {
                's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'infodist': {
                    'bus': 'infodist/infodist',
                    'shape': (dim_s_proprio, 1, 1)
                },
            },
            'desc': 'Single infodist configuration',

            # subplot
            'subplots': [[
                {'input': ['pre_l2/h'], 'plot': timeseries},
                ]
            ],
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
            'xlim_share': True,
            'inputs': {
                's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'infodist': {
                    'bus': 'infodist/infodist',
                    'shape': (dim_s_proprio, 1, 1)
                },
            },
            'desc': 'Single episode pm1d baseline',
            
            'subplots': [
                # row 1: pre, s
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
                        'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        # 'mode': 'stack'
                    },
                ],
                
                # row 2: pre_l2, s
                [
                    {
                        'input': ['pre_l2', 's_p'],
                        'plot': [
                            partial(timeseries, alpha = 1.0, xlabel = None),
                            partial(timeseries, alpha = 0.5, xlabel = None),
                        ],
                        'title': 'proprio and f(proprio)',
                    },
                    {
                        'input': ['pre_l2', 's_p'],
                        'plot': [
                            partial(
                                histogram, orientation = 'horizontal', histtype = 'stepfilled',
                                yticks = False, xticks = False, alpha = 0.5, normed = False) for _ in range(2)],
                        'title': 'proprio and f(proprio) (histogram)',
                        'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                        # 'mode': 'stack'
                    },
                ],

                # row 3: budget
                [
                    {'input': 'credit_l1', 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
                         'title': 'agent budget (timeseries)',
                        'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    },
                    {'input': 'credit_l1', 'plot': partial(
                        histogram, orientation = 'horizontal', histtype = 'stepfilled',
                        yticks = False, ylim = (0, 1000), alpha = 1.0, normed = False),
                        'title': 'agent budget (histogram)',
                        'xlabel': 'count [n]',
                        'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                    },
                ],
                
                # [
                #     {
                #         'input': ['infodist'],
                #         'ndslice': (slice(None), 0, slice(None)),
                #         'shape': (dim_s_proprio, 1),
                #         'plot': [
                #             partial(timeseries, linewidth = 1.0, alpha = 1.0, marker = 'o', xlabel = None),
                #         ],
                #         'title': 'd(proprio, f(proprio))',
                #     },
                #     {
                #         'input': ['infodist'],
                #         'ndslice': (slice(None), 0, slice(None)),
                #         'shape': (dim_s_proprio, 1),
                #         'plot': [partial(
                #             histogram, orientation = 'horizontal', histtype = 'stepfilled',
                #             yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(1)],
                #         'title': 'd(proprio, f(proprio)) (histogram)',
                #         'desc': 'infodist \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
                #         # 'mode': 'stack'
                #     },
                # ],
            ],
        },
    })
])
