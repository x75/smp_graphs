"""smp_graphs smp_expr0060

.. moduleauthor:: Oswald Berthold, 2017

the plan 20171127

adaptive internal models
0060
 - x keep pre_l2, configure for mild infodist
 - introduce pre_l2_pre_2_robot1_s0 map as batch learning internal model: sklearn model block
 - loop over few d's and models (save / load models)

0061
 - 0060 add online learning

self-exploration
0062
 - put transfer func back into system and recreate 0062
 - configure mild distortion and noise, time delay = 1
 - run 0062 and see it fail
 - explain fail, motivate prerequisites: delay by tapping; introspection by error statistics; adaptation to slow components by mu-coding or sfa; limits by learning progress; modulation, spawn, and kill by introspection

0064
 - add pre/meas pairs
 - add meas stack statistics or expand meas stack resp.
 - move meas stack into brain
 - spawn block / kill block
 - run expr and show how error statistics can drive model learning
 - learn the first model (finally)
"""

from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

from smp_base.plot import table

from smp_graphs.common import compose
from smp_graphs.block import FuncBlock2, TrigBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MeasBlock2, MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2

from numpy import sqrt, mean, square
from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin, f_meansquare, f_sum, f_rootmeansquare

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/10
numbins = 21
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

desc = """The purpose of this, and of the next few experiments is to
illustrate the effect of map distortions and external entropy on the
information distance between two corresponding variables, e.g. sensory
modalities. Here, an abstract parameterized model is introduced that
is able to produce these effects. The model consists of a transfer
function whose values over the interval [-1, 1] can be controlled by
three groups of parameters associated with unimodal gaussian
deformation, colored noise, and point-wise independent noise."""

#predicted variables
# p_vars = ['pre_l0/pre']
p_vars = ['robot1/s0']
# measured variables
# m_vars = ['robot1/s0']
m_vars = ['pre_l2/y']

# local conf dict for looping
lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': 1001, # sampling grid
        'l_a': 0.0,
        'd_a': 0.98,
        'd_s': 0.5,
        's_a': 0.02,
        's_f': 2.0,
        'e': 0.0,
    },
    'div_meas': 'chisq', # 'kld'
    'model_s2s_params': {
        'debug': True,
        'blocksize': numsteps,
        'models': {
            # from top config
            # 'pre_l2_2_robot1_s0': shln,
            'pre_l2_2_robot1_s0': {
                'type': 'sklearn',
                # 'skmodel': 'linear_model.LinearRegression',
                # 'skmodel_params': {'alpha': 1.0},
                # 'skmodel': 'kernel_ridge.KernelRidge',
                # 'skmodel_params': {'alpha': 0.1, 'kernel': 'rbf'},
                'skmodel': 'gaussian_process.GaussianProcessRegressor',
                'skmodel_params': {'kernel': ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)},
                # 'skmodel': 'gaussian_process.kernels.WhiteKernel, ExpSineSquared',
                # 'skmodel': model_selection.GridSearchCV
            },
        },
        'inputs': {
            # input
            'x_in': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
            # target
            'x_tg': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
        },
        'outputs': {
            'y': {'shape': (dim_s0, numsteps)},
            'h': {'shape': (dim_s0, numelem), 'trigger': 'trig/pre_l2_t1'},
        },
    }
}

div_meas = lconf['div_meas']
numelem = lconf['infodistgen']['numelem']

m_hist_bins       = np.linspace(-1.1, 1.1, numbins + 1)
m_hist_bincenters = m_hist_bins[:-1] + np.mean(np.abs(np.diff(m_hist_bins)))/2.0

# local variable shorthands
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

outputs = {
    'latex': {'type': 'latex',},
}

# configure system block 
systemblock   = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = 1, order = order)
# systemblock   = get_systemblock['sa'](
#     dim_s0 = dim, dim_s_extero = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s0 = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s0

infodistgen = lconf['infodistgen']

model_s2s_sklearn = {
    'block': ModelBlock2,
    'params': lconf['model_s2s_params'],
}

lconf['model_s2s'] = model_s2s_sklearn

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
    # triggers
    ('trig', {
        'block': TrigBlock2,
        'params': {
            'trig': np.array([numsteps]),
            'outputs': {
                'pre_l2_t1': {'shape': (1, 1)},
            }
        },
    }),

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
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s0), # area of goal
                        'inputs': {
                            # 'credit': {'bus': 'budget/credit', 'shape': (1,1)},
                            's0': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            # 's0': {'bus': m_vars[0], 'shape': (dim_s0, 1)},
                            's0_ref': {'bus': 'pre_l1/pre', 'shape': (dim_s0, 1)},
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

                # new artifical modality m2 with distortion parameters
                ('pre_l2', {
                    'block': ModelBlock2,
                    'params': {
                        'debug': False,
                        'models': {
                            # from top config
                            'infodistgen': infodistgen,
                        },
                        'inputs': {
                            'x': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                        },
                        'outputs': {
                            'y': {'shape': (dim_s0, 1)},
                            'h': {'shape': (dim_s0, lconf['infodistgen']['numelem']), 'trigger': 'trig/pre_l2_t1'},
                        },
                    }
                }),

                # inverse model s2s
                ('mdl1', lconf['model_s2s']),
                
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
                            # 'mdltr': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            'mdltr': {'bus': m_vars[0], 'shape': (dim_s0, 1)},
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
    
    # measures
    # m: mutual information I(m1;m2)
    ('m_mi', {
        'block': MIBlock2,
        'params': {
            'blocksize': numsteps,
            'shift': (0, 1),
            'inputs': {
                'x': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                # 'y': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                'y': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
            },
            'outputs': {
                'mi': {'shape': (1, 1, 1)},
            }
        }
    }),
    # m: information distance d(m1, m2) = 1 - (I(m1; m2)/H(m1,m2))
    ('m_di', {
        'block': InfoDistBlock2,
        'params': {
            'blocksize': numsteps,
            'shift': (0, 1),
            'inputs': {
                'x': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                # 'y': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                'y': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
            },
            'outputs': {
                'infodist': {'shape': (1, 1, 1)},
            }
        }
    }),

    # m: budget moments
    ('m_budget', {
        'block': MomentBlock2,
        'params': {
            'id': 'm_budget',
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': {
                # 'credit': {'bus': 'pre_l1/credit', 'shape': (1, numsteps)},
                'y': {'bus': 'budget/credit', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y_mu': {'shape': (1, 1)},
                'y_var': {'shape': (1, 1)},
                'y_min': {'shape': (1, 1)},
                'y_max': {'shape': (1, 1)},
            },
        },
    }),

    # m: error
    ('m_err', {
        'block': MeasBlock2,
        'params': {
            'id': 'm_err',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'basic',
            'scope': 'local',
            'meas': 'sub',
            'inputs': {
                'x1': {'bus': p_vars[0], 'shape': (1, numsteps)},
                'x2': {'bus': m_vars[0], 'shape': (1, numsteps)},
            },
            'outputs': {
                'y': {'shape': (1, numsteps)},
            },
        },
    }),
    
    # m: error
    ('m_err_mdl1', {
        'block': MeasBlock2,
        'params': {
            'id': 'm_err',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'basic',
            'scope': 'local',
            'meas': 'sub',
            'inputs': {
                'x1': {'bus': p_vars[0], 'shape': (1, numsteps)},
                'x2': {'bus': 'mdl1/y', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y': {'shape': (1, numsteps)},
            },
        },
    }),
    
    # m: (root) mean squared error
    ('m_rmse', {
        'block': FuncBlock2,
        'params': {
            # 'id': 'm_rmse',
            'blocksize': numsteps,
            'debug': False,
            'func': f_rootmeansquare,
            'inputs': {
                'x': {'bus': 'm_err/y', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y': {'shape': (1, 1)},
            },
        },
    }),

    # testing function composition
    # # m: (root) mean squared error
    # ('m_rmse', {
    #     'block': FuncBlock2,
    #     'params': {
    #         # 'id': 'm_rmse',
    #         'blocksize': numsteps,
    #         'debug': False,
    #         'func': compose(sqrt, mean, square),
    #         'inputs': {
    #             'x': {'bus': 'm_err/y', 'shape': (1, numsteps)},
    #         },
    #         'outputs': {
    #             'y': {'shape': (1, 1)},
    #         },
    #     },
    # }),
    
    # m: histogram
    ('m_hist', {
        'block': MeasBlock2,
        'params': {
            'id': 'm_hist',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'hist',
            'scope': 'local',
            'meas': 'hist',
            # direct histo input?
            # or signal input
            'inputs': {
                'x1': {'bus': p_vars[0], 'shape': (1, numsteps)},
                'x2': {'bus': m_vars[0], 'shape': (1, numsteps)},
            },
            'bins': m_hist_bins,
            'outputs': {
                'h_x1': {'shape': (1, numbins)},
                'h_x2': {'shape': (1, numbins)},
            },
        },
    }),
    
    # m: divergence histos
    ('m_div', {
        'block': MeasBlock2,
        'params': {
            'id': 'm_div',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'basic',
            'scope': 'local',
            'meas': div_meas, # ['chisq', 'kld'],
            # direct histo input?
            # or signal input
            'inputs': {
                'x1': {'bus': 'm_hist/h_x1', 'shape': (1, numbins)},
                'x2': {'bus': 'm_hist/h_x2', 'shape': (1, numbins)},
            },
            'outputs': {
                'y': {'shape': (1, numbins)},
            },
        },
    }),
    
    # m: sum divergence
    ('m_sum_div', {
        'block': FuncBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'func': f_sum,
            'inputs': {
                'x': {'bus': 'm_div/y', 'shape': (1, numbins)},
            },
            'outputs': {
                'y': {'shape': (1, 1)},
            },
        },
    }),
    
    # plotting random_lookup influence
    # one configuration plot grid:
    # | transfer func h | horizontal output | horziontal histogram |
    # | vertical input  | information meas  | -                    |
    # | vertical histo  | -                 | - (here the model)   |
    ('plot', {
        'block': PlotBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.15,
            'hspace': 0.1,
            'xlim_share': True,
            'ylim_share': True,
            'inputs': {
                's_p': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l2': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
                'pre_l2_h': {'bus': 'pre_l2/h', 'shape': (dim_s0, numelem)},
                'mdl1_y': {'bus': 'mdl1/y', 'shape': (dim_s0, numsteps)},
                'mdl1_h': {'bus': 'mdl1/h', 'shape': (dim_s0, numelem)},
                'm_err_mdl1': {'bus': 'm_err_mdl1/y', 'shape': (1, numsteps)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'budget_mu': {'bus': 'm_budget/y_mu', 'shape': (1, 1)},
                'budget_var': {'bus': 'm_budget/y_var', 'shape': (1, 1)},
                'budget_min': {'bus': 'm_budget/y_min', 'shape': (1, 1)},
                'budget_max': {'bus': 'm_budget/y_max', 'shape': (1, 1)},
                'm_di': {
                    'bus': 'm_di/infodist',
                    'shape': (dim_s0, 1, 1)
                },
                'm_mi': {
                    'bus': 'm_mi/mi',
                    'shape': (dim_s0, 1, 1)
                },
                'm_err': {'bus': 'm_err/y', 'shape': (1, numsteps)},
                'm_rmse': {'bus': 'm_rmse/y', 'shape': (1, 1)},
                'm_div': {'bus': 'm_div/y', 'shape': (1, numbins)},
                'm_sum_div': {'bus': 'm_sum_div/y', 'shape': (1, 1)},
            },
            'desc': 'Single infodist configuration',

            # subplot
            'subplots': [
                # row 1: transfer func, out y time, out y histo
                [
                    {
                        'input': ['pre_l2_h'], 'plot': timeseries,
                        'title': 'transfer function $h$', 'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['pre_l2'], 'plot': timeseries,
                        'title': 'timeseries $y$', 'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'xlim': None, 'xticks': False, 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['mdl1_h'], 'plot': timeseries,
                        'title': 'transfer function $h$', 'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['pre_l2'], 'plot': histogram,
                        'title': 'histogram $y$', 'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, 'xticklabels': None,
                        'xlabel': 'count $c$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                ],
                
                # row 2: in x time, error x - y time, none
                [
                    {
                        'input': ['s_p'], 'plot': timeseries,
                        'title': 'timeseries $x$',
                        'aspect': 2.2/numsteps,
                        'orientation': 'vertical',
                        'xlim': None, 'xticks': False, # 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'yticks': False,
                        'ylim': (-1.1, 1.1),
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['m_err', 'm_err_mdl1'], 'plot': [timeseries, partial(timeseries, alpha = 0.7)],
                        'title': 'error $x - y$',
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xlabel': 'time step $k$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': (-1.1, 1.1), # None,
                        # 'legend_loc': 'right',
                    },
                    {
                        'input': ['mdl1_y'], 'plot': timeseries,
                        'title': 'timeseries $x$',
                        'aspect': 2.2/numsteps,
                        'orientation': 'vertical',
                        'xlim': None, 'xticks': False, # 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'yticks': False,
                        'ylim': (-1.1, 1.1),
                        'legend_loc': 'left',
                    },
                    {},
                ],
                
                # row 3: in x histo, measures global, divergence h1, h2
                [
                    {
                        'input': ['s_p'], 'plot': histogram,
                        'title': 'histogram $x$', 'aspect': 'shared', # (1*numsteps)/(2*2.2),
                        'orientation': 'vertical',
                        'xlim': (-1.1, 1.1), 'xinvert': False, # 'xticks': False, 'xticklabels': None, #
                        'xlabel': 'input $x \in [-1, ..., 1]$', 
                        'ylim': None, 'yinvert': True,  # 'yticks': None, 'yticklabels': None,
                        'ylabel': 'count $c$',
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['budget_%s' % (outk,) for outk in ['mu', 'var', 'min', 'max']] + ['m_mi', 'm_di', 'm_rmse', 'm_sum_div'],
                        'shape': [(1, 1) for outk in ['mu', 'var', 'min', 'max', 'm_mi', 'm_di', 'm_rmse', 'm_sum_div']],
                        'mode': 'stack',
                        'title': 'measures', 'title_pos': 'bottom',
                        'plot': table,
                    },
                    {},
                    {
                        'input': ['m_div'], 'plot': partial(timeseries, linestyle = 'none', marker = 'o'),
                        'title': 'histogram divergence %s $h1 - h2$' % (div_meas, ),
                        'shape': (1, numbins),
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xaxis': m_hist_bincenters, 'xlabel': 'bins $k$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': None, 'ylabel': 'divergence'
                        # 'legend_loc': 'right',
                    },
                ],
                
            ],
        },
    }),

    # # plotting
    # ('plot', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'id': 'plot',
    #         'blocksize': numsteps,
    #         'saveplot': saveplot,
    #         'savetype': 'pdf',
    #         'wspace': 0.15,
    #         'hspace': 0.15,
    #         'xlim_share': True,
    #         'inputs': {
    #             's_p': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
    #             's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
    #             'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l2': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
    #             'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
    #             'm_di': {
    #                 'bus': 'm_di/infodist',
    #                 'shape': (dim_s0, 1, 1)
    #             },
    #         },
    #         'desc': 'Single episode pm1d baseline',
            
    #         'subplots': [
    #             # row 1: pre, s
    #             [
    #                 {
    #                     'input': ['pre_l0', 's_p', 'pre_l1'],
    #                     'plot': [
    #                         partial(timeseries, linewidth = 1.0, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, linewidth = 2.0, alpha = 1.0, xticks = False, xlabel = None)],
    #                     'title': 'two-level prediction and measurement (timeseries)',
    #                 },
    #                 {
    #                     'input': ['pre_l0', 's_p', 'pre_l1'],
    #                     'plot': [partial(
    #                         histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                         yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(3)],
    #                     'title': 'two-level prediction and measurement (histogram)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                     # 'mode': 'stack'
    #                 },
    #             ],
                
    #             # row 2: pre_l2, s
    #             [
    #                 {
    #                     'input': ['pre_l2', 's_p'],
    #                     'plot': [
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 0.5, xlabel = None),
    #                     ],
    #                     'title': 'proprio and f(proprio)',
    #                 },
    #                 {
    #                     'input': ['pre_l2', 's_p'],
    #                     'plot': [
    #                         partial(
    #                             histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                             yticks = False, xticks = False, alpha = 0.5, normed = False) for _ in range(2)],
    #                     'title': 'proprio and f(proprio) (histogram)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                     # 'mode': 'stack'
    #                 },
    #             ],

    #             # row 3: budget
    #             [
    #                 {'input': 'credit_l1', 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
    #                      'title': 'agent budget (timeseries)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                 },
    #                 {'input': 'credit_l1', 'plot': partial(
    #                     histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                     yticks = False, ylim = (0, 1000), alpha = 1.0, normed = False),
    #                     'title': 'agent budget (histogram)',
    #                     'xlabel': 'count [n]',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                 },
    #             ],
                
    #             # [
    #             #     {
    #             #         'input': ['m_di'],
    #             #         'ndslice': (slice(None), 0, slice(None)),
    #             #         'shape': (dim_s0, 1),
    #             #         'plot': [
    #             #             partial(timeseries, linewidth = 1.0, alpha = 1.0, marker = 'o', xlabel = None),
    #             #         ],
    #             #         'title': 'd(proprio, f(proprio))',
    #             #     },
    #             #     {
    #             #         'input': ['m_di'],
    #             #         'ndslice': (slice(None), 0, slice(None)),
    #             #         'shape': (dim_s0, 1),
    #             #         'plot': [partial(
    #             #             histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #             #             yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(1)],
    #             #         'title': 'd(proprio, f(proprio)) (histogram)',
    #             #         'desc': 'infodist \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #             #         # 'mode': 'stack'
    #             #     },
    #             # ],
    #         ],
    #     },
    # })
])
