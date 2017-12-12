"""smp_graphs smp_expr0062

.. moduleauthor:: Oswald Berthold, 2017

the plan 20171127, see :mod:`expr0060_pm1d_mem000_ord0_model_s2s`

ea: embodied agent

adaptive internal models
0060
 - x keep pre_l2, configure for mild infodist
 - x introduce pre_l2_pre_2_robot1_s0 map as batch learning internal model: sklearn model block
 - x loop over few d's and models (save / load models)

0061
 - x 0060 add online learning

self-exploration
0062
 - x put transfer func back into system and recreate 0062
 - x configure mild distortion and noise, time delay = 1, 
 - x run 0062 and see it fail
 - x explain fail: time

0063
 - 0062 fixed with delay

0064
 - add pre/meas pairs
 - add meas stack statistics or expand meas stack resp.
 - move meas stack into brain
 - spawn block / kill block
 - run expr and show how error statistics can drive model learning
 - learn the first model (finally)

 - motivate prerequisites: delay by tapping; introspection by error
   statistics; adaptation to slow components by mu-coding or sfa;
   limits by learning progress and error stats; modulation, spawn, and
   kill by introspection
 - motivation, sampling, limits, energy, ...
"""

import re 
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

from smp_base.plot import table, bar

from smp_graphs.common import compose
from smp_graphs.block import FuncBlock2, TrigBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MeasBlock2, MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2

from numpy import sqrt, mean, square
from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin, f_meansquare, f_sum, f_rootmeansquare, f_envelope

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

# predicted variables
p_vars = ['pre_l0/pre']
p_del_vars = p_vars
# p_del_vars = ['delay/dy']
# p_vars = ['robot1/s0']
# measured variables
m_vars = ['robot1/s0']
# m_vars = ['pre_l2/y']

desc = """In this experiment an \\emph{{embodied agent}} is reobtained
by putting the distortion model from the previous experiments back
into the agent's body and environment. This establishes the first
complete embodied agent which will be used and incrementally extended
over the remaining sections of this chapter. The experiment appears to
be identical to \\ref{{{0}}} but on close inspection, the computation
graph is slightly different.

An example interpretation for this model is that of proprioceptive
space. Proprioception means self-perception and refers to a sense of
body configuration, conveyed via joint angle measurements. In this
thesis, proprioception is used generally to refer to an agent's
low-level motor space represented by the predictions, measurements,
and other statistics of the corresponding variables. Other examples
are an outgoing motor voltage (prediction) and measured rotation rate
on a wheeled robot, or the predicted motor current and the measured
torque on a joint on a torque-controlled robot.

If a robots actually is constructed in such a way, that there exists a
sensor that measures something \\emph{{physically}} close to the action
itself, it can be assumed that there will be a residual caused by
microscopic but systematic divergence between actions and their
corresponding measurements. The experiment shows how the simple agent
compensates these deviations with adaptive inverse predictions.
""".format('sec:smp-expr0045-pm1d-mem000-ord0-random-infodist-id')

# configuration as table
desc += """
\\begin{{tabularx}}{{\\textwidth}}{{rrrr}}
\\textbf{{Numsteps}} & \\textbf{{measurement vars}} & \\textbf{{prediction vars}} & \\textbf{{crossmodal prediction m2p}} \\\\
{0} & {1} & {2} & {3} \\\\
\\end{{tabularx}}""".format(
    numsteps, re.sub(r'_', r'\\_', str(m_vars)), re.sub(r'_', r'\\_', str(p_vars)), ['mdl1/y'])



dim_s0 = 1
numelem = 1001

# local conf dict for looping
lconf = {
    # environment / system
    'sys': {
        # global
        # 'debug': True,
        'budget': 1000/1,
        'dim': dim_s0,
        'dims': {
            # expr0062: setting proprio lag to zero (<< environment minlag 1 resp.) models
            #           fast in-body transmission and feedback
            'm0': {'dim': dim_s0, 'dist': 0, 'lag': 0}, # , 'mins': [-1] * dim_s0, 'maxs': [1] * dim_s0
            's0': {'dim': dim_s0, 'dist': 0, 'dissipation': 1.0},
        },
        'dim_s0': dim_s0,
        'dim_s1': dim_s0,
        # time delay
        'dt': 0.1,
        'order': 0,
        # 'lag': 3,
        # distortion
        'transfer': 3,
        # distortion and memory
        'coupling_sigma': 0, # 1e-3,
        # external entropy
        'anoise_mean': 0.0,
        'anoise_std': 0.0, # 1e-2,
        'sysnoise': 0.0, # 1e-2,
        'lim': 1.0,
        # ground truth cheating
        'h_numelem': numelem, # sampling grid
    },
    # agent / models
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': numelem, # sampling grid
        'l_a': 0.0,
        'd_a': 0.98,
        'd_s': 0.5,
        's_a': 0.02,
        's_f': 2.0,
        'e': 0.0,
    },
    'div_meas': 'pyemd', # 'chisq', # 'kld'
    'model_s2s_params': {
        'debug': True,
        'blocksize': numsteps,
        'models': {
            # from top config
            # 'pre_l2_2_robot1_s0': shln,
            'pre_l2_2_robot1_s0': {
                'type': 'sklearn',
                'load': False,
                # 'skmodel': 'linear_model.Ridge',
                # 'skmodel_params': {'alpha': 1.0},
                'skmodel': 'kernel_ridge.KernelRidge',
                'skmodel_params': {'alpha': 0.1, 'kernel': reduce(lambda x, y: x + y, [ExpSineSquared(np.random.exponential(0.3), 5.0, periodicity_bounds=(1e-2, 1e1)) for _ in range(10)])}, # 'rbf'},
                # 'skmodel': 'gaussian_process.GaussianProcessRegressor',
                # 'skmodel_params': {'kernel': ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)},
                # 'skmodel': 'gaussian_process.kernels.WhiteKernel, ExpSineSquared',
                # 'skmodel': model_selection.GridSearchCV
            },
        },
        'inputs': {
            # input
            'x_in': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
            # target
            # 'x_tg': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
            'x_tg': {'bus': p_del_vars[0], 'shape': (dim_s0, numsteps)},
        },
        'outputs': {
            'y': {'shape': (dim_s0, numsteps)},
            'h': {'shape': (dim_s0, numelem), 'trigger': 'trig/t1'},
        },
    }
}

div_meas = lconf['div_meas']
numelem = lconf['infodistgen']['numelem']

m_hist_bins       = np.linspace(-1.1, 1.1, numbins + 1)
m_hist_bincenters = m_hist_bins[:-1] + np.mean(np.abs(np.diff(m_hist_bins)))/2.0

# local variable shorthands
dim = lconf['sys']['dim']
budget = lconf['sys']['budget'] # 510
order = lconf['sys']['order']
lim = lconf['sys']['lim'] # 1.0
outputs = {
    'latex': {'type': 'latex',},
}

# configure system block
# for 0062 this start to become more involved because the system's
# response needs to be parameterized with
# - amount of contractive / expansive map distortion (info distance d)
# - smoothness of map with respect to input changes (inverse frequency s, aka beta in 1/f noise)
# - external entropy / independence (e/i)

# pointmass
systemblock   = get_systemblock['pm'](**lconf['sys'])
#     dim_s_proprio = dim, dim_s_extero = dim, lag = 1, order = order)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1  = dim_s0 # systemblock['params']['dims']['s1']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s0

# simplearm
# systemblock   = get_systemblock['sa'](
#     dim_s0 = dim, dim_s_extero = dim, lag = 1)

# agent models
infodistgen = lconf['infodistgen']

model_s2s_sklearn = {
    'block': ModelBlock2,
    'params': lconf['model_s2s_params'],
}

lconf['model_s2s'] = model_s2s_sklearn

# graph
graph = OrderedDict([
    # triggers
    ('trig', {
        'block': TrigBlock2,
        'params': {
            'trig': np.array([numsteps]),
            'outputs': {
                # 'pre_l2_t1': {'shape': (1, 1)},
                't1': {'shape': (1, 1)},
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
                        # 'debug': True,
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim},
                            'hi': {'val': lim}},
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
                        }
                    },
                }),

                # # delay blocks for dealing with sensorimotor delays
                # ('delay', {
                #     'block': DelayBlock2,
                #     'params': {
                #         # 'debug': True,
                #         'blocksize': 1,
                #         # 'inputs': {'y': {'bus': 'motordiff/dy'}},
                #         'inputs': {
                #             'y':     {'bus': 'pre_l0/pre', 'shape': (dim_s0, 1)},
                #             'mdl_y': {'bus': 'mdl1/y',     'shape': (dim_s0, numsteps)}},
                #         'delays': {'y': 0, 'mdl_y': 0},
                #     }
                # }),
        
                # inverse model s2s
                ('mdl1', lconf['model_s2s']),
                
            ]),
        }
    }),

    # measures
    ('measures', {
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'measures',
            'nocache': True,
            'graph': OrderedDict([
                # m: mutual information I(m1;m2)
                ('m_mi', {
                    'block': MIBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'shift': (0, 1),
                        'inputs': {
                            'x': {'bus': p_del_vars[0], 'shape': (dim_s0, numsteps)},
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
                            'x': {'bus': p_del_vars[0], 'shape': (dim_s0, numsteps)},
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
                        'blocksize': numsteps,
                        # 'debug': True,
                        'mode': 'basic',
                        'scope': 'local',
                        'meas': 'sub',
                        'inputs': {
                            'x1': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
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
                        'blocksize': numsteps,
                        # 'debug': True,
                        'mode': 'basic',
                        'scope': 'local',
                        'meas': 'sub',
                        'inputs': {
                            # 'x1': {'bus': p_vars[0], 'shape': (1, numsteps)},
                            # 'x1': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                            'x1': {'bus': 'mdl1/y', 'shape': (1, numsteps)},
                            # 'x1': {'bus': 'delay/dmdl_y',  'shape': (1, numsteps)},
                            'x2': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                        },
                        'outputs': {
                            'y': {'shape': (1, numsteps)},
                        },
                    },
                }),
    
                # m: (root) mean squared error
                ('m_err_mdl1_amp', {
                    'block': FuncBlock2,
                    'params': {
                        # 'id': 'm_rmse',
                        'blocksize': numsteps,
                        'debug': False,
                        'func': f_envelope,
                        'inputs': {
                            'x': {'bus': 'm_err_mdl1/y', 'shape': (1, numsteps)},
                            'c': {'val': 0.01, 'shape': (1, 1)},
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
                            'x1_p': {'shape': (1, numbins)},
                            'x1_x': {'shape': (1, numbins + 1)},
                            'x2_p': {'shape': (1, numbins)},
                            'x2_x': {'shape': (1, numbins + 1)},
                        },
                    },
                }),

                # m: divergence histos
                ('m_div', {
                    'block': MeasBlock2,
                    'params': {
                        'id': 'm_div',
                        'blocksize': numsteps,
                        'debug': True,
                        'mode': 'div', # 'basic',
                        'scope': 'local',
                        'meas': div_meas, # ['chisq', 'kld'],
                        # direct histo input?
                        # or signal input
                        'inputs': {
                            'x1_p': {'bus': 'm_hist/x1_p', 'shape': (1, numbins)},
                            'x1_x': {'bus': 'm_hist/x1_x', 'shape': (1, numbins + 1)},
                            'x2_p': {'bus': 'm_hist/x2_p', 'shape': (1, numbins)},
                            'x2_x': {'bus': 'm_hist/x2_x', 'shape': (1, numbins + 1)},
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
            ]),
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
            'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.15,
            'hspace': 0.1,
            'xlim_share': True,
            'ylim_share': True,
            'inputs': {
                's0': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)}, # 
                's1': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)},
                'sys_h': {'bus': 'robot1/h', 'shape': (dim_s0, numelem)},
                'pre_l0': {'bus': p_vars[0], 'shape': (dim_s_goal, numsteps)}, # 'pre_l0/pre'
                'pre_l0_del': {'bus': 'delay/dy', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                # 'pre_l2': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
                # 'pre_l2_h': {'bus': 'pre_l2/h', 'shape': (dim_s0, numelem)},
                'mdl1_y': {'bus': 'mdl1/y', 'shape': (dim_s0, numsteps)},
                'mdl1_h': {'bus': 'mdl1/h', 'shape': (dim_s0, numelem)},
                'mdl1_y_del': {'bus': 'delay/dmdl_y', 'shape': (dim_s0, numsteps)},
                'm_err_mdl1': {'bus': 'm_err_mdl1/y', 'shape': (1, numsteps)},
                'm_err_mdl1_amp': {'bus': 'm_err_mdl1_amp/y', 'shape': (1, numsteps)},
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
            'desc': 'result. In the lower left part of the figure the prediction histogram and timeseries are shown. In the experiment, the signal is transferred by the transfer function pre\_l2/$h$ and is transformed into the signal $y$ whose histogram diverges by the amount shown in the lower right corner divergence plot. In the error timeseries panel, the original prediction error and the error after the adaptive model\'s transformation are shown on top of each other, together with a magnitude estimate of the second error shown as a red line.',

            # subplot
            'subplots': [
                # row 1: transfer func, out y time, out y histo
                [
                    {
                        'input': ['sys_h'], 'plot': partial(timeseries), # color = 'k'
                        'title': 'transfer function $h$', 'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend_loc': 'right',
                    },
                    {
                        # 'input': ['s0', 'pre_l0', 'pre_l0_del'], 'plot': [timeseries for _ in range(3)],
                        'input': ['pre_l0', 's0', ], 'plot': [partial(timeseries, alpha = 0.3), timeseries],
                        # 'input': ['s0'], 'plot': [timeseries for _ in range(3)],
                        'title': 'timeseries $y$', 'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'xlim': None, 'xticks': False, 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['mdl1_h'], 'plot': partial(timeseries), # , color = 'k'),
                        'title': 'transfer function $h$', 'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['s0'], 'plot': histogram,
                        'cmap_off': [1, 1],
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
                        'input': ['pre_l0'], 'plot': timeseries,
                        # 'input': ['pre_l0', 'pre_l0_del'], 'plot': timeseries,
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
                        'input': ['m_err', 'm_err_mdl1', 'm_err_mdl1_amp'], 'plot': [timeseries, partial(timeseries, alpha = 0.7), partial(timeseries, color = 'r')],
                        'cmap_off': [1, 1, 1, 1],
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
                        'input': ['pre_l0', 'mdl1_y'], 'plot': timeseries,
                        'cmap_off': [0, 1, 1],
                        # 'cmap_idx': [0, 30, 60],
                        # 'input': ['pre_l0_del', 'mdl1_y', 'mdl1_y_del'], 'plot': timeseries,
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
                        'input': ['pre_l0'], 'plot': histogram,
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
                        'input': ['m_div'], 'plot': bar,
                        # 'input': ['m_div'], 'plot': partial(timeseries, linestyle = 'none', marker = 'o'),
                        'title': 'histogram divergence %s $h1 - h2$' % (div_meas, ),
                        'shape': (1, numbins),
                        # 'aspect': 'auto',
                        'orientation': 'vertical',
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
    #             's0': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
    #             's1': {'bus': 'robot1/s_extero', 'shape': (dim_s1, numsteps)},
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
    #                     'input': ['pre_l0', 's0', 'pre_l1'],
    #                     'plot': [
    #                         partial(timeseries, linewidth = 1.0, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, linewidth = 2.0, alpha = 1.0, xticks = False, xlabel = None)],
    #                     'title': 'two-level prediction and measurement (timeseries)',
    #                 },
    #                 {
    #                     'input': ['pre_l0', 's0', 'pre_l1'],
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
    #                     'input': ['pre_l2', 's0'],
    #                     'plot': [
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 0.5, xlabel = None),
    #                     ],
    #                     'title': 'proprio and f(proprio)',
    #                 },
    #                 {
    #                     'input': ['pre_l2', 's0'],
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
