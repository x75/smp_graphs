"""smp_graphs configuration

.. moduleauthor:: Oswald Berthold, 2017

smp_expr0045 transfer function for making uniform random strategy
fail, base version
"""

from smp_base.plot import table

from smp_base.common import compose
from smp_graphs.block import FuncBlock2, TrigBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MeasBlock2, MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2
from smp_graphs.block_plot import TextBlock2

from numpy import sqrt, mean, square
from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin, f_meansquare, f_sum, f_rootmeansquare

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/5
numbins = 21
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

expr_number = 5
expr_name="Experiment {0}".format(expr_number)
desc = """The purpose of this and the next few experiments is to
illustrate the effect of divergence through tight control of the motor
to sensor mapping distortions, and the amount external entropy
injected. The effect is then measured using mutual information,
information distance, root mean squared error, and the earth mover's
distance between a sensorimotor state prediction (action) and a state
measurement by a sensor. Parameters are introduced to the basic $pm_0$
system model which control the magnitude of these effects in the model
system. The divergence is modelled by a transfer function whose values
over the interval $[-1, 1]$ are controlled by three groups of
parameters associated with (unimodal) gaussian deformation, colored
noise, and point-wise independent noise. The principal senorimotor
delay is controlled by the lag parameter. External entropy is modelled
as a noise term $\nu_t$. This experiment extends expr0010 by applying
these measurements to the otherwise unmodified pm$_0$ system of
\\autoref{{eq:smp-expr0000-B}}. The half-Gaussian leakage at the left-
and rightmost bin edges are due to local Gaussian noise present in the
system.""".format()

dim_s0 = 1
numelem = 1001

# local conf dict for looping
lconf = {
    'expr_number': expr_number,
    'expr_name': expr_name,
    'dim': dim_s0,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': numelem,
        'l_a': 1.0,
        'd_a': 0.0,
        'd_s': 1.0,
        's_a': 0.0,
        's_f': 3.0,
        'e': 0.0,
    },
    'div_meas': 'chisq', # 'pyemd', # 'kld'
    'plot_desc': """An episode of {0} steps of a basic agent on the
        pm$_0$ point mass system. The vertical timeseries realizations
        of $X$ in the left column of the second row is the
        proprioceptive state prediction $\\hat{{s}}^{{l_0}}_p$. The
        top left plot represents the system's transfer function
        $h$. The timeseries plot of $Y$ adjacent to the right of the
        transfer function is the resulting measurement
        $s^{{l_0}}_p$. The prediction and result are identical except
        for a noise term. The point wise error $X-Y$ is shown in the
        error panel in the center of the figure. The bin-wise
        divergence is plotted in the bottom right
        panel.""".format(numsteps),
}

expr_number = lconf['expr_number']
div_meas = lconf['div_meas']
plot_desc = lconf['plot_desc']

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
#     dim_s_proprio = dim, dim_s_extero = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio


infodistgen = lconf['infodistgen']

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
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {
                            # 'credit': {'bus': 'budget/credit', 'shape': (1,1)},
                            # 's0': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, 1)},
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

                # new artifical modality m2 with distortion parameters
                ('pre_l2', {
                    'block': ModelBlock2,
                    'params': {
                        'debug': False,
                        'models': {
                            # from top config
                            'infodistgen': infodistgen,
                            # 'infodistgen2': infodistgen,
                        },
                        'inputs': {
                            'x': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, 1)},
                        },
                        'outputs': {
                            'y': {'shape': (dim_s_proprio, 1)},
                            'h': {'shape': (dim_s_proprio, lconf['infodistgen']['numelem']), 'trigger': 'trig/pre_l2_t1'},
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
                            # 'mdltr': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
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
                            'pre': {'shape': (dim_s_proprio, 1)},
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
                'x': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
                # 'y': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
                'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
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
                'x': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
                # 'y': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
                'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
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
                'x1': {'bus': 'robot1/s0', 'shape': (1, numsteps)},
                'x2': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
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
                'x1': {'bus': 'robot1/s0', 'shape': (1, numsteps)},
                'x2': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
            },
            'bins': m_hist_bins,
            'outputs': {
                'x1_p': {'shape': (1, numbins)},
                'x2_p': {'shape': (1, numbins)},
            },
        },
    }),
    
    # m: divergence histos
    ('m_div', {
        'block': MeasBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'mode': 'div',
            'scope': 'local',
            'meas': div_meas, # ['chisq', 'kld'],
            # direct histo input?
            # or signal input
            'inputs': {
                # 'x1': {'bus': 'm_hist/x1_p', 'shape': (1, numbins)},
                # 'x2': {'bus': 'm_hist/x2_p', 'shape': (1, numbins)},
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
    ('m_div_sum', {
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
    
    # m: divergence histos
    ('m_div_kld', {
        'block': MeasBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'mode': 'div',
            'scope': 'local',
            'meas': 'kld',
            # direct histo input?
            # or signal input
            'inputs': {
                # 'x1': {'bus': 'm_hist/x1_p', 'shape': (1, numbins)},
                # 'x2': {'bus': 'm_hist/x2_p', 'shape': (1, numbins)},
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
    ('m_div_kld_sum', {
        'block': FuncBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'func': f_sum,
            'inputs': {
                'x': {'bus': 'm_div_kld/y', 'shape': (1, numbins)},
            },
            'outputs': {
                'y': {'shape': (1, 1)},
            },
        },
    }),
    
    # # m: divergence histos
    # ('m_div_emd', {
    #     'block': MeasBlock2,
    #     'params': {
    #         'blocksize': numsteps,
    #         'debug': False,
    #         'mode': 'div',
    #         'scope': 'local',
    #         'meas': 'emd',
    #         # direct histo input?
    #         # or signal input
    #         'inputs': {
    #             # 'x1': {'bus': 'm_hist/x1_p', 'shape': (1, numbins)},
    #             # 'x2': {'bus': 'm_hist/x2_p', 'shape': (1, numbins)},
    #             'x1_p': {'bus': 'robot1/s0', 'shape': (1, numsteps)},
    #             'x2_p': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
    #             'x1_x': {'bus': 'robot1/s0', 'shape': (1, numsteps)},
    #             'x2_x': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
    #             # 'x1_p': {'bus': 'm_hist/x1_p', 'shape': (1, numbins)},
    #             # 'x1_x': {'bus': 'm_hist/x1_x', 'shape': (1, numbins + 1)},
    #             # 'x2_p': {'bus': 'm_hist/x2_p', 'shape': (1, numbins)},
    #             # 'x2_x': {'bus': 'm_hist/x2_x', 'shape': (1, numbins + 1)},
    #         },
    #         'outputs': {
    #             'y': {'shape': (1, numbins)},
    #         },
    #     },
    # }),
    
    # # m: sum divergence
    # ('m_div_pyemd_sum', {
    #     'block': FuncBlock2,
    #     'params': {
    #         'blocksize': numsteps,
    #         'debug': False,
    #         'func': f_sum,
    #         'inputs': {
    #             'x': {'bus': 'm_div_pyemd/y', 'shape': (1, numbins)},
    #         },
    #         'outputs': {
    #             'y': {'shape': (1, 1)},
    #         },
    #     },
    # }),
    
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
            'savesize': (8, 6),
            'wspace': 0.4,
            'hspace': 0.3,
            'xlim_share': True,
            'ylim_share': True,
            'title': expr_name,
            'inputs': {
                's0': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                'pre_l2_h': {'bus': 'pre_l2/h', 'shape': (dim_s_proprio, 1001)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'budget_mu': {'bus': 'm_budget/y_mu', 'shape': (1, 1)},
                'budget_var': {'bus': 'm_budget/y_var', 'shape': (1, 1)},
                'budget_min': {'bus': 'm_budget/y_min', 'shape': (1, 1)},
                'budget_max': {'bus': 'm_budget/y_max', 'shape': (1, 1)},
                'm_di': {
                    'bus': 'm_di/infodist',
                    'shape': (dim_s_proprio, 1, 1)
                },
                'm_mi': {
                    'bus': 'm_mi/mi',
                    'shape': (dim_s_proprio, 1, 1)
                },
                'm_err': {'bus': 'm_err/y', 'shape': (1, numsteps)},
                'm_rmse': {'bus': 'm_rmse/y', 'shape': (1, 1)},
                'm_div': {'bus': 'm_div/y', 'shape': (1, numbins)},
                'm_div_kld': {'bus': 'm_div_kld/y', 'shape': (1, numbins)},
                'm_sum_div': {'bus': 'm_div_sum/y', 'shape': (1, 1)},
                'm_sum_kld_div': {'bus': 'm_div_kld_sum/y', 'shape': (1, 1)},
                # 'm_div_pyemd': {'bus': 'm_div_pyemd/y', 'shape': (1, numbins)},
                # 'm_sum_pyemd_div': {'bus': 'm_div_pyemd_sum/y', 'shape': (1, 1)},
            },
            
            'desc': plot_desc,
                
            # subplot
            'subplots': [
                # row 1: transfer func, out y time, out y histo
                [
                    {
                        'input': ['pre_l2_h'], 'plot': timeseries,
                        'title': 'transfer function $h$',
                        'title_pos': 'top_out',
                        'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $Y \sim h(X)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend': False,
                        'legend_loc': 'right',
                    },
                    
                    {
                        'input': ['pre_l2'], 'plot': timeseries,
                        'title': 'timeseries $Y$',
                        'title_pos': 'top_out',
                        'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'xlim': None, 'xticks': False, 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        # 'legend': False,
                        'legend': {'$s^{l_0}_p$': 0},
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['pre_l2'], 'plot': histogram,
                        'title': 'histogram $Y$',
                        'title_pos': 'top_out',
                        'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, 'xticklabels': None,
                        'xlabel': 'Rel. freq.',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend': {'$s^{l_0}_p$': 0},
                        'legend_loc': 'left',
                    },
                ],
                
                # row 2: in x time, error x - y time, none
                [
                    {
                        'input': ['s0'], 'plot': timeseries,
                        'title': 'timeseries $X$',
                        'title_pos': 'top_out',
                        'aspect': 2.2/numsteps,
                        'orientation': 'vertical',
                        'xlim': None, 'xticks': False, # 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'yticks': False,
                        'ylim': (-1.1, 1.1),
                        'legend': {'$\hat{s}^{l_0}_p$': 0},
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['m_err'], 'plot': timeseries,
                        'title': 'error $X - Y$',
                        'title_pos': 'top_out',
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xlabel': 'time step $[k]$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': (-1.1, 1.1), # None,
                        'legend': {'$e^{l_0}(\hat{s}^{l_0}_p, s_p)$': 0},
                        # 'legend_loc': 'right',
                    },
                    {},
                ],
                
                # row 3: in x histo, measures global, divergence h1, h2
                [
                    {
                        'input': ['s0'], 'plot': histogram,
                        'title': 'histogram $x$', 'aspect': 'shared', # (1*numsteps)/(2*2.2),
                        'title_pos': 'top_out',
                        'orientation': 'vertical',
                        'xlim': (-1.1, 1.1), 'xinvert': False, # 'xticks': False, 'xticklabels': None, #
                        'xlabel': 'input $x \in [-1, ..., 1]$', 
                        'ylim': None, 'yinvert': True,  # 'yticks': None, 'yticklabels': None,
                        'ylabel': 'Rel. freq.',
                        'legend': {'$\hat{s}^{l_0}_p$': 0},
                        'legend_loc': 'right',
                    },
                    {
                        # 'input': ['budget_%s' % (outk,) for outk in ['mu', 'var', 'min', 'max']] + ['m_mi', 'm_di', 'm_rmse', 'm_sum_div'],
                        # 'shape': [(1, 1) for outk in ['mu', 'var', 'min', 'max', 'm_mi', 'm_di', 'm_rmse', 'm_sum_div']],
                        # 'mode': 'stack',
                        # 'title': 'measures', 'title_pos': 'bottom',
                        # 'plot': table,
                    },
                    {
                        # 'input': ['m_div'],
                        'input': ['m_div', 'm_div_kld'],
                        # 'input': ['m_div', 'm_div_kld', 'm_div_pyemd'],
                        'plot': partial(timeseries, linestyle = 'none', marker = 'o'),
                        'title': 'divergence$(P_X, P_Y)$',
                        'title_pos': 'top_out',
                        'shape': (1, numbins),
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xaxis': m_hist_bincenters, 'xlabel': 'bins $k$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': None, 'ylabel': 'amount of div',
                        'legend': {'%s$(P_X, P_Y)$' % (div_meas, ): 0},
                        # 'legend_loc': 'right',
                    },
                ],
                
            ],
        },
    }),

    # results table
    ('table', {
        'block': TextBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'tex',
            # 'title': """{0} episode measures""".format(expr_number),
            'title': """Episode measures of {0}. From top to bottom,
this table gives the budget descriptive statistics in the top four
rows, followed by the mutual information, information distance, root
means square error, and divergence of the proprioceptive state
prediction and measurement.""".format(expr_number),
            'desc': '',
            'inputs': {
                # global budget stats
                'budget_mu': {'bus': 'm_budget/y_mu', 'shape': (1, 1)},
                'budget_var': {'bus': 'm_budget/y_var', 'shape': (1, 1)},
                'budget_min': {'bus': 'm_budget/y_min', 'shape': (1, 1)},
                'budget_max': {'bus': 'm_budget/y_max', 'shape': (1, 1)},
                # meas0: direct pre2meas
                'm_di': {'bus': 'm_di/infodist', 'shape': (dim_s0, 1, 1)},
                'm_mi': {'bus': 'm_mi/mi', 'shape': (dim_s0, 1, 1)},
                'm_rmse': {'bus': 'm_rmse/y', 'shape': (1, 1)},
                'm_div_sum': {'bus': 'm_div_sum/y', 'shape': (1, 1)},
                'm_div_kld_sum': {'bus': 'm_div_kld_sum/y', 'shape': (1, 1)},
                # 'm_div_pyemd_sum': {'bus': 'm_div_pyemd_sum/y', 'shape': (1, 1)},
            },
            'layout': {
                'numrows': 8,
                'numcols': 2,
                'rowlabels': ['Measure', 'global', 'direct'],
                'collabels': ['budget_mu', 'budget_var', 'budget_min', 'budget_max', 'rmse', 'di', 'chisq', 'kld'],
                # 'collabels': ['budget_mu', 'budget_var', 'budget_min', 'budget_max', 'rmse', 'di', 'chisq', 'kld', 'pyemd'],
                'cells': [
                    ['budget_mu', ] + [None] * 1,
                    ['budget_var', ] + [None] * 1,
                    ['budget_min', ] + [None] * 1,
                    ['budget_max', ] + [None] * 1,
                    # [None, 'm_mi'],
                    [None, 'm_rmse'],
                    [None, 'm_di'],
                    [None, 'm_div_sum'],
                    [None, 'm_div_kld_sum'],
                    # [None, 'm_div_pyemd_sum'],
                ],
            },
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
    #             's0': {'bus': 'robot1/s0', 'shape': (dim_s_proprio, numsteps)},
    #             's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
    #             'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
    #             'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
    #             'm_di': {
    #                 'bus': 'm_di/infodist',
    #                 'shape': (dim_s_proprio, 1, 1)
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
    #             #         'shape': (dim_s_proprio, 1),
    #             #         'plot': [
    #             #             partial(timeseries, linewidth = 1.0, alpha = 1.0, marker = 'o', xlabel = None),
    #             #         ],
    #             #         'title': 'd(proprio, f(proprio))',
    #             #     },
    #             #     {
    #             #         'input': ['m_di'],
    #             #         'ndslice': (slice(None), 0, slice(None)),
    #             #         'shape': (dim_s_proprio, 1),
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
