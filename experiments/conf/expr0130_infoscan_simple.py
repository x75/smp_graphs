"""smp_graphs infoscan simple

explore the sensorimotor manifold / pointcloud

originally for andi gerken's puppy data, extended to several datasets
/ robots

perform shared information scan for a motor source and a sensory
destination by shifting the destination back in time with respect to
the source. measures are global over the multivariate measurements
(simple)
"""

from collections import OrderedDict
from functools import partial
import numpy as np
from matplotlib.pyplot import hexbin
from smp_base.plot import histogramnd, timeseries, histogram
from smp_graphs.common import escape_backslash
from smp_graphs.utils_conf_meas import make_input_matrix, make_input_matrix_ndim
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block import LoopBlock2
from smp_graphs.block_ols import FileBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2
from smp_graphs.block_meas_infth import MIMVBlock2, CMIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_plot import PlotBlock2, SnsMatrixPlotBlock2, ImgPlotBlock2, TextBlock2

# global config
showplot = True
randseed = 12345
saveplot = False
plotgraph_layout = 'linear_hierarchical'

# add latex output
outputs = {
    'latex': {'type': 'latex',},
}

################################################################################
# legacy
# all files 147000
# 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5',
# medium version 10000
# 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5',
# short version 2000
# 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5',
# test data
# 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
# 'data/experiment_20170512_170835_generate_sin_noise_pd.h5',
# 'data/experiment_20170512_153409_generate_sin_noise_pd.h5',
# '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
# '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
# 'data/testlog3.npz',

# datafiles and dependent params: puppy type 1
ppycnf = {
    # 'numsteps': 27000,
    # # 'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'logfile': 'data/experiment_20170526_160018_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    'numsteps': 10000,
    'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'numsteps': 2000,
    # 'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

# datafiles and dependent params: puppy type 2
ppycnf2 = {
    # python2 pickles
    # 'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_72_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_72_1.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # python3 pickles
    'logfile': 'data/stepPickles/step_period_76_0_p3.pkl',
    'numsteps': 1000,
    'ydim_eff': 1,
    # 'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle',
    # 'numsteps': 5000,
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
}
    
# datafiles and dependent params: sinewave and noise toy example
testcnfsin = {
    'numsteps': 1000,
    'xdim': 1,
    'xdim_eff': 1,
    'ydim': 1,
    'logfile': 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
    'logtype': 'selflog',
}

# datafiles and dependent params: sphero data
sphrcnf = {
	'numsteps': 5000,
	'xdim': 2,
    'xdim_eff': 1,
	'ydim': 1,
    'logtype': 'sphero_res_learner',
    'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
    # 'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
    'sys_slicespec': {'x': {'gyr': slice(0, 1)}},
}

# datafiles and dependent params: additional testdata
testcnf = {
    'numsteps': 1000,
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logfile': 'data/testlog3.npz',
    'logtype': 'testdata1',
}

# configuration for 1D pointmass prediction-measurement delay [wip]
pmcnf = {
}

lconf = {
    'delay_embed_len': 1,
}

# 20200510 - python3 NameError, name ... is not defined
global xdim, ydim
global xdim_eff, ydim_eff
global numsteps
global scanlen
global partial, timeseries

# assign an option to the actual configuration 
cnf = ppycnf2
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if 'ydim_eff' in cnf:
    ydim_eff = cnf['ydim_eff']
else:
    ydim_eff = ydim

# slice out gyro data from imu group
if 'sys_slicespec' in cnf:
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, int(xdim/2)), 'gyr': slice(int(xdim/2), xdim)}}

# configure the scan range
scanstart = 0  # -100
scanstop = int(76*1+1) # 11 # 21 # 51 #    1
scanlen = scanstop - scanstart
delay_embed_len = lconf['delay_embed_len']

# prepare scan plot xticks depending on input size
plot_infoscan_xticks_step = scanlen // 5
plot_infoscan_xticks = list(range(0, scanlen, plot_infoscan_xticks_step))
plot_infoscan_xticklabels = list(range(scanstart*1, scanstop*1, plot_infoscan_xticks_step*1))

lrp_alpha = 0.01
tap_thr = 0.3

datasetname = escape_backslash(cnf['logfile'])
data_x = 'puppyzero/x_r'

expr_number = 17
expr_name = 'Experiment {0}'.format(expr_number)
desc = """A real world robot example is the Puppy robot, initially
proposed in \\cite{{iida_cheap_2004}}. There exist several proposals
for modifications of the original design. Here, a soft legged design
by Andreas Gerken is used, which is described in more detail in
\\parencite{{gerken_behavioral_2017}}. The initial question was, what
is the motor-sensor delay of this robot, measured in units of
sensorimotor loop steps. The answer is that at the given loop
frequency there is no single global delay but rather a set of delays
spread out in time. This is caused by differences in speed of
information propagation through the robot body. In particular,
propagation speed is frequency dependent.

\\par{{The experiment consists of a data source and a maximum window
size prior. Three scans are performed with three types of multivariate
\\emph{{global}} measures that differ in how they account for multiple
channels of coupling. Global means that all source- and destination
variables are each lumped together to compute the shared
information. The scan result is a vector $\text{{scan}}$ with each
scalar element $\text{{scan}}_i$ being a dependency measurement for
the corresponding time shift of $-i$ of the destination with respect
to the source. The learned tappings are compared with a rectangular
window baseline using linear regression probes
\parencite{{alain_understanding_2016}}. If the effective coupling is
sparse within the window, the tapped input outperforms the baseline
probe measured via the mean squared prediction error. In addition the
sparsely tapped probes have significantly lower parameter norms when
the regularization parameter is set to a low value, e.g. here $\\alpha
= {0}$.}}

\\par{{In this run, the same signal is sent to all four motors of
Puppy. The signal consists of a square wave with an amplitude range of
$[-0.2, 0.2] and a period of 76 time steps. The scan length is set to
twice the period length. The periodicity is clearly visible in the
mutual information measurement, which is causally spurious but
statistically correct precicely due to the strict periodicity of the
source.$}}
""".format(lrp_alpha)

# smp graph
graph = OrderedDict([
    # get the data from logfile
    ('puppylog', {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'inputs': {},
            # 'type': 'selflog',
            # 'type': 'sphero_res_learner',
            # 'type': 'testdata1',
            'type': cnf['logtype'],
            'blocksize': numsteps,
            'file': {'filename': cnf['logfile']},
            'outputs': {
                'log': {'shape': None},
                'x': {'shape': (xdim, numsteps)}, 'y': {'shape': (ydim, numsteps)}
            },
        }
    }),

    # ('wav', {
    #     'block': FileBlock2,
    #     'params': {
    #         'blocksize': 1,
    #         'type': 'wav',
    #         # 'file': ['data/res_out.wav'],
    #         'file': {'filename': 'data/res_out.wav', 'filetype': 'wav', 'offset': 100000, 'length': numsteps},
    #         'file': {'filename': '../../smp/sequence/data/blackbird_XC330200/XC330200-1416_299-01hipan.wav', 'filetype': 'wav', 'offset': 0, 'length': numsteps},
            
    #         'outputs': {'x': {'shape': (2, 1)}}
    #         },
    #     }),

    # mean removal / mu-sigma-res coding
    ('puppyzero', {
        'block': ModelBlock2,
        'params': {
            'debug': True,
            'blocksize': numsteps,
            'inputs': {
                'x': {'bus': 'puppylog/x', 'shape': (xdim, numsteps)},
            },
            'models': {
                'msr': {'type': 'msr'},
            },
        }
    }),
    
    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslice',
            'blocksize': numsteps,
            # puppy sensors
            # 'inputs': {'x': {'bus': 'puppylog/x', 'shape': (xdim, numsteps)}},
            'inputs': {'x': {'bus': data_x, 'shape': (xdim, numsteps)}},
            'slices': sys_slicespec,
            }
        }),
            
    # slice block to split puppy motors y into single channels
    ('puppyslice_y', {
        'block': SliceBlock2,
        'params': {
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)}},
            'slices': {'y': dict([('c%d' % i, [i]) for i in range(ydim)])},
        }
    }),
        
    # puppy process data block: integrate acc, diff motors
    ('motordiff', {
        'block': dBlock2,
        'params': {
            'id': 'motordiff',
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)}},
            'outputs': {},
            'd': 0.1,
            'leak': 0.01,
            },
        }),
    
    # puppy process data block: delay motors by lag to align with their sensory effects
    ('motordel', {
        'block': DelayBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'flat2': True,
            # 'inputs': {'y': {'bus': 'motordiff/dy'}},
            'inputs': {
                'x': {'bus': data_x, 'shape': (xdim, numsteps)},
                'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)}
            },
            'delays': {
                'x': list(range(1, delay_embed_len+1)), # [1],
                # 'y': range(1, delay_embed_len+1), # [1, 0, -1, -2, -3], # * delay_embed_len, # [1],
                'y': list(range(0, delay_embed_len)), # [1, 0, -1, -2, -3], # * delay_embed_len, # [1],
            },
        }
    }),

    # stack delay x and y together as condition for m -> s mi
    ('motorstack', {
        'block': StackBlock2,
        'params': {
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {
                'x': {'bus': 'motordel/dx'},# 'shape': (xdim, )},
                'y': {'bus': 'motordel/dy'},
            },
            'outputs': {'y': {'shape': (
                xdim * delay_embed_len + ydim * delay_embed_len, numsteps
            )}} # overwrite
        }
    }),
    
    # joint entropy
    ('jh', {
        'block': JHBlock2,
        'enable': False,
        'params': {
            'id': 'jh',
            'blocksize': numsteps,
            'debug': False,
            # 'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
            'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}},
            'shift': (scanstart, scanstop),
            # 'outputs': {'mi': [((ydim + xdim)**2, 1)}}
            'outputs': {'jh': {'shape': (1, scanlen)}}
        }
    }),
        
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('mimv', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'debug': False,
            'id': 'mimv',
            'loop': [
                (
                    'inputs', {
                        'x': {'bus': data_x},
                        'y': {'bus': 'puppylog/y'}
                    }
                ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_gyr'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_acc'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),
                
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIMVBlock2,
                'params': {
                    'id': 'mimv',
                    'blocksize': numsteps,
                    'debug': False,
                    # 'norm_out': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'mimv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('cmimv', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'debug': False,
            'loop': [
                (
                    'inputs', {
                        'x': {'bus': data_x},
                        # 'y': {'bus': 'puppylog/y'},
                        'y': {'bus': 'motordel/dy'},
                        # 'cond': {'bus': 'motorstack/y'},
                        # 'cond': {'bus': 'motordel/dy'},
                        'cond_delay': {'bus': 'motordel/dx'},
                    }
                ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_gyr'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_acc'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),
                
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': CMIMVBlock2,
                'params': {
                    # 'id': 'mimv',
                    'blocksize': numsteps,
                    'debug': False,
                    # 'norm_out': False,
                    'inputs': {
                        'x': {'bus': 'puppyslice/x_gyr'},
                        'y': {'bus': 'puppylog/y'},
                        'cond': {'bus': 'motordel/dy'}
                    },
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'cmimv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('temv', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'id': 'temv',
            'loop': [
                (
                    'inputs', {
                        'x': {'bus': data_x},
                        'y': {'bus': 'puppylog/y'}
                    }
                ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_gyr'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),
                
                # (
                #     'inputs', {
                #         'x': {'bus': 'puppyslice/x_acc'},
                #         'y': {'bus': 'puppylog/y'}
                #     }
                # ),

            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': TEMVBlock2,
                'params': {
                    'id': 'temv',
                    'blocksize': numsteps,
                    'debug': False,
                    'k': delay_embed_len, 'l': delay_embed_len,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'temv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),

    # # mutual information analysis of data
    # ('infodist', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'infodist',
    #         'loop': [('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}}),
    #                  # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
    #                  # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
    #         ],
    #         'loopblock': {
    #             'block': InfoDistBlock2,
    #             'params': {
    #                 'id': 'infodist',
    #                 'blocksize': numsteps,
    #                 'debug': False,
    #                 'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}},
    #                 'shift': (-3, 4),
    #                 # 'outputs': {'infodist': {'shape': ((ydim + xdim)**2, 1)}}
    #                 'outputs': {'infodist': {'shape': (7 * ydim * xdim_eff, 1)}}
    #             }
    #         },
    #     }
    # }),

    # scan to tap
    ('tap', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'loop': [

                (
                    'inputs', {'qtap': {'bus': 'mimv_ll0_ll0/mimv'}}
                ),
                
                (
                    'inputs', {'qtap': {'bus': 'cmimv_ll0_ll0/cmimv'}}
                ),
                
                (
                    'inputs', {'qtap': {'bus': 'temv_ll0_ll0/temv'}}
                ),
                # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': ModelBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    'inputs': {
                        'qtap': {'bus': 'mimv_ll0_ll0/mimv'},
                    },
                    'outputs': {
                        'tap_x': {'shape': (1, scanlen)},
                        'tap_y': {'shape': (1, scanlen)},
                    },
                    'models': {
                        'tap': {'type': 'qtap', 'thr': tap_thr,}
                    },
                },
            },
        },
    }),
    
    # linear regression probe
    ('lrp', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'loop': [
                (
                    'inputs', {
                        'tap': {'val': np.arange(0, scanlen)},
                        'x': {'bus': 'puppylog/y'},
                        'y': {'bus': data_x},
                    }
                ),

                (
                    'inputs', {
                        'tap': {'bus': 'tap_ll0_ll0/tap_x'},
                        'x': {'bus': 'puppylog/y'},
                        'y': {'bus': data_x},
                    }
                ),

                (
                    'inputs', {
                        'tap': {'bus': 'tap_ll1_ll0/tap_x'},
                        'x': {'bus': 'puppylog/y'},
                        'y': {'bus': data_x},
                    }
                ),
                
                (
                    'inputs', {
                        'tap': {'bus': 'tap_ll2_ll0/tap_x'},
                        'x': {'bus': 'puppylog/y'},
                        'y': {'bus': data_x},
                    }
                ),
                # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': ModelBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    'inputs': {
                        'tap': {'bus': 'tap/tap_x'},
                        'x': {'bus': 'puppylog/y'},
                        'y': {'bus': data_x},
                    },
                    'outputs': {
                        'y': {'shape': (xdim, numsteps)},
                        'y_res': {'shape': (1,1)},
                        'y_idx': {'shape': (1,scanlen)},
                        'w_norm': {'shape': (1,1)},
                        'b_norm': {'shape': (1,1)},
                    },
                    'models': {
                        'lrp': {'type': 'linear_regression_probe', 'alpha': lrp_alpha, 'meas': 'rmse'}
                    },
                },
            },
        },
    }),


    # slice block to split puppy motors y into single channels
    ('lrpslice', {
        'block': LoopBlock2,
        # 'enable': False,
        'params': {
            'loop': [
                (
                    'inputs', {
                        'y': {'bus': 'lrp_ll%d_ll0/y' % i, 'shape': (xdim, numsteps)},
                    }
                ) for i in range(4)],
            'loopmode': 'parallel',
            'loopblock': {
                'block': SliceBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    # puppy sensor predictions
                    'inputs': {},
                    'slices': {'y': {'acc': slice(0, int(xdim/2)), 'gyr': slice(int(xdim/2), xdim)}},
                },
            },
        },
    }),
    
    # elementwise measures #########################################################

    # mutual information
    ('mi', {
        'block': LoopBlock2,
        'enable': False,
        'params': {
            'id': 'mi',
            'loop': [
                (
                    'inputs', {
                        'x': {'bus': 'puppyslice/x_gyr'},
                        'y': {'bus': 'puppylog/y'}
                    }
                ),
                # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': True,
                    'inputs': {
                        'x': {'bus': 'puppyslice/x_gyr'},
                        'y': {'bus': 'puppylog/y'}
                    },
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    # 'outputs': {'mi': {'shape': (scanlen * ydim * xdim_eff, 1)}}
                    'outputs': {'mi': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # transfer entropy
    ('te', {
        'block': LoopBlock2,
        'enable': False,
        'params': {
            'id': 'te',
            'loopmode': 'parallel',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': TEBlock2,
                'params': {
                    'id': 'te',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    'outputs': {'te': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # cond motor_i!=j conditional transfer entropy
    ('cte', {
        'block': LoopBlock2,
        'enable': False,
        'params': {
            'id': 'cte',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': CTEBlock2,
                'params': {
                    'id': 'cte',
                    'blocksize': numsteps,
                    'debug': False,
                    'xcond': True,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    # change this to ndim data, dimstack
                    # 'outputs': {'cte': {'shape': (scanlen * ydim * xdim_eff, 1)}}
                    'outputs': {'cte': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # # # puppy process data block: integrate acc, diff motors
    # # ('accint', {
    # #     'block': IBlock2,
    # #     'params': {
    # #         'id': 'accint',
    # #         'blocksize': numsteps,
    # #         'inputs': {'x_acc': {'bus': 'puppyslice/x_acc'}},
    # #         'outputs': {},
    # #         'd': 0.1,
    # #         'leak': 0.01,
    # #         },
    # #     }),
    
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslicem', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'puppyslicem',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': {'x': {'bus': 'puppylog/y'}},
    #         'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    #     }
    # }),
     
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppystack', {
    #     'block': StackBlock2,
    #     'params': {
    #         'id': 'puppystack',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': make_input_matrix('xcorr', 'xcorr', xdim = xdim, ydim = ydim, scan = (scanstart, scanstop))),
    #         # 'inputs': {'x': {'bus': 'puppylog/y'}},
    #         # 'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    #         'outputs': {'y': {'shape': (xdim * ydim, 1)}} # overwrite
    #     }
    # }),
    
    # # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # # ('puppyslicemd', {
    # #     'block': SliceBlock2,
    # #     'params': {
    # #         'id': 'puppyslicemd',
    # #         'blocksize': numsteps,
    # #         # puppy sensors
    # #         'inputs': {'x': {'bus': 'motordel/dy'}},
    # #         'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    # #     }
    # # }),
    
    # plot raw data timeseries and histograms
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'wspace': 0.5,
            'hspace': 0.5,
            'saveplot': saveplot,
            'title_pos': 'top_out',
            'inputs': {
                'x_gyr': {'bus': 'puppyslice/x_gyr'},
                'x_acc': {'bus': 'puppyslice/x_acc'},
                # 'd4': {'bus': 'accint/Ix_acc'}, # 'puppylog/y'}
                # 'x_gyr': {'bus': data_x},
                'd4': {'bus': 'puppylog/y'}, # 'puppylog/y'}
                # 'd5': {'bus': 'motordel/dy'}, # 'puppylog/y'}
                # 'd6': {'bus': 'puppyslicem/x_y0'}, # /t
                # 'd7': {'bus': 'puppyslicemd/x_y0'}, # /t
                'lrp0': {'bus': 'lrp_ll0_ll0/y'},
                'lrp1': {'bus': 'lrp_ll1_ll0/y'},
                'lrp2': {'bus': 'lrp_ll1_ll0/y'},
                'lrp3': {'bus': 'lrp_ll1_ll0/y'},
                'lrp_y_acc_0': {'bus': 'lrpslice_ll0_ll0/y_acc'},
                'lrp_y_acc_1': {'bus': 'lrpslice_ll1_ll0/y_acc'},
                'lrp_y_acc_2': {'bus': 'lrpslice_ll2_ll0/y_acc'},
                'lrp_y_acc_3': {'bus': 'lrpslice_ll3_ll0/y_acc'},
                'lrp_y_gyr_0': {'bus': 'lrpslice_ll0_ll0/y_gyr'},
                'lrp_y_gyr_1': {'bus': 'lrpslice_ll1_ll0/y_gyr'},
                'lrp_y_gyr_2': {'bus': 'lrpslice_ll2_ll0/y_gyr'},
                'lrp_y_gyr_3': {'bus': 'lrpslice_ll3_ll0/y_gyr'},
            },
            'outputs': {},#'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {
                        'input': ['x_gyr'] + ['lrp_y_gyr_%i' % i for i in range(4)],
                        'plot': timeseries,
                        'title': 'Gyros',
                        'ylim': (-0.3, 0.3),
                        # label: handle-idx
                        'legend': {'gyr-meas': 1, 'gyr-pred-0': 4, 'gyr-pred-1': 7, 'gyr-pred-2': 10, 'gyr-pred-3': 13},
                    },
                    {
                        'input': ['x_gyr'],
                        'plot': histogram,
                        'ylim': (-0.3, 0.3),
                        'title': 'Gyros histogram'
                    },
                ],
                [
                    {
                        'input': ['x_acc'] + ['lrp_y_acc_%i' % i for i in range(4)],
                        'plot': timeseries,
                        'title': 'Accelerometers',
                        'ylim': (-0.5, 0.5),
                        'legend': {'acc-meas': 1, 'acc-pred-0': 4, 'acc-pred-1': 7, 'acc-pred-2': 10, 'acc-pred-3': 13},
                    },
                    {
                        'input': ['x_acc'],
                        'plot': histogram,
                        'title': 'Accelerometers histogram',
                        'ylim': (-0.5, 0.5),
                    },
                ],
                [
                    {
                        'input': ['d4'],
                        'plot': timeseries,
                        'title': 'Motor',
                    },
                    {
                        'input': ['d4'],
                        'plot': histogram,
                        'title': 'Motor histogram'},
                ],
                # [
                #     {'input': ['d5'], 'plot': timeseries},
                #     {'input': 'd5', 'plot': histogram},
                # ],
                # [
                #     # {'input': ['d6', 'd7'], 'plot': partial(timeseries, marker = ".")},
                #     {'input': ['d3', 'd4'], 'plot': partial(timeseries, marker = ".")},
                #     {'input': 'd6', 'plot': timeseries},
                # ],
            ]
        },
    }),
    
    # # cross correlation
    # ('xcorr', {
    #     'block': XCorrBlock2,
    #     'enable': False,
    #     'params': {
    #         'id': 'xcorr',
    #         'blocksize': numsteps,
    #         # 'inputs': {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/y'}},
    #         'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppyslice_y/y_c0'}},
    #         'shift': (scanstart, scanstop),
    #         'outputs': {'xcorr': {'shape': (ydim_eff, xdim, scanlen)}},
    #         }
    #     }),

    # # plot cross-correlation matrix
    # ('plot_xcor_line', {
    #     'block': PlotBlock2,
    #     'enable': False,
    #     'params': {
    #         'id': 'plot_xcor_line',
    #         'logging': False,
    #         'debug': False,
    #         'saveplot': saveplot,
    #         'blocksize': numsteps,
    #         'inputs': make_input_matrix_ndim(
    #             xdim = xdim, ydim = ydim,
    #             with_t = True, scan = (scanstart, scanstop)),
    #         'outputs': {}, #'x': {'shape': (3, 1)}},
    #         'wspace': 0.5,
    #         'hspace': 0.5,
    #         # 'xslice': (0, scanlen), 
    #         'subplots': [
    #             [
    #                 {
    #                     'input': ['d3'], 'ndslice': (slice(scanlen), i, j),
    #                     'xaxis': 't',
    #                     'shape': (1, scanlen),
    #                     'plot': partial(timeseries, linestyle="-", marker=".")
    #                 } for j in range(xdim)
    #             ] for i in range(ydim)
    #         ],
                
    #         #     [{'input': 'd3_%d_%d' % (i, j), 'xslice': (0, scanlen), 'xaxis': 't',
    #         #       'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(xdim)]
    #         # for i in range(ydim)],
            
    #     },
    # }),

    # # plot cross-correlation matrix
    # ('plot_xcorr_scan', {
    #     'block': ImgPlotBlock2,
    #     'enable': False,
    #     'params': {
    #         'id': 'plot_xcor_img',
    #         'logging': False,
    #         'saveplot': saveplot,
    #         'debug': False,
    #         'desc': 'Cross-correlation scan for the %s dataset' % (datasetname),
    #         'title': 'Cross-correlation scan for the %s dataset' % (cnf['logfile']),
    #         'blocksize': numsteps,
    #         'inputs': make_input_matrix_ndim(xdim = xdim, ydim = ydim_eff, with_t = True, scan = (scanstart, scanstop)),
    #         'wspace': 0.5,
    #         'hspace': 0.5,
    #         'subplots': [
                
    #             [
    #                 {
    #                     'input': ['d3'],
    #                     'ndslice': (slice(scanlen), i, j),
    #                     'shape': (1, scanlen), 'cmap': 'RdGy',
    #                     'title': 'xcorr s-%d/s-%d' % (i, j),
    #                     'vmin': -1.0, 'vmax': 1.0,
    #                     'vaxis': 'cols',
    #                     'xlabel': False,
    #                     'xticks': i == (ydim - 1), # False,
    #                     'ylabel': None,
    #                     'yticks': False,
    #                     'colorbar': j == (xdim - 1), 'colorbar_orientation': 'vertical',
    #                 } for j in range(xdim)
    #             ] for i in range(ydim_eff)
                
    #         ],
            
    #     },
    # }),


    
    # # plot multivariate (global) mutual information over timeshifts
    # ('plot_infoscan', {
    #     'block': ImgPlotBlock2,
    #     'enable': False,
    #     'params': {
    #         'logging': False,
    #         'saveplot': saveplot,
    #         'savesize': (3 * 3, 3),
    #         'debug': False,
    #         'wspace': 0.5,
    #         'hspace': 0.5,
    #         'blocksize': numsteps,
    #         'desc':  'Mutual information, conditional MI and TE scan for dataset %s' % (datasetname),
    #         'title': 'Mutual information, conditional MI and TE scan for dataset %s' % (datasetname),
    #         'inputs': {
    #             'd1': {'bus': 'mimv_ll0_ll0/mimv', 'shape': (1, scanlen)},
    #             'd3': {'bus': 'cmimv_ll0_ll0/cmimv', 'shape': (1, scanlen)},
    #             'd2': {'bus': 'temv_ll0_ll0/temv', 'shape': (1, scanlen)},
    #             # 'tap1': {'bus': 'tap_ll1_ll0/tap_x', 'shape': (1, scanlen)},
    #             # 'tap2': {'bus': 'tap_ll2_ll0/tap_x', 'shape': (1, scanlen)},
    #             # 'tap3': {'bus': 'tap_ll3_ll0/tap_x', 'shape': (1, scanlen)},
    #             # 'd3': {'bus': 'jh/jh', 'shape': (1, scanlen)},
    #             't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},
    #         },
    #         'outputs': {}, #'x': {'shape': (3, 1)}},
    #         'subplots': [
    #             [
                    
    #                 {
    #                     'input': 'd1', 'xslice': (0, scanlen),
    #                     'xticks': list(range(0, scanlen, 5)),
    #                     'xticklabels': list(range(scanstart*1, scanstop*1, 5*1)),
    #                     'xlabel': 'Lag [n]',
    #                     'yslice': (0, 1),
    #                     'ylabel': None,
    #                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
    #                     'title': 'Mutual information $I(X;Y)$',
    #                     'colorbar': True, 'colorbar_orientation': 'vertical',
    #                     'shape': (1, scanlen)
    #                 },
                    
    #                 {
    #                     'input': 'd3',
    #                     'xslice': (0, scanlen),
    #                     'xticks': list(range(0, scanlen, 5)),
    #                     'xticklabels': list(range(scanstart*1, scanstop*1, 5*1)),
    #                     'xlabel': 'Lag [n]',
    #                     'yslice': (0, 1),
    #                     'ylabel': None,
    #                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
    #                     'title': 'Cond. MI $CMI(Y;X;C)$',
    #                     'colorbar': True, 'colorbar_orientation': 'vertical',
    #                     'shape': (1, scanlen)
    #                 },
                    
    #                 {
    #                     'input': 'd2',
    #                     'xslice': (0, scanlen),
    #                     'xticks': list(range(0, scanlen, 5)),
    #                     'xticklabels': list(range(scanstart*1, scanstop*1, 5*1)),
    #                     'xlabel': 'Lag [n]',
    #                     'yslice': (0, 1),
    #                     'ylabel': None,
    #                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
    #                     'title': 'Transfer entropy $TE(Y;X;X^-)$',
    #                     'colorbar': True, 'colorbar_orientation': 'vertical',
    #                     'shape': (1, scanlen)
    #                 }
                    
    #             ],
                
    #         ]
    #     },
    # }),
    
    # plot multivariate (global) mutual information over timeshifts
    ('plot_infoscan2', {
        'block': ImgPlotBlock2,
        # 'enable': False,
        'params': {
            # 'logging': False,
            'saveplot': saveplot,
            'savesize': (4 * 3, 4 * 1),
            'savetype': 'pdf',
            # 'debug': True,
            'wspace': 0.2,
            'hspace': 0.45,
            'blocksize': numsteps,
            'desc':  'Taps from info scan for dataset %s' % (datasetname),
            'title': 'Taps from info scan for dataset %s' % (cnf['logfile']),
            'title_pos': 'top_out',
            'vlim_share': False,
            'inputs': {
                'duniform': {'val': np.ones((1, scanlen)) * 0.01},
                'd1': {'bus': 'mimv_ll0_ll0/mimv', 'shape': (1, scanlen)},
                'd3': {'bus': 'cmimv_ll0_ll0/cmimv', 'shape': (1, scanlen)},
                'd2': {'bus': 'temv_ll0_ll0/temv', 'shape': (1, scanlen)},

                # 'tap1': {'bus': 'tap_ll1_ll0/tap_x', 'shape': (1, scanlen)},
                # 'tap2': {'bus': 'tap_ll2_ll0/tap_x', 'shape': (1, scanlen)},
                # 'tap3': {'bus': 'tap_ll3_ll0/tap_x', 'shape': (1, scanlen)},

                'tap0': {'bus': 'lrp_ll0_ll0/y_idx', 'shape': (1, scanlen)},
                'tap1': {'bus': 'lrp_ll1_ll0/y_idx', 'shape': (1, scanlen)},
                'tap2': {'bus': 'lrp_ll2_ll0/y_idx', 'shape': (1, scanlen)},
                'tap3': {'bus': 'lrp_ll3_ll0/y_idx', 'shape': (1, scanlen)},
                # 'd3': {'bus': 'jh/jh', 'shape': (1, scanlen)},
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},
            },
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [

                [
                    
                    {
                        'input': 'duniform', 'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': False, # 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        # 'vmax': 0.1,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Uniform baseline',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'd1', 'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': False, # 'Lag [n]',
                        'yslice': (0, 1),
                        'yticks': False,
                        'ylabel': False,
                        'vmin': 0,
                        # 'vmax': 0.1,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Mutual information $I(X;Y)$',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'd3',
                        'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': False, # 'Lag [n]',
                        'yslice': (0, 1),
                        'yticks': False,
                        'ylabel': False,
                        'vmin': 0,
                        # 'vmax': 0.1,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Cond. MI $CMI(Y;X;C)$',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'd2',
                        'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': False, # 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        # 'vmax': 0.1,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Transfer entropy $TE(Y;X;X^-)$',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    }
                    
                ],

                # tapping row
                [
                    
                    {
                        'input': ['tap0'], 'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Uniform tapping',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'tap1', 'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Computed tapping (MI)',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'tap3',
                        'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Computed tapping (CMI)',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    },
                    
                    {
                        'input': 'tap2',
                        'xslice': (0, scanlen),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'xlabel': 'Lag [n]',
                        'yslice': (0, 1),
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0,
                        'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                        'title': 'Computed tapping (TE)',
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'shape': (1, scanlen)
                    }
                    
                ],
                
            ] 
       },
    }),
    
    # plot multivariate (global) mutual information over timeshifts
    ('plot_infoscan2_ts', {
        'block': PlotBlock2,
        # 'enable': False,
        'params': {
            # 'logging': False,
            'saveplot': saveplot,
            'savesize': (4 * 3, 4 * 1),
            'savetype': 'pdf',
            # 'debug': True,
            'wspace': 0.2,
            'hspace': 0.45,
            'blocksize': numsteps,
            'desc':  'Taps from info scan for dataset %s' % (datasetname),
            'title': 'Taps from info scan for dataset %s' % (cnf['logfile']),
            'title_pos': 'top_out',
            'vlim_share': False,
            'inputs': {
                'duniform': {'val': np.ones((1, scanlen)) * 0.01},
                'd1': {'bus': 'mimv_ll0_ll0/mimv', 'shape': (1, scanlen)},
                'd3': {'bus': 'cmimv_ll0_ll0/cmimv', 'shape': (1, scanlen)},
                'd2': {'bus': 'temv_ll0_ll0/temv', 'shape': (1, scanlen)},

                # 'tap1': {'bus': 'tap_ll1_ll0/tap_x', 'shape': (1, scanlen)},
                # 'tap2': {'bus': 'tap_ll2_ll0/tap_x', 'shape': (1, scanlen)},
                # 'tap3': {'bus': 'tap_ll3_ll0/tap_x', 'shape': (1, scanlen)},

                'tap0': {'bus': 'lrp_ll0_ll0/y_idx', 'shape': (1, scanlen)},
                'tap1': {'bus': 'lrp_ll1_ll0/y_idx', 'shape': (1, scanlen)},
                'tap2': {'bus': 'lrp_ll2_ll0/y_idx', 'shape': (1, scanlen)},
                'tap3': {'bus': 'lrp_ll3_ll0/y_idx', 'shape': (1, scanlen)},
                # 'd3': {'bus': 'jh/jh', 'shape': (1, scanlen)},
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},
            },
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [

                [
                    {
                        'input': 'd1',
                        'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        # 'ndslice': [(slice(scanlen), i, 0)] * 2,
                        # 'shape': (dim_s1, scanlen),
                        'cmap': ['glasbey_warm'],
                        'title': 'Mutual information $I(X;Y)$',
                        'title_pos': 'top_out',
                        'ylim': (0.0, 0.1),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'legend': {'mutual information': 0}, # 'MI prop': 0, 
                    },

                    {
                        'input': 'd3',
                        'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        # 'ndslice': [(slice(scanlen), i, 0)] * 2,
                        # 'shape': (dim_s1, scanlen),
                        'cmap': ['glasbey_warm'],
                        'title': 'Cond. MI $CMI(Y;X;C)$',
                        'title_pos': 'top_out',
                        'ylim': (0.0, 0.02),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'legend': {'mutual information': 0}, # 'MI prop': 0, 
                    },
                    
                    {
                        'input': 'd2',
                        'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        # 'ndslice': [(slice(scanlen), i, 0)] * 2,
                        # 'shape': (dim_s1, scanlen),
                        'cmap': ['glasbey_warm'],
                        'title': 'Transfer entropy $TE(Y;X;X^-)$',
                        'title_pos': 'top_out',
                        'ylim': (0.0, 0.02),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        'legend': {'mutual information': 0}, # 'MI prop': 0, 
                    }
                    
                    #  {
                    #     'input': 'duniform', 'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': False, # 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     # 'vmax': 0.1,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Uniform baseline',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'd1', 'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': False, # 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'yticks': False,
                    #     'ylabel': False,
                    #     'vmin': 0,
                    #     # 'vmax': 0.1,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Mutual information $I(X;Y)$',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'd3',
                    #     'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': False, # 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'yticks': False,
                    #     'ylabel': False,
                    #     'vmin': 0,
                    #     # 'vmax': 0.1,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Cond. MI $CMI(Y;X;C)$',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'd2',
                    #     'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': False, # 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     # 'vmax': 0.1,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Transfer entropy $TE(Y;X;X^-)$',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # }
                    
                ],

                # tapping row
                [

                    {
                        'input': 'tap1', 'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        'cmap': ['glasbey_warm'],
                        'title': 'Mutual information taps',
                        'title_pos': 'top_out',
                        # 'ylim': (-1.2, 1.2),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        # 'xticklabels': list(range(scanstart, scanstop)),
                        'legend': {'mutual information taps': 0}, # 'MI prop': 0, 
                    },

                    {
                        'input': 'tap3', 'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        'cmap': ['glasbey_warm'],
                        'title': 'Mutual information taps',
                        'title_pos': 'top_out',
                        # 'ylim': (-1.2, 1.2),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        # 'xticklabels': list(range(scanstart, scanstop)),
                        'legend': {'mutual information taps': 0}, # 'MI prop': 0, 
                    },

                    {
                        'input': 'tap2', 'xslice': (0, scanlen),
                        'plot': partial(timeseries, linestyle="none", marker="o"),
                        'cmap': ['glasbey_warm'],
                        'title': 'Mutual information taps',
                        'title_pos': 'top_out',
                        # 'ylim': (-1.2, 1.2),
                        'xticks': plot_infoscan_xticks,
                        'xticklabels': plot_infoscan_xticklabels,
                        # 'xticklabels': list(range(scanstart, scanstop)),
                        'legend': {'mutual information taps': 0}, # 'MI prop': 0, 
                    }
                    

                    # {
                    #     'input': ['tap0'], 'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Uniform tapping',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'tap1', 'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Computed tapping (MI)',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'tap3',
                    #     'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Computed tapping (CMI)',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # },
                    
                    # {
                    #     'input': 'tap2',
                    #     'xslice': (0, scanlen),
                    #     'xticks': plot_infoscan_xticks,
                    #     'xticklabels': plot_infoscan_xticklabels,
                    #     'xlabel': 'Lag [n]',
                    #     'yslice': (0, 1),
                    #     'ylabel': False,
                    #     'yticks': False,
                    #     'vmin': 0,
                    #     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    #     'title': 'Computed tapping (TE)',
                    #     'colorbar': True, 'colorbar_orientation': 'vertical',
                    #     'shape': (1, scanlen)
                    # }
                    
                ],
                
            ] 
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
            'title': 'Results expr0130 for uniformly tapped and info-tapped prediction probes',
            'desc': 'Residual (RMSE), predictor weight norm ($|W|$), bias norm ($|b|$)',
            'inputs': {
                # baseline
                'lrp0_res': {'bus': 'lrp_ll0_ll0/y_res', 'shape': (1, 1)},
                'lrp0_w_norm': {'bus': 'lrp_ll0_ll0/w_norm', 'shape': (1, 1)},
                'lrp0_b_norm': {'bus': 'lrp_ll0_ll0/w_norm', 'shape': (1, 1)},
                # baseline
                'lrp1_res': {'bus': 'lrp_ll1_ll0/y_res', 'shape': (1, 1)},
                'lrp1_w_norm': {'bus': 'lrp_ll1_ll0/w_norm', 'shape': (1, 1)},
                'lrp1_b_norm': {'bus': 'lrp_ll1_ll0/w_norm', 'shape': (1, 1)},
                # baseline
                'lrp2_res': {'bus': 'lrp_ll2_ll0/y_res', 'shape': (1, 1)},
                'lrp2_w_norm': {'bus': 'lrp_ll2_ll0/w_norm', 'shape': (1, 1)},
                'lrp2_b_norm': {'bus': 'lrp_ll2_ll0/w_norm', 'shape': (1, 1)},
                # baseline
                'lrp3_res': {'bus': 'lrp_ll3_ll0/y_res', 'shape': (1, 1)},
                'lrp3_w_norm': {'bus': 'lrp_ll3_ll0/w_norm', 'shape': (1, 1)},
                'lrp3_b_norm': {'bus': 'lrp_ll3_ll0/w_norm', 'shape': (1, 1)},
            },
            'layout': {
                'numrows': 4,
                'numcols': 3,
                'collabels': ['Baseline', 'MI', 'CMI', 'TE'],
                'rowlabels': ['Tapping', 'RMSE', '$|W|$', '$|b|$'],
                'cells': [
                    ['lrp0_res', 'lrp0_w_norm', 'lrp0_b_norm'],
                    ['lrp1_res', 'lrp1_w_norm', 'lrp1_b_norm'],
                    ['lrp2_res', 'lrp2_w_norm', 'lrp2_b_norm'],
                    ['lrp3_res', 'lrp3_w_norm', 'lrp3_b_norm'],
                ],
            },
        },
    }),

    # plot mi matrix as image
    ('plot_mi_te', {
        'block': ImgPlotBlock2,
        'enable': False,
        'params': {
            'id': 'plot_mi_te',
            'logging': False,
            'saveplot': saveplot,
            'debug': False,
            'wspace': 0.1,
            'hsapce': 0.1,
            'blocksize': numsteps,
            # 'inputs': {'d1': {'bus': 'mi_1/mi'}, 'd2': {'bus': 'infodist_1/infodist'}},
            'inputs': {'d1': {'bus': 'mi_ll0_ll0/mi'}, 'd2': {'bus': 'te_ll0_ll0/te'}, 'd3': {'bus': 'cte_ll0_ll0/cte'}},
            'outputs': {}, #'x': {'shape': (3, 1)}},
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, (xdim + ydim)**2),
            #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, (xdim + ydim)**2),
            #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
            #     ],
            # ],
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, xdim * ydim),
            #              'shape': (ydim, xdim), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, xdim * ydim),
            #              'shape': (ydim, xdim), 'plot': 'bla'},
            #     ],
            # ],
            'subplots': [
                [
                    {'input': 'd1',
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'shape': (ydim, xdim_eff),
                     'title': 'mi-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'plot': 'bla'} for i in range(scanlen)
                ],
                [
                    {'input': 'd2',
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'title': 'te-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'shape': (ydim, xdim_eff),
                     'plot': 'bla'} for i in range(scanlen)
                ],
                [
                    {'input': 'd3',
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'title': 'cte-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'shape': (ydim, xdim_eff),
                     'plot': 'bla'} for i in range(scanlen)
                ],
            ],
        },
    }),
    
    # sns matrix plot
    ('plot_snsmat', {
        'block': SnsMatrixPlotBlock2,
        'enable': False,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': True,
            'saveplot': saveplot,
            'blocksize': numsteps,
            'plotf_diag': hexbin,
            'plotf_offdiag': hexbin,
            'inputs': {
                'd3': {'bus': 'puppyslice/x_gyr'},
                # 'd3': {'bus': data_x},
                # 'd3': {'bus': 'motordel/dx'},
                # 'd4': {'bus': 'puppylog/y'},
                'd4': {'bus': 'motordel/dy'},
            },
            'outputs': {},#'x': {'shape': 3, 1)}},
            'subplots': [
                [
                    # stack inputs into one vector (stack, combine, concat
                    {'input': ['d3', 'd4'], 'mode': 'stack',
                         'plot': histogramnd},
                ],
            ],
        },
    })
])
