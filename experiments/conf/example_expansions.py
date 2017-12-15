"""smp_graphs expansions

an expansion is a function f of x and possibly x_t-n for all n in some
index set, taking x to a higher dimensional representation, x \in R^n,
f(x) \in R^m, m>n

they are considered and implemented as 'models' in block_models.py, they are
commonly used in machine learning (e.g. kernel machines, convolutional layers)
or neuroscience (SFA, reservoir computing, ...)

this example loads some sensorimotor data from a logfile and demonstrates
use and configuration of a few expansions
"""

from smp_graphs.block_meas_infth import MIMVBlock2
from smp_graphs.block import SliceBlock2
from smp_graphs.block_plot import ImgPlotBlock2
from smp_graphs.block_models import CodingBlock2, ModelBlock2

ppycnf = {
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170526_160018_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    # 'numsteps': 10000,
    # 'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'numsteps': 2000,
    # 'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', 2K
    'numsteps': 20000,
    'logfile': 'data/experiment_20170530_174612_process_logfiles_pd.h5', # step fast-to-slow newB all concatenated
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

ppycnf2 = {
    # 'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle', # continuous sweep
    'numsteps': 5000,
    # 'logfile': 'data/goodPickles/recording_eC0.20_eA0.02_c0.50_n1000_id0.pickle',
    # 'numsteps': 1000,
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
}

cnf = ppycnf2

# copy params to namespace
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if 'sys_slicespec' in cnf:
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = -60
scanstop = 0
scanlen = scanstop - scanstart

# 1000/1000
winsize = 1
overlap = 1
srcsize = overlap

graph = OrderedDict([
    ('data', {
        'block': FileBlock2,
        'params': {
            'id': 'data',
            'debug': False,
            # 'blocksize': overlap, # numsteps,
            'blocksize': srcsize, # numsteps,
            'type': cnf['logtype'],
            'file': {'filename': cnf['logfile']},
            'outputs': {'log': None, 'x': {'shape': (xdim, srcsize)},
                            'y': {'shape': (ydim, srcsize)}},
        },
    }),

    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('dataslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'dataslice',
            # 'blocksize': overlap,
            'blocksize': srcsize,
            'debug': True,
            # puppy sensors
            'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, srcsize)}},
            'slices': sys_slicespec,
            # 'slices': ,
            }
        }),
        
    ('coding', {
        'block': CodingBlock2,
        'params': {
            'id': 'coding',
            'blocksize': srcsize,
            'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, srcsize)}},
            'outputs': {
                'x_mu': {'shape': (xdim, srcsize)},
                'x_sig': {'shape': (xdim, srcsize)},
                'x_std': {'shape': (xdim, srcsize)},
            },
        }
    }),

    ('mb0', {
        'block': ModelBlock2,
        'params': {
            'id': 'mb0',
            'blocksize': 1,
            'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, 1)}},
            # 'inputs': {'x': {'bus': 'coding/x_std', 'shape': (xdim, 1)}},
            'outputs': {},
            # models': {'res': "reservoir"},
            'models': {
                # this specfies the actually modelling class
                'polyexp': {'type': 'polyexp'},
            },
        }
    }),

    ('mb1', {
        'block': ModelBlock2,
        'params': {
            'id': 'mb1',
            'blocksize': 1,
            # 'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, 1)}},
            'inputs': {'x': {'bus': 'coding/x_std', 'shape': (xdim, 1)}},
            'outputs': {},
            # models': {'res': "reservoir"},
            'models': {
                'musig': {'type': 'musig', 'a1': 0.996},
                'res': {
                    'type': 'res', 'N': 100, 'input_num': xdim, 
                    'output_num': 1, 'input_scale': 1.0, 'bias_scale': 0.0,
                    'oversampling': 2}
            },
        }
    }),

    ('mimv', {
        'block': MIMVBlock2,
        'params': {
            'id': 'mimv',
            'blocksize': 100,
            'debug': False,
            'inputs': {
                'y': {'bus': 'data/x', 'shape': (xdim, 100)},   # src
                'x': {'bus': 'mb1/x_res', 'shape': (100, 100)}, # dst
                # 'norm': {'val': np.array([[7.0]]), 'shape': (1, 1)},
            },
            'embeddingscan': "src",
            'shift': (scanstart, scanstop),
            'outputs': {'mimv': {'shape': (1, scanlen)}}
        }
    }),
    
    ('mimv2', {
        'block': MIMVBlock2,
        'params': {
            'id': 'mimv2',
            'blocksize': 100,
            'debug': False,
            'inputs': {
                'y': {'bus': 'data/y', 'shape': (ydim, 100)},
                'x': {'bus': 'data/x', 'shape': (xdim, 100)},
                # 'norm': {'val': np.array([[7.0]]), 'shape': (1, 1)},
            },
            'shift': (scanstart, scanstop),
            'outputs': {'mimv': {'shape': (1, scanlen)}}
        }
    }),
    
    ('plot_ts', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot_ts',
            'blocksize': 500,
            'debug': False,
            'saveplot': False,
            'savetype': 'pdf',
            'wspace': 0.2, 'hspace': 0.2,
            'inputs': {'d1': {'bus': 'data/y', 'shape': (ydim, numsteps)},
                       'd2': {'bus': 'coding/x_mu', 'shape': (xdim, numsteps)},
                       'd3': {'bus': 'coding/x_sig', 'shape': (xdim, numsteps)},
                       'd4': {'bus': 'coding/x_std', 'shape': (xdim, numsteps)},
                       'd5': {'bus': 'mb1/x_res', 'shape': (100, numsteps)},
                       'd6': {'bus': 'mb1/x_sig', 'shape': (xdim, numsteps)},
                       'd7': {'bus': 'mb0/polyexp', 'shape': (83, numsteps)},
             },
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'ndslice': (slice(None), slice(None)), 'shape': (ydim, numsteps), 'plot': timeseries},
                    {'input': 'd7', 'ndslice': (slice(None), slice(None)), 'shape': (83, numsteps), 'plot': timeseries},
                ],
                [
                    {'input': 'd2', 'ndslice': (slice(None), slice(None)), 'shape': (xdim, numsteps), 'plot': timeseries},
                    {'input': 'd3', 'ndslice': (slice(None), slice(None)), 'shape': (xdim, numsteps), 'plot': timeseries},
                ],
                [
                    {'input': 'd4', 'ndslice': (slice(None), slice(None)), 'shape': (xdim, numsteps), 'plot': timeseries},
                    {'input': 'd4', 'ndslice': (slice(None), slice(None)), 'shape': (xdim, numsteps), 'plot': histogram},
                ],
                [
                    {'input': 'd5', 'ndslice': (slice(None), slice(None)), 'shape': (100, numsteps), 'plot': timeseries},
                    {'input': 'd6', 'ndslice': (slice(None), slice(None)), 'shape': (xdim, numsteps), 'plot': timeseries},
                ],
            ]
        }
    }),

    # plot multivariate (global) mutual information over timeshifts
    ('plot_infth', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_infth',
            'logging': False,
            'saveplot': saveplot,
            'debug': True,
            'wspace': 0.5,
            'hspace': 0.5,
            'blocksize': 100, # numsteps,
            'inputs': {
                'd1': {'bus': 'mimv/mimv', 'shape': (1, scanlen * numsteps/100)},
                'd2': {'bus': 'mimv2/mimv', 'shape': (1, scanlen * numsteps/100)},
                # 'd3': {'bus': 'temv/temv', 'shape': (1, scanlen * numsteps/overlap)},                
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},},
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {'input': 'd1', 
                         'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/100, scanlen)},
                    
                    {'input': 'd2', 
                    'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/100, scanlen)},
                    # {
                    # 'input': 'd3',
                    # 'ndslice': (slice(None), slice(None)),
                    # # 'dimstack': {'x': [2, 1], 'y': [0]},
                    # 'shape': (numsteps/overlap, scanlen),
                    # 'cmap': 'Reds'},
                ],
            ]
        },
    }),
    
])
