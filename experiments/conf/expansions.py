"""smp_graphs perform windowed short time mutual info scan"""

from smp_graphs.block_meas_infth import MIMVBlock2
from smp_graphs.block import SliceBlock2
from smp_graphs.block_plot import ImgPlotBlock2
from smp_graphs.block_models import CodingBlock2

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
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'numsteps': 5000,
}

cnf = ppycnf2

# copy params to namespace
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if cnf.has_key('sys_slicespec'):
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = -30
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
            'inputs': {'x': {'bus': 'data/y', 'shape': (ydim, srcsize)}},
            'outputs': {'x_mu': {'shape': (ydim, srcsize)}, 'x_sig': {'shape': (ydim, srcsize)}}
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
                       'd2': {'bus': 'coding/x_mu', 'shape': (ydim, numsteps)},
                       'd3': {'bus': 'coding/x_sig', 'shape': (ydim, numsteps)}},
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': timeseries},
                    {'input': 'd1', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': timeseries},
                    {'input': 'd2', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': histogram},
                ],
                [
                    {'input': 'd3', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': timeseries},
                    {'input': 'd3', 'ndslice': (slice(None), slice(None)), 'shape': (4, numsteps), 'plot': histogram},
                ],
            ]
        }
    }),
])
