"""smp_graphs perform windowed short time mutual info scan"""

from smp_graphs.block_meas_infth import MIMVBlock2, JHBlock2, TEMVBlock2
from smp_graphs.block import SliceBlock2, SeqLoopBlock2
from smp_graphs.block_plot import ImgPlotBlock2

saveplot = True
recurrent = True

lpzbarrelcnf = {
    'numsteps': 1000,
    # 'logfile': 'data/experiment_20170626_120004_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115924_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115813_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115719_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115628_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115540_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115457_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115406_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115323_actinf_m1_goal_error_ND_pd.h5', # 250
    # learning static target
    # 'logfile': 'data/experiment_20170626_124247_actinf_m1_goal_error_ND_pd.h5', # 500
    # just stepping
    'logfile': 'data/experiment_20170626_125226_actinf_m1_goal_error_ND_pd.h5', # 500
    'logfile': 'data/experiment_20170626_140407_actinf_m1_goal_error_ND_pd.h5', # 1000
    'xdim': 2,
    'xdim_eff': 2,
    'ydim': 2,
    'logtype': 'selflog',
    'sys_slicespec': {'x': {'gyr': slice(0, 2)}}    
}
    
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
    'numsteps': 2000,
    'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    # 'numsteps': 20000,
    # 'logfile': 'data/experiment_20170530_174612_process_logfiles_pd.h5', # step fast-to-slow newB all concatenated
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

ppycnf2 = {
    'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle', # continuous sweep without battery
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'numsteps': 5000,
}

cnf = lpzbarrelcnf

# copy params to namespace
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if 'sys_slicespec' in cnf:
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = -30
scanstop = 0
scanlen = scanstop - scanstart

# 1000/1000
winsize = 1000
overlap = 1000
# winsize = 50
# overlap = 50
srcsize = overlap


loopblocksize = numsteps

# data_x_key = 'x'
# data_y_key = 'y'
data_x_key = 's_proprio'
data_y_key = 'pre'

loopblock = {
        'block': Block2,
        'params': {
            'id': 'bhier',
            'debug': False,
            'topblock': False,
            'logging': False,
            'numsteps': numsteps,
            'blocksize': 1,
            'blockphase': [0],
            # 'outputs': {'jh': {'shape': (1,1)}},
            'outputs': {'jh': {'shape': (1, 1), 'buscopy': 'jh/jh'}},
            # contains the subgraph specified in this config file
            'graph': OrderedDict([
                ('ldata', {
                    'block': FileBlock2,
                    'params': {
                        'debug': True,
                        'blocksize': numsteps,
                        'type': cnf['logtype'],
                        'file': {'filename': cnf['logfile']},
                        'outputs': {
                            'log': None,
                            data_x_key: {'shape': (xdim, numsteps), 'storekey': '/robot1/s_proprio'},
                            data_y_key: {'shape': (ydim, numsteps), 'storekey': '/pre_l0/pre'}},
                    },
                }),
                
                # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
                ('ldataslice', {
                    'block': SliceBlock2,
                    'params': {
                        'blocksize': srcsize,
                        'inputs': {'x': {'bus': 'ldata/%s' % data_x_key, 'shape': (xdim, numsteps)}},
                        'slices': sys_slicespec,
                    }
                }),
                
                # joint entropy
                ('jh', {
                    'block': JHBlock2,
                    'params': {
                        'id': 'jh',
                        'blocksize': numsteps,
                        'debug': False,
                        'logging': False,
                        'inputs': {'x': {'bus': 'ldataslice/x_gyr'}, 'y': {'bus': 'ldata/%s' % data_y_key}},
                        'shift': (0, 1),
                        'outputs': {'jh': {'shape': (1, 1)}}
                    }
                })
            ]),
        }
    }

graph = OrderedDict([
    # a loop block calling the enclosed block len(loop) times,
    # returning data of looplength in one outer step
    ("jhloop", {
        'debug': True,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'jhloop',
            # loop specification, check hierarchical block to completely pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'blockphase': [1],
            'numsteps':  numsteps,
            'loopblocksize': loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {'jh': {'shape': (1, 1)}},
            # 'loop': [('inputs', {
            #     'lo': {'val': np.random.uniform(-i, 0, (3, 1)), 'shape': (3, 1)}, 'hi': {'val': np.random.uniform(0.1, i, (3, 1)), 'shape': (3, 1)}}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            # 'loop': [('inputs', {'x': {'val': np.random.uniform(np.pi/2, 3*np.pi/2, (3,1))]}) for i in range(1, numsteps+1)],
            # 'loop': partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
            'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),
    
    # ('data', {
    #     'block': FileBlock2,
    #     'params': {
    #         'id': 'data',
    #         'debug': False,
    #         # 'blocksize': overlap, # numsteps,
    #         'blocksize': srcsize, # numsteps,
    #         'type': cnf['logtype'],
    #         'file': {'filename': cnf['logfile']},
    #         'outputs': {'log': None, 'x': {'shape': (xdim, srcsize)},
    #                         'y': {'shape': (ydim, srcsize)}},
    #     },
    # }),

    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('dataslice', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'dataslice',
    #         # 'blocksize': overlap,
    #         'blocksize': srcsize,
    #         'debug': True,
    #         # puppy sensors
    #         'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, srcsize)}},
    #         'slices': sys_slicespec,
    #         # 'slices': ,
    #         }
    #     }),
        
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('mimvl', {
        'block': LoopBlock2,
        'params': {
            'id': 'mimvl',
            'blocksize': overlap,
            'debug': False,
            'loop': [('inputs', {'x': {'bus': 'ldataslice/x_acc', 'shape': (xdim_eff, winsize)},
                                 'y': {'bus': 'ldata/%s' % data_y_key, 'shape': (ydim, winsize)},
                                 'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
                                 # 'norm': {'val': np.array([[7.0]]), 'shape': (1, 1)},
                                 }),
                          # ('inputs', {'x': {'bus': 'dataslice/x_gyr'}, 'y': {'bus': 'data/y'}}),
                     # ('inputs', {'x': {'bus': 'data/x'}, 'y': {'bus': 'data/r'}}),
                     # ('inputs', {'x': {'bus': 'data/y'}, 'y': {'bus': 'data/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIMVBlock2,
                'params': {
                    'id': 'mimv',
                    'blocksize': overlap,
                    'debug': False,
                    'inputs': {'x': {'bus': 'ldataslice/x_gyr',
                                         'shape': (xdim_eff, winsize)},
                                   'y': {'bus': 'ldata/%s' % data_y_key,
                                             'shape': (ydim, winsize)}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'mimv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),

    ('mimv', {
        'block': MIMVBlock2,
        'params': {
            'id': 'mimv',
            'blocksize': overlap,
            'debug': False,
            'inputs': {'x': {'bus': 'ldataslice/x_gyr', 'shape': (xdim_eff, winsize)},
                           'y': {'bus': 'ldata/%s' % data_y_key, 'shape': (ydim, winsize)},
                                 'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
                           },
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
            'outputs': {'mimv': {'shape': (1, scanlen)}}
        }
    }),
    
    ('temv', {
        'block': TEMVBlock2,
        'params': {
            'id': 'temv',
            'blocksize': overlap,
            'debug': False,
            'inputs': {'x': {'bus': 'ldataslice/x_gyr', 'shape': (xdim_eff, winsize)},
                           'y': {'bus': 'ldata/%s' % (data_y_key, ), 'shape': (ydim, winsize)},
                                 'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
                           },
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            'outputs': {'temv': {'shape': (1, scanlen)},}
        }
    }),

    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("plot_ts", {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.2, 'hspace': 0.2,
            'inputs': {'d1': {'bus': 'ldata/%s' % (data_x_key, ), 'shape': (xdim, numsteps)}, 'd2': {'bus': 'ldata/%s' % (data_y_key, ), 'shape': (ydim, numsteps)}},
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                    {'input': 'd1', 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'plot': timeseries},
                    {'input': 'd2', 'plot': histogram},
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
            'debug': False,
            'wspace': 0.5,
            'hspace': 0.5,
            'blocksize': overlap, # numsteps,
            'inputs': {
                'd1': {'bus': 'mimv/mimv', 'shape': (1, scanlen * numsteps/overlap)},
                'd2': {'bus': 'mimvl|0/mimv', 'shape': (1, scanlen * numsteps/overlap)},
                'd3': {'bus': 'temv/temv', 'shape': (1, scanlen * numsteps/overlap)},                
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},},
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {'input': 'd1', 
                         'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/overlap, scanlen)},
                    
                    {'input': 'd2', 
                    'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/overlap, scanlen)},
                    {
                    'input': 'd3',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [2, 1], 'y': [0]},
                    'shape': (numsteps/overlap, scanlen),
                    'cmap': 'Reds'},
                ],
            ]
        },
    }),
    
    
])
