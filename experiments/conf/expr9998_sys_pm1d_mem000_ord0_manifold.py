"""smp_graphs config

sensorimotor manifold of n-dimensional point mass
"""
from smp_graphs.utils_conf import get_systemblock

randseed = 2
numsteps = 1000
loopblocksize = numsteps
saveplot = not True

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 2,
    'budget': 1000,
    'lim': 1.0,
    'order': 0,
}
    
dim = lconf['dim']
lag = lconf['lag']

systemblock = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = lag)

dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
dim_s0 = dim_s_proprio
dim_s1 = dim_s_extero
m_mins = np.array([systemblock['params']['m_mins']]).T
m_maxs = np.array([systemblock['params']['m_maxs']]).T

lag = systemblock['params']['lag']
lag_past = systemblock['params']['lag_past']
lag_future = systemblock['params']['lag_future']

dt = systemblock['params']['dt']



"""
sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepsys_steps = 1000 # 6
sweepsys_input_flat = np.power(sweepsys_steps, dim_s0)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s0']['shape'] = (dim_s0, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s1']['shape']  = (dim_s1, sweepsys_input_flat)

sweepmdl_steps = 1000
sweepmdl_input_flat = sweepmdl_steps # np.power(sweepmdl_steps, dim_s0 * 2)
sweepmdl_func = f_random_uniform

# sweepmdl_steps = 3
# sweepmdl_input_flat = np.power(sweepmdl_steps, dim_s0 * 2)
# sweepmdl_func = f_meshgrid

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': sweepsys_input_flat,  # inner numsteps when used as loopblock (sideways time)
        'blocksize': 1, # compute single steps
        'blockphase': [0],        # phase = 0
        'outputs': {
            'meshgrid': {
                'shape': (dim_s0, sweepsys_input_flat),
                'buscopy': 'sweepsys_grid/meshgrid',
            },
        },
        'subgraph_rewrite_id': True,
        
        # # subgraph file
        # 'subgraph': 'conf/sweepsys_grid.py'

        # subgraph dict
        'subgraph': OrderedDict([
        # 'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'numsteps': sweepsys_input_flat,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s0)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
                    # 'func': f_meshgrid
                    'func': f_random_uniform,
                },
            }),
            
            # sys to sweep
            sweepsys,
            
            # meshgrid delayed
            ('motordel', {
                'block': DelayBlock2,
                'params': {
                    'id': 'motordel',
                    'blocksize': sweepsys_input_flat, # 1,
                    'flat': False,
                    'inputs': {
                        # 'y0': {'bus': 'sweepsys_grid/meshgrid', 'delay': [0], },
                        'y': {'bus': 'sweepsys_grid/meshgrid', 'delay': [0, 1, 2], },
                        's00': {'bus': 'robot0/s0', 'delay': 0, },
                        's01': {'bus': 'robot0/s0', 'delay': 1, },
                        },
                    }
                }),
        
            # velocity difference
            ('sdiff', {
                'block': dBlock2,
                'params': {
                    'id': 'sdiff',
                    'blocksize': sweepsys_input_flat, # 1,
                    'inputs': {
                        's1': {'bus': 'robot0/s1', },
                        },
                    'd': 1/dt,
                    # 'leak': 0.01,
                    }
                }),
        ]),
        
    },
}


graph = OrderedDict([
    # sweep system
    ("sweepsys", {
        'debug': False,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'sweepsys',
            'logging': False,
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # execution cycle, same as global numsteps
            'blockphase': [1],     # execute on first time step only
            'numsteps':  numsteps,          # numsteps      / loopblocksize = looplength
            'loopblocksize': loopblocksize, # loopblocksize * looplength    = numsteps
            # can't do this dynamically yet without changing init passes
            'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
            'loop': [('none', {}) for i in range(3)], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock,
            'subgraph_rewrite_id': False,
        },
    }),

    # plot the system sweep result
    ('plot_sweep_1', {
        'block': PlotBlock2,
        'params': {
            'debug': False,
            'blocksize': numsteps, # sweepsys_input_flat,
            'title': 'system sweep',
            'saveplot': saveplot,
            'logging': False,
            'inputs_log': None,
            'inputs': {
                # 'meshgrid_d0': {
                #     # 'bus': 'sweepsys/meshgrid',
                #     'bus': 'motordel_ll0/dy0',
                #     'shape': (dim_s0, 3, sweepsys_input_flat)},
                'meshgrid_d1': {
                    'bus': 'motordel_ll0/dy',
                    'shape': (dim_s0, 3, sweepsys_input_flat)},
                's00': {
                    'bus': 'motordel_ll0/ds00',
                    'shape': (dim_s0, sweepsys_input_flat)},
                's01': {
                    'bus': 'motordel_ll0/ds01',
                    'shape': (dim_s0, sweepsys_input_flat)},
                's1': {
                    'bus': 'robot0_ll0/s1',
                    'shape': (dim_s1, sweepsys_input_flat)},
                'ds1': {
                    'bus': 'sdiff_ll0/ds1',
                    'shape': (dim_s1, sweepsys_input_flat)},
                },
                'hspace': 0.2,
                'subplots': [
                    [
                        # {'input': ['meshgrid_d0', 's00'],
                        #      'shape': (dim_s0, sweepsys_input_flat),
                        # 'plot': timeseries},
                        # {'input': ['meshgrid_d1',],
                        #      'ndslice': (slice(None), slice(None), slice(None)),
                        #      'shape': (dim_s0 * 3, sweepsys_input_flat),
                        {
                            'input': [
                                'meshgrid_d1', 's00'],
                            'ndslice': [
                                (slice(None), slice(None), slice(None)),
                                (slice(None), slice(None)),],
                            'shape': [
                                (dim_s0 * 3, sweepsys_input_flat),
                                (dim_s0 * 1, sweepsys_input_flat),],
                            'plot': timeseries
                        },
                    ],
                    # [
                    #     {'input': ['s00'], 'plot': timeseries},
                    # ],
                    # [
                    #     {'input': ['s00', 'ds1'], 'plot': timeseries},
                    # ],
                    # [
                    #     {'input': ['s1'], 'plot': timeseries},
                    # ],
                    ],
            }
        }),

    # sns matrix plot
    ('plot_sweep_2', {
        'enable': 1,
        'block': SnsMatrixPlotBlock2,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': False,
            'saveplot': saveplot,
            'blocksize': numsteps,
            'inputs': {
                # 'meshgrid_d0': {
                #     #'bus': 'sweepsys_grid/meshgrid',
                #     'bus': 'motordel_ll0/dy0',
                #     'shape': (dim_s0, 3, sweepsys_input_flat)},
                'meshgrid_d1': {
                    'bus': 'motordel_ll0/dy',
                    'shape': (dim_s0, 3, sweepsys_input_flat)},
                's00': {
                    'bus': 'motordel_ll0/ds00',
                    'shape': (dim_s0, sweepsys_input_flat)},
                's01': {
                    'bus': 'motordel_ll0/ds01',
                    'shape': (dim_s0, sweepsys_input_flat)},
                's1': {
                    'bus': 'robot0_ll0/s1',
                    'shape': (dim_s1, sweepsys_input_flat)},
                'ds1': {
                    'bus': 'sdiff_ll0/ds1',
                    'shape': (dim_s1, sweepsys_input_flat)},
                },
            'outputs': {},#'x': {'shape': 3, 1)}},
            'subplots': [
                [
                    # stack inputs into one vector (stack, combine, concat)
                    # {'input': ['meshgrid_d0', 'meshgrid_d1', 's00', 's01', 's1', 'ds1'], 'mode': 'stack',
                    #      'plot': 'hist2d'},
                    {
                        'input': ['meshgrid_d1', 's00'], 'mode': 'stack',
                         'ndslice': [
                             (slice(None), slice(None), slice(None)),
                             (slice(None), slice(None)),
                             ],
                         'shape': [
                             (dim_s0 * 3, sweepsys_input_flat),
                             (dim_s0 * 1, sweepsys_input_flat),
                             ],
                    'plot': 'hist2d'},
                ],
            ],
        },
    }),

        
    ])
