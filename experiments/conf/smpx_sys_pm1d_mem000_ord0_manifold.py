
from smp_graphs.utils_conf import get_systemblock

randseed = 1
numsteps = 1000
loopblocksize = numsteps

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000,
    'lim': 1.0,
}
    
dim = lconf['dim']

systemblock = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = 1)

dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
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
sweepsys_input_flat = np.power(sweepsys_steps, dim_s_proprio)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s_proprio']['shape'] = (dim_s_proprio, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s_extero']['shape']  = (dim_s_extero, sweepsys_input_flat)

sweepmdl_steps = 1000
sweepmdl_input_flat = sweepmdl_steps # np.power(sweepmdl_steps, dim_s_proprio * 2)
sweepmdl_func = f_random_uniform

# sweepmdl_steps = 3
# sweepmdl_input_flat = np.power(sweepmdl_steps, dim_s_proprio * 2)
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
                'shape': (dim_s_proprio, sweepsys_input_flat),
                'buscopy': 'sweepsys_grid/meshgrid'
            }
        },
        'subgraph_rewrite_id': True,
        # subgraph
        # 'subgraph': 'conf/sweepsys_grid.py'
        'subgraph': OrderedDict([
        # 'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'numsteps': sweepsys_input_flat,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s_proprio)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
                    # 'func': f_meshgrid
                    'func': f_random_uniform,
                },
            }),
                
            # sys to sweep
            sweepsys,

        #     # ('sweepsys_grid', {
        #     #     'block': FuncBlock2,
        #     #     'params': {
        #     #         'debug': False,
        #     #         'blocksize': sweepsys_input_flat,
        #     #         'inputs': {
        #     #             'ranges': {'val': np.array([[-1, 1]] * dim_s_proprio)},
        #     #             'steps':  {'val': sweepsys_steps},
        #     #             },
        #     #         'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
        #     #         'func': f_meshgrid
        #     #         },
        #     #     }),
                
        #     #     # sys to sweep
        #     #     sweepsys,

             ]),
        }
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
            'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
            'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock,
            'subgraph_rewrite_id': True,
        },
    }),

    # meshgrid delayed
    ('motordel', {
        'block': DelayBlock2,
        'params': {
            'id': 'motordel',
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'sweepsys/meshgrid'}},
            'delays': {'y': 1},
            }
        }),
        
    # plot the system sweep result
    ('plot_sweep_1', {
        'block': PlotBlock2,
        'params': {
            'debug': False,
            'logging': False,
            'blocksize': numsteps, # sweepsys_input_flat,
            'title': 'system sweep',
            'inputs': {
                'meshgrid': {
                    # 'bus': 'sweepsys/meshgrid',
                    'bus': 'motordel/dy',
                    'shape': (dim_s_proprio, sweepsys_input_flat)},
                's_proprio': {
                    'bus': 'robot0/s_proprio',
                    'shape': (dim_s_proprio, sweepsys_input_flat)},
                's_extero': {
                    'bus': 'robot0/s_extero',
                    'shape': (dim_s_extero, sweepsys_input_flat)},
                },
                'hspace': 0.2,
                'subplots': [
                    [
                        {'input': ['meshgrid'], 'plot': timeseries},
                    ],
                    [
                        {'input': ['s_proprio'], 'plot': timeseries},
                    ],
                    [
                        {'input': ['s_extero'], 'plot': timeseries},
                    ],
                    ],
            }
        }),

    # sns matrix plot
    ('plot_sweep_2', {
        'block': SnsMatrixPlotBlock2,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': False,
            'saveplot': saveplot,
            'blocksize': numsteps,
            'inputs': {
                'meshgrid': {
                    #'bus': 'sweepsys_grid/meshgrid',
                    'bus': 'motordel/dy',
                    'shape': (dim_s_proprio, sweepsys_input_flat)},
                's_proprio': {
                    'bus': 'robot0/s_proprio',
                    'shape': (dim_s_proprio, sweepsys_input_flat)},
                's_extero': {
                    'bus': 'robot0/s_extero',
                    'shape': (dim_s_extero, sweepsys_input_flat)},
                },
            'outputs': {},#'x': {'shape': 3, 1)}},
            'subplots': [
                [
                    # stack inputs into one vector (stack, combine, concat
                    {'input': ['meshgrid', 's_proprio', 's_extero'], 'mode': 'stack',
                         'plot': histogramnd},
                ],
            ],
        },
    }),

        
    ])
