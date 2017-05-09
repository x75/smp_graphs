"""smp_graphs puppy recurrence plot conf
"""

# reuse
numsteps = 500
blocksize = 100
debug = False
# graph
graph = OrderedDict([
    # puppy data
    ('puppydata', {
        'block': FileBlock2,
        'params': {
            'id': 'puppydata',
            'idim': None,
            'odim': 'auto',
            'debug': False,
            'blocksize': blocksize,
            'file': [
                'data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle',
            ],
            'outputs': {'x': [None], 'y': [None]},
        },
    }),
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
            'outputs': {'x': [(3, 1)]},
            'debug': False,
            'blocksize': blocksize,
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'idim': 6,
            'odim': 3,
            # 'lo': 0,
            # 'hi': 1,
            'inputs': {'lo': [0], 'hi': ['b1/x']},
            'outputs': {'x': [(3, 1)]},
            'debug': False,
            'blocksize': blocksize,
        },
    }),
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 4,
            'odim': 1,
            'debug': False,
            'inputs': {
                'd1': ['puppydata/x'],
                'd2': ['puppydata/y'],
                'd3': ['b1/x'],
                'd4': ['b2/x']},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 6), 'plot': timeseries},
                    {'input': 'd1', 'slice': (0, 6), 'plot': histogram},
                    {'input': 'd1', 'slice': (0, 6), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'd2', 'slice': (0, 4), 'plot': timeseries},
                    {'input': 'd2', 'slice': (0, 4), 'plot': histogram},
                    {'input': 'd2', 'slice': (0, 4), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'd3', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd3', 'slice': (0, 3), 'plot': histogram},
                    {'input': 'd3', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'd4', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd4', 'slice': (0, 3), 'plot': histogram},
                    {'input': 'd4', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
                ],
            ]
        }
    })
])
