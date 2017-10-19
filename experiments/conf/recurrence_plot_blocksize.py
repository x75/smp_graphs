"""smp_graphs puppy recurrence plot conf
"""

from smp_base.plot import rp_timeseries_embedding

# reuse
numsteps = 500
blocksize = 100
xdim = 6
ydim = 4
debug = False

# graph
graph = OrderedDict([
    # puppy data
    ('puppydata', {
        'block': FileBlock2,
        'params': {
            'id': 'puppydata',
            'debug': False,
            'blocksize': blocksize,
            'type': 'puppy',
            'file': {'filename':
                'data/goodPickles/recording_eC0.35_eA0.05_c0.50_n1000_id0.pickle',
                'filetype': 'puppy',},
            'outputs': {'x': {'shape': (xdim, blocksize)}, 'y': {'shape': (ydim, blocksize)}},
        },
    }),
    
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': {'val': np.tile(np.random.uniform(-1, 1, (3, 1)), blocksize)}},
            'outputs': {'x': {'shape': (3, blocksize)}},
            'debug': False,
            'blocksize': blocksize,
        },
    }),
    
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'inputs': {'lo': {'val': 0, 'shape': (3, 1)}, 'hi': {'bus': 'b1/x', 'shape': (3, blocksize)}},
            'outputs': {'x': {'shape': (3, blocksize)}},
            'debug': False,
            'blocksize': blocksize,
        },
    }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'wspace': 0.2,
            'hspace': 0.2,
            'debug': False,
            # 'inputs': {
            #     'd1': {'bus': 'puppydata/x'},
            #     'd2': {'bus': 'puppydata/y'},
            #     'd3': {'bus': 'b1/x'},
            #     'd4': {'bus': 'b2/x'}},
            'inputs': {
                'd1': {'bus': 'puppydata/x', 'shape': (xdim, numsteps)},
                'd2': {'bus': 'puppydata/y', 'shape': (ydim, numsteps)},
                'd3': {'bus': 'b1/x', 'shape': (3, numsteps)},
                'd4': {'bus': 'b2/x', 'shape': (3, numsteps)}},
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
