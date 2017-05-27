"""smp_graphs puppy recurrence plot conf
"""

import random

# reuse
numsteps = 500
xdim = 6
ydim = 4

filearray = [
    'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC1.12_eA1.66_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC0.24_eA0.77_c0.50_n1000_id0.pickle',
    ]
    
filearray = [
	'data/stepPickles/step_period_34_0.pickle',
	'data/stepPickles/step_period_16_0.pickle',
	'data/stepPickles/step_period_2_0.pickle',
	'data/stepPickles/step_period_64_0.pickle',
	'data/stepPickles/step_period_14_0.pickle',
	'data/stepPickles/step_period_20_0.pickle',
]

looparray = [('file', {'filename': fname, 'filetype': 'puppy', 'offset': random.randint(0, 500), 'length': numsteps}) for fname in filearray]

loopblock = {
    'block': FileBlock2,
    'params': {
        'id': 'puppydata',
        'debug': False,
        'blocksize': numsteps,
        'type': 'puppy',
        'file': {'filename': 'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle', 'filetype': 'puppy', 'offset': 300, 'length': numsteps},
        'outputs': {'x': {'shape': (xdim, numsteps)}, 'y': {'shape': (ydim, numsteps)}},
        },
    }

def make_puppy_rb_plot_inputs():
    global looparray, numsteps
    inspec = {'d3': {'bus': 'b2/x', 'shape': (3, numsteps)}}
    for i,l in enumerate(looparray):
        inspec['x_%d' % (i, )] = {'bus': 'puppyloop_%d/x' % (i, )}
        inspec['y_%d' % (i, )] = {'bus': 'puppyloop_%d/y' % (i, )}
    return inspec

# graph
graph = OrderedDict([
    # puppy data
    ('puppyloop', {
        'block': LoopBlock2,
        'params': {
            'id': 'puppydata',
            'loop': looparray,
            'loopblock': loopblock,
        },
    }),
             
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 1))}},
            'outputs': {'x': {'shape': (3, 1)}},
            'debug': False,
        },
    }),
    
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            # 'lo': 0,
            # 'hi': 1,
            'inputs': {'lo': {'val': 0}, 'hi': {'bus': 'b1/x'}},
            'outputs': {'x': {'shape': (3, 1)}},
            'debug': False,
        },
    }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'debug': False,
            'wspace': 0.4, 'hspace': 0.4,
            'inputs': make_puppy_rb_plot_inputs(),
            'subplots': [[{'input': '%s_%d' % (itup[1], itup[0]), 'plot': timeseries}, {'input': '%s_%d' % (itup[1], itup[0]), 'plot': histogram}, {'input': '%s_%d' % (itup[1], itup[0]), 'plot': rp_timeseries_embedding}] for itup in zip(map(lambda x: x/2, range(len(looparray)*2)), ['x', 'y'] * len(looparray))]

            # [
            #     [
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': timeseries},
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': histogram},
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': rp_timeseries_embedding}]
            #     for ink in zip(range(3)]
                # [[i for i in range(3)]]
                # + [
                #     [
                #     {'input': 'd3', 'slice': (0, 3), 'plot': timeseries},
                #     {'input': 'd3', 'slice': (0, 3), 'plot': histogram},
                #     {'input': 'd3', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
                # ]]
                # [
                #     {'input': 'd1', 'slice': (0, 6), 'plot': timeseries},
                #     {'input': 'd1', 'slice': (0, 6), 'plot': histogram},
                #     {'input': 'd1', 'slice': (0, 6), 'plot': rp_timeseries_embedding},
                # ],
                # [
                #     {'input': 'd2', 'slice': (0, 4), 'plot': timeseries},
                #     {'input': 'd2', 'slice': (0, 4), 'plot': histogram},
                #     {'input': 'd2', 'slice': (0, 4), 'plot': rp_timeseries_embedding},
                # ],
                # ,
            #]
        }
    })
])
