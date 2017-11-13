"""smp_graphs puppy recurrence plot conf
"""

import random

from smp_base.plot import rp_timeseries_embedding

from smp_graphs.block_meas_essentia import EssentiaBlock2

# reuse
numsteps = 8192 # 65535
# xdim = 6
# ydim = 4
xdim = 1
ydim = 1
mfccdim = xdim * 12
# offset_range = 

lconf = {
    'file': {
        'filename': 'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
        'filetype': 'puppy',
        'offset': 300,
        'length': numsteps,
    },
}


# puppy pickle data homeokinesis episodes
filearray = [
    'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC1.12_eA1.66_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC0.24_eA0.77_c0.50_n1000_id0.pickle',
    ]

# puppy pickle data step response sweep
filearray = [
	'data/stepPickles/step_period_34_0.pickle',
	# 'data/stepPickles/step_period_16_0.pickle',
	# 'data/stepPickles/step_period_2_0.pickle',
	# 'data/stepPickles/step_period_64_0.pickle',
	# 'data/stepPickles/step_period_14_0.pickle',
	# 'data/stepPickles/step_period_20_0.pickle',
]

# audio files
filearray = [
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125/Burial - Beachfires.mp3', 'mp3'),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125/Autechre - spl47.mp3', 'mp3'),
]

looparray = [
    ('file', {
        'filename': fname,
        'filetype': ftype,
        'offset': random.randint(100000, 200000),
        'length': numsteps,
    }) for fname, ftype in filearray]

# # loopblock 1: puppy
# loopblock = {
#     'block': FileBlock2,
#     'params': {
#         'id': 'puppydata',
#         'debug': False,
#         'blocksize': numsteps,
#         'type': 'puppy',
#         'file': {
#             'filename': 'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
#             'filetype': 'puppy',
#             'offset': 300,
#             'length': numsteps,
#         },
#         'outputs': {
#             'x': {'shape': (xdim, numsteps)},
#             'y': {'shape': (ydim, numsteps)}
#         },
#         },
#     }

# loopblock 2: audio with essentia
loopblock_graph = OrderedDict([
    # file block
    ('filedata', {
        'block': FileBlock2,
        'params': {
            'id': 'filedata',
            'debug': False,
            'blocksize': numsteps,
            'logging': False,
            'type': 'puppy',
            # 'lconf': {},
            'file': {},
            'outputs': {
                'x': {'shape': (xdim, numsteps)},
                'y': {'shape': (ydim, numsteps)}
            },
        },
    }),
    # analysis block
    ('essentia1', {
        'block': EssentiaBlock2,
        'params': {
            'id': 'puppydata',
            'debug': False,
            'blocksize': numsteps,
            'samplerate': 1024,
            'inputs': {
                'x': {'bus': 'filedata/x'},
            },
            'outputs': {
                'centroid': {'shape': (xdim * 1, numsteps / 1024 + 1), 'etype': 'centroid'},
                'mfcc': {'shape': (mfccdim, numsteps / 1024 + 1), 'etype': 'mfcc', 'numberCoefficients': mfccdim},
                # 'sbic': {'shape': (xdim * 1, numsteps / 1024 + 1), 'etype': 'sbic'},
                # 'y': {'shape': (xdim, numsteps)},
                # 'y': {'shape': (ydim, numsteps)}
            },
        },
    }),
    
    # action block
    # output / render block
])

loopblock = {
    'block': Block2,
    'params': {
        'id': 'loopblock_graph',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': numsteps,
        'blocksize': 1,
        'blockphase': [0],
        # 'lconf': {},
        # 'outputs': {'jh': {'shape': (1,1)}},
        # 'outputs': {'jh': {'shape': (1, 1), 'buscopy': 'jh/jh'}},
        'subgraph_rewrite_id': True,
        # contains the subgraph specified in this config file
        'subgraph': loopblock_graph,
        'outputs': {
            'x': {'buscopy': 'filedata/x'},
            'y': {'buscopy': 'filedata/y'},
        },
    },
}

def make_puppy_rp_plot_inputs():
    global looparray, numsteps, xdim, ydim, mfccdim
    # buskey_base = ['puppyloop']
    # buskey_bases = ['filedata', 'essentia1']
    # buskey_signal = ['x', 'y']
    # buskey_bases = ['essentia1']
    # buskey_signal = [ 'y']
    buskey_bases = ['essentia1'] * 2
    buskey_signal = [ 'centroid', 'mfcc']
    buskey_shapes = [ (xdim, numsteps / 1024 + 1), (mfccdim, numsteps / 1024 + 1)]
    # inspec = {'d3': {'bus': 'b2/x', 'shape': (3, numsteps)}}
    inspec = {}
    for i,l in enumerate(looparray):
        # print "l", l
        for j, buskey_base in enumerate(buskey_bases):
            inspec['%s_ll%d_%d' % (buskey_signal[j], i, j, )] = {
                'bus': '%s_ll%d_ll0_ll0/%s' % (buskey_base, i, buskey_signal[j]),
                'name': '%s %s[%d:%d]' % (buskey_signal[j], l[1]['filename'].split('/')[-1], l[1]['offset'], l[1]['offset'] + l[1]['length']),
                'shape': buskey_shapes[j], # (xdim, numsteps / 1024 + 1),
            }
            # inspec['%s_ll%d_%d' % (buskey_signal[j], i, j, )] = {'bus': '%s_ll%d_ll0_ll0/%s' % (buskey_base, i, buskey_signal[j])}
            # inspec['y_ll%d_%d' % (i, j, )] = {'bus': '%s_ll%d_ll0_ll0/y' % (buskey_base, i, )}
            # inspec['x_e_ll%d' % (i, )] = {'bus': '%s_ll%d_ll0_ll0/x' % (buskey_base, i, )}
    return inspec

# graph
graph = OrderedDict([
    # puppy data
    ('puppyloop', {
        'block': LoopBlock2,
        'params': {
            'id': 'puppydata',
            'blocksize': numsteps,
            'loop': looparray,
            # 'loopblock': loopblock,
            'loopblock': loopblock_graph,
            'numsteps': numsteps,
            # 'outputs': {
            #     'x': {'shape': (xdim, numsteps)},
            #     'y': {'shape': (ydim, numsteps)},
            #     },
        },
    }),
             
    # # a constant
    # ("b1", {
    #     'block': ConstBlock2,
    #     'params': {
    #         'id': 'b1',
    #         'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 1))}},
    #         'outputs': {'x': {'shape': (3, 1)}},
    #         'debug': False,
    #     },
    # }),
    
    # # a random number generator, mapping const input to hi
    # ("b2", {
    #     'block': UniformRandomBlock2,
    #     'params': {
    #         'id': 'b2',
    #         # 'lo': 0,
    #         # 'hi': 1,
    #         'inputs': {'lo': {'val': 0}, 'hi': {'bus': 'b1/x'}},
    #         'outputs': {'x': {'shape': (3, 1)}},
    #         'debug': False,
    #     },
    # }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'debug': False,
            'wspace': 0.4, 'hspace': 0.4,
            'inputs': make_puppy_rp_plot_inputs(),
            'ylim_share': False,
            'subplots': [[
                # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': timeseries},
                # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': histogram},
                # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': timeseries},
                # {'input': '%s' % (itup[0], ), 'plot': timeseries},
                {
                    'input': '%s' % (itup[0], ),
                    'plot': partial(timeseries, marker = 'o'),
                    'title': itup[1]['name'],
                    'shape': itup[1]['shape'],
                },
                # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': rp_timeseries_embedding}
                # , 'y'
                # ] for itup in zip(map(lambda x: x/2, range(len(looparray)*2)), ['x'] * len(looparray))]
                ] for itup in make_puppy_rp_plot_inputs().items()]
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
