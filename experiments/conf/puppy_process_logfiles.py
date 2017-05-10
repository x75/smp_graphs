"""smp_graphs process puppy logfiles

loop over puppy pickles and concatenate all x,y data

"""

from smp_graphs.block import SeqLoopBlock2

# reuse
looparray = [
    ('file', ['data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle']),
    ('file', ['data/pickles_puppy_03_22_14U/recording_eC0.00_eA0.26_c0.50_n1000_id0.pickle']),
]

f = open('data/goodPickles/allpickles.txt')
looparray = [('file', [fname.rstrip()]) for fname in f.readlines()]
f.close()

loopblocksize = 1000
numsteps = len(looparray) * loopblocksize
debug = True

loopblock = {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'logging': False,
            'inputs': {},
            'debug': True,
            'blocksize': loopblocksize,
            'type': 'puppy',
            # 'inputs': {'file': [
            #     'data/experiment_20170509_131125_puppy_rp_blocksize_pd.h5',
            #     ]},
            # this is looping demand
            'file': [
                'data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle',
                'data/pickles_puppy_03_22_14U/recording_eC0.00_eA0.26_c0.50_n1000_id0.pickle',
                # 'data/experiment_20170509_131125_puppy_rp_blocksize_pd.h5',
                # 'data/experiment_20170507_154742_pd.h5',
                # 'data/experiment_20170505_111138_pd.h5', # puppy_rp, 500 steps
                # 'data/experiment_20170505_110833_pd.h5' # default2_loop 1000 steps
                # 'data/experiment_20170505_084006_pd.h5'
                # 'data/experiment_20170505_083801_pd.h5',
                # 'data/experiment_20170505_003754_pd.h5',
                # 'data/experiment_20170505_001511_pd.h5',
                # 'data/experiment_20170505_001143_pd.h5',
                # 'data/experiment_20170505_000540_pd.h5',
                # 'data/experiment_20170504_192821_pd.h5',
                # 'data/experiment_20170504_202016_pd.h5',
                # 'data/experiment_20170504_222828_pd.h5',
                ],
            # 'outputs': {'conf': [(1,1)], 'conf_final': [(1,1)]},
            # 'outputs': {'log': [None]},
            'outputs': {'x': [None], 'y': [None]},
        },
    }

# graph
graph = OrderedDict([
    # # a constant
    ("puppylog", {
        'block': SeqLoopBlock2,
        'params': {
            'id': 'puppylog',
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps,
            'numsteps': numsteps, # same as loop length
            'loopblocksize': loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {'x': [(6, numsteps * 1000)], 'y': [(4, numsteps * 1000)]},
            # 'loop': [('inputs', {
            #     'lo': [np.random.uniform(-i, 0, (3, 1))], 'hi': [np.random.uniform(0.1, i, (3, 1))]}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            'loop': looparray,
            # 'loop': partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),
    
    # ("puppylog", loopblock),
    
    ('plotter', {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'plotter',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                # 'd1': ['puppylog/b1/x'],
                # 'd2': ['puppylog/b2/x'],
                'd3': ['puppylog/x'],
                'd4': ['puppylog/y'],
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                # [
                #     {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                #     # {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                # ],
                # [
                #     {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                #     # {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
                [
                    {'input': 'd3', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd3', 'slice': (3, 6), 'plot': histogram},
                ],
                [
                    {'input': 'd4', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd4', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        },
    }),
])
