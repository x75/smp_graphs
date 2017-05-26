"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reuse
numsteps = 1

# graph
graph = OrderedDict([
    # a constant
    ("selflog", {
        'block': FileBlock2,
        'params': {
            'id': 'selflog',
            'logging': False,
            'inputs': {},
            'debug': False,
            'blocksize': numsteps,
            'type': 'selflogconf',
            # this is looping demand
            'file': {
                'filename': 'data/experiment_20170509_131125_puppy_rp_blocksize_pd.h5',
                # 'data/experiment_20170509_130752_puppy_rp_blocksize_pd.h5',
                # 'data/experiment_20170507_154742_pd.h5',
                # 'data/experiment_20170505_110833_pd.h5' # default2_loop 1000 steps
                # 'data/experiment_20170504_192821_pd.h5',
                # 'data/experiment_20170504_202016_pd.h5',
                },
            'outputs': {'conf': {'shape': (1,1)}, 'conf_final': {'shape': (1,1)}},
            # 'outputs': {'x': [None], 'y': [None]},
        },
    }),
])
