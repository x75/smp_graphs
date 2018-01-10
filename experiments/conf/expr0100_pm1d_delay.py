"""expr0100 pointmass 1D with delay open loop
"""

from smp_graphs.utils_conf import get_systemblock

numsteps = 20

robot1 = get_systemblock['pm'](dim_s0 = 1)
dim_s0 = robot1['params']['dims']['s0']['dim']
dim_m0 = robot1['params']['dims']['m0']['dim']
m_mins = np.array([robot1['params']['m_mins']]).T
m_maxs = np.array([robot1['params']['m_maxs']]).T

outputs = {'latex': {'type': 'latex'}}

graph = OrderedDict([
    # point mass system
    ('robot1', robot1),
    
    # noise
    ('pre_l0', {
        'block': ModelBlock2,
        'params': {
            'blocksize': 1,
            'blockphase': [0],
            'inputs': {                        
                'lo': {'val': m_mins, 'shape': (dim_s0, 1)},
                'hi': {'val': m_maxs, 'shape': (dim_s0, 1)},
                },
            'outputs': {'pre': {'shape': (dim_m0, 1)}},
            'models': {
                'goal': {'type': 'random_uniform'}
                },
            'rate': numsteps/4,
            },
        }),
    
    # measurement
    ('meas_l0', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'saveplot': saveplot, 'savetype': 'pdf',
            'inputs': {
                's0': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's0p': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)},
            },
            'subplots': [
                [
                    {
                        'input': ['s0p', 's0'], 'plot': [partial(timeseries, marker = 'o')] * 2,
                        'xticks': range(0, numsteps, 2),
                    },
                ]
            ],
        },
    }),
])
