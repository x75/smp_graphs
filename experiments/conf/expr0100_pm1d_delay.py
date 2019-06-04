"""expr0100 pointmass 1D with delay open loop
"""

from smp_graphs.utils_conf import get_systemblock

numsteps = 20
saveplot=True

robot1 = get_systemblock['pm'](dim_s0 = 1)
dim_s0 = robot1['params']['dims']['s0']['dim']
dim_m0 = robot1['params']['dims']['m0']['dim']
m_mins = np.array([robot1['params']['m_mins']]).T
m_maxs = np.array([robot1['params']['m_maxs']]).T

outputs = {'latex': {'type': 'latex'}}

expr_number = 13
expr_name = 'Experiment {0}: Lag'.format(expr_number)
desc = """In this experiment the action consists of uniform noise
sampled at intervals of {0} steps for a duration of {1} steps. The
robot has an inherent delay from motor input to sensor feedback of 2
timesteps. An agent does not know the timing parameters a priori for
all bodies, environments or tasks. The agent could be supplied with
all past and multimodal information but quite often, the relevant
variables are sparsely distributed within any contiguous submatrix of
SMT. Knowing the sites of relevant variables greatly increases the
speed of learning. In this case, the sensor reponse is linear in the
motor input so the temporal offset can easily be found with
cross-correlation methods.""".format(numsteps/4, numsteps)

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
            'saveplot': saveplot,
            'savetype': 'pdf',
            'title': expr_name,
            'desc': 'Illustration of the temporal offset',
            # 'desc': """Experiment 6.3.1-1 Illustration of the temporal
            # offset lag of a measurement \upsidedownhat{s} with respect
            # to the motor prediction \hat{s}, shown as s_0 and s_0^p,
            # respectively.""",
            'inputs': {
                's0': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's0p': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)},
            },
            'subplots': [
                [
                    {
                        'input': ['s0p', 's0'], 'plot': [partial(timeseries, marker = 'o')] * 2,
                        'title': None,
                        'cmap': ['rainbow'],
                        'xticks': tuple(range(0, numsteps+2, 2)),
                        'xlabel': 'time step [k]',
                        'ylim': (-1, 1),
                        'yticks': tuple((-1, -0.5, 0, 0.5, 1)),
                        'legend': {'prediction': 0, 'measurement': 1},
                        # 'yticks': [-1.0, -0.5, 0.0, 0.5, 1.0],
                    },
                ]
            ],
        },
    }),
])
