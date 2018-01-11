"""expr0110 pointmass 1D with delay open loop xcorr scan
"""

from smp_graphs.utils_conf import get_systemblock
from smp_graphs.utils_conf_meas import make_input_matrix_ndim
from smp_graphs.block_meas import XCorrBlock2

numsteps = 1000

robot1 = get_systemblock['pm'](dim_s0 = 1)
robot1['params']['transfer'] = 4 # cosine
dim_s0 = robot1['params']['dims']['s0']['dim']
dim_m0 = robot1['params']['dims']['m0']['dim']
m_mins = np.array([robot1['params']['m_mins']]).T
m_maxs = np.array([robot1['params']['m_maxs']]).T

outputs = {'latex': {'type': 'latex'}}
desc = 'numsteps = {0}'.format(numsteps/4)

# scan parameters
scanstart = -10
scanstop = 0
scanlen = scanstop - scanstart

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
            'rate': 5,
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
                    },
                ],
            ],
        },
    }),
    
    # cross-correlation scan
    ('xcorr', {
        'block': XCorrBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'pre_l0/pre', 'shape': (dim_s0, numsteps)}, 'x': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)}},
            'shift': (scanstart, scanstop),
            'outputs': {'xcorr': {'shape': (dim_m0, dim_s0, scanlen)}},
        }
    }),
    
    # cross-correlation plot
    ('xcorr_plot', {
        'block': ImgPlotBlock2,
        'params': {
            'logging': False,
            'saveplot': saveplot,
            'debug': False,
            'blocksize': numsteps,
            # 'inputs': make_input_matrix(xdim = dim_m0, ydim = dim_s0, with_t = True),
            'inputs': make_input_matrix_ndim(xdim = dim_m0, ydim = dim_s0, with_t = True, scan = (scanstart, scanstop)),
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            'subplots': [
                # [{'input': ['d3'], 'ndslice': (i, j, ), 'xslice': (0, scanlen), 'xaxis': 't',
                #   'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(dim_m0)]
                # for i in range(dim_s0)],
                [
                    {
                        'input': ['d3'], 'ndslice': (slice(scanlen), i, j),
                        'shape': (1, scanlen), 'cmap': 'RdGy', 'title': 'xcorrs',
                        'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                        'xaxis': range(scanstart, scanstop),
                        'colorbar': True,
                    } for j in range(dim_m0)] # 'seismic'
            for i in range(dim_s0)],
        },
    }),
    
])
