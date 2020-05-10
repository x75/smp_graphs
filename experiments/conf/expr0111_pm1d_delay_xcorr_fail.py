"""expr0110 pointmass 1D with delay open loop xcorr scan
"""

from smp_graphs.utils_conf import get_systemblock, get_subplots_expr0111, get_subplots_expr0111_mi
from smp_graphs.utils_conf_meas import make_input_matrix_ndim
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import MIBlock2

numsteps = 2000
# debug = True
saveplot = True

robot1 = get_systemblock['pm'](dim_s0 = 1)
robot1['params']['transfer'] = 4 # cosine
dim_s0 = robot1['params']['dims']['s0']['dim']
dim_m0 = robot1['params']['dims']['m0']['dim']
m_mins = np.array([robot1['params']['m_mins']]).T
m_maxs = np.array([robot1['params']['m_maxs']]).T

outputs = {'latex': {'type': 'latex'}}

expr_number = 15
expr_name = 'Experiment {0}: cross-correlation fails on nonlinear dependency'.format(expr_number)
desc = """When the previous experiment (expr0110) is repeated on a system
with a nonlinear functional relationship between motor and sensor
values, $s_0 = \\cos(m_0)$ for example, the cross-correlation method
fails because it can only capture linear relationships. The result can
be restored however by using the mutual information instead of
cross-correlation as the point-wise dependency measure.""".format(numsteps)

# scan parameters
scanstart = 0 # -10
scanstop = 11
scanlen = scanstop - scanstart

scan_plot_inputs = make_input_matrix_ndim(
    xdim=dim_m0, ydim = dim_s0,
    with_t = True,
    scan = (scanstart, scanstop))
scan_plot_inputs['d4'] = {'bus': 'xcorr/xcorr_abs', 'shape': (dim_s0, dim_m0, scanlen)}
scan_plot_inputs['d1'] = {'bus': 'mi_ll0_ll0/mi', 'shape': (dim_s0, dim_m0, scanlen)}

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
    ('plot0_meas_l0', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'savesize': (6, 3),
            'title': expr_name,
            'desc': """Timeseries with the motor values in
            blue and the sensor values in green. The
            motor-sensor relationship of this system, for example a
            joint angle controlled cartesian end-effector coordinate,
            still is systematic but not bijective anymore and the
            sensor responses lump together in the positive
            half-plane.""",
            'inputs': {
                's0': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's0p': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)},
            },
            'subplots': [
                [
                    {
                        'input': ['s0p', 's0'],
                        'plot': [partial(timeseries, marker = 'o')] * 2,
                        'title': None,
                        'xlabel': 'time step [k]',
                        'ylabel': 'normalized state',
                        'legend': {'prediction': 0, 'measurement': 1},
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
            'inputs': {
                'y': {'bus': 'pre_l0/pre', 'shape': (dim_s0, numsteps)},
                'x': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)}},
            'shift': (scanstart, scanstop),
            'outputs': {
                'xcorr': {'shape': (dim_m0, dim_s0, scanlen)},
                'xcorr_abs': {'shape': (dim_m0, dim_s0, scanlen)}
            },
        }
    }),
    
    # motor/sensor mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loop': [
                (
                    'inputs', {'y': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)}, 'x': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)}}
                ),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': True,
                    'inputs': {'x': {'bus': 'puppyslice/x_all'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': {'shape': ((dim_s0 + xdim)**2, 1)}}
                    # 'outputs': {'mi': {'shape': (scanlen * dim_s0 * dim_m0, 1)}}
                    'outputs': {'mi': {'shape': (dim_s0, dim_m0, scanlen)}}
                }
            },
        }
    }),
    
    # scan plot
    # ('plot1_scan', {
    #     'block': ImgPlotBlock2,
    #     'params': {
    #         # 'logging': False,
    #         'saveplot': saveplot,
    #         'savetype': 'pdf',
    #         'desc': """Results of a cross-correlation scan (top) and a
    #         mutual information scan (bottom). Normalized correlation
    #         coefficients take on values in the interval $[-1, 1]$. The
    #         normalized mutual information has a range of $[0,
    #         1]$. Cross-correlation is not able to pick up the
    #         systematic dependence of $s_0$ on $\\hat s_0$ indicated by
    #         values close to zero. Mutual information restores the
    #         qualitative picture from the linear case.""",
    #         'debug': False,
    #         'blocksize': numsteps,
    #         # 'inputs': make_input_matrix(xdim = dim_m0, ydim = dim_s0, with_t = True),
    #         'inputs': scan_plot_inputs,
    #         # 'outputs': {}, #'x': {'shape': (3, 1)}},
    #         'wspace': 0.5,
    #         'hspace': 0.5,
    #         'subplots': get_subplots_expr0111(dim_s0, dim_m0, scanlen, scanstart, scanstop, mode='img'),
    #     },
    # }),

    # scan plot
    ('plot1_scan_ts', {
        'block': PlotBlock2,
        'params': {
            # 'logging': False,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'savesize': (6, 3),
            'title': expr_name,
            'desc': """Cross-correlation (XC) vs. mutual information
            (MI) results of a cross-correlation scan (top) and a
            mutual information scan (bottom). Normalized correlation
            coefficients take on values in the interval $[-1, 1]$. The
            normalized mutual information has a range of $[0,
            1]$. Cross-correlation is not able to pick up the
            systematic dependence of measurement on prediction
            indicated by values close to zero. Mutual information
            restores the qualitative picture from the linear case.""",
            'debug': False,
            'blocksize': numsteps,
            # 'inputs': make_input_matrix(xdim = dim_m0, ydim = dim_s0, with_t = True),
            'inputs': scan_plot_inputs,
            # 'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            'subplots': get_subplots_expr0111(dim_s0, dim_m0, scanlen, scanstart, scanstop, mode='ts'),
        },
    }),

    # plot mi matrix as image
    ('plot2_mi', {
        'enable': False,
        'block': ImgPlotBlock2,
        'params': {
            # 'logging': False,
            'saveplot': saveplot,
            # 'debug': False,
            'wspace': 0.3,
            'hsapce': 0.3,
            'blocksize': numsteps,
            # 'inputs': {'d1': {'bus': 'mi_1/mi'}, 'd2': {'bus': 'infodist_1/infodist'}},
            'inputs': {'d1': {'bus': 'mi_ll0_ll0/mi', 'shape': (dim_s0, dim_m0, scanlen)}}, # .update(make_input_matrix_ndim(xdim = dim_m0, ydim = dim_s0, with_t = True, scan = (scanstart, scanstop))),
            # 'outputs': {}, #'x': {'shape': (3, 1)}},
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, (xdim + dim_s0)**2),
            #              'shape': (xdim+dim_s0, xdim+dim_s0), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, (xdim + dim_s0)**2),
            #              'shape': (xdim+dim_s0, xdim+dim_s0), 'plot': 'bla'},
            #     ],
            # ],
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, xdim * dim_s0),
            #              'shape': (dim_s0, xdim), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, xdim * dim_s0),
            #              'shape': (dim_s0, xdim), 'plot': 'bla'},
            #     ],
            # ],
            'subplots': [
                
                get_subplots_expr0111_mi(scanlen, dim_s0, dim_m0)
                # [
                #     {'input': 'd2',
                #      # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                #      # 'xslice': (0, 1),
                #      'ndslice': (i, slice(None), slice(None)),
                #      'title': 'te-matrix', 'cmap': 'Reds',
                #      'vaxis': 'rows',
                #      'shape': (dim_s0, dim_m0),
                #      'plot': 'bla'} for i in range(scanlen)
                # ],
                # [
                #     {'input': 'd3',
                #      # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                #      # 'xslice': (0, 1),
                #      'ndslice': (i, slice(None), slice(None)),
                #      'title': 'cte-matrix', 'cmap': 'Reds',
                #      'vaxis': 'rows',
                #      'shape': (dim_s0, dim_m0),
                #      'plot': 'bla'} for i in range(scanlen)
                # ],
            ],
        },
    }),
    
])
