"""expr0120 pointmass 1D with delay open loop xcorr scan
"""

from smp_graphs.utils_conf import get_systemblock
from smp_graphs.utils_conf_meas import make_input_matrix_ndim
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import MIBlock2

numsteps = 2000
debug = True

robot1 = get_systemblock['pm'](dim_s0 = 1, dim_s1 = 1, lag = 2)
robot1['params']['transfer'] = 4 # cosine
robot1['params']['dims']['s1']['dissipation'] = 0.2

print(("robot1 dims", robot1['params']['dims']))
dim_s0 = robot1['params']['dims']['s0']['dim']
dim_m0 = robot1['params']['dims']['m0']['dim']
dim_s1 = robot1['params']['dims']['s1']['dim']
m_mins = np.array([robot1['params']['m_mins']]).T
m_maxs = np.array([robot1['params']['m_maxs']]).T

outputs = {'latex': {'type': 'latex'}}

desc = """The system used in the experiment expr0111 is extended
further by an additional \\emph{{order}}. This means, the dimension of
the primary motor variable at order 0 is kept the same but an
additional variable is introduced into the system state at order 1,
which is computed by integrating the order 0 variable. A simple
interpretation is the relation of acceleration and velocity. Also, the
nonlinear functional relationship between motor and sensor values at
order 0 is being kept. The coupling between action and effect is set
to a lag of two time steps. The raw timeseries of the full system
state is shown in
\\autoref{{fig:smp-expr0120-pm1d-ar1d-scan-example-plot0_meas_l0}}
with the motor and proprioceptive signal in the top panel (order 0),
and the velocity (order 1) in the bottom panel. The velocity is
computed by integrating the acceleration with a dissipative term
modelling friction. Thus, the velocity cannot grow without bounds and
saturates close to a value of 0.55. The scan results are shown in
\\autoref{{fig:smp-expr0120-pm1d-ar1d-scan-example-plot_xcorr}}. Four
pairwise scans are performed in total one the pairs $(m_0, s_1), (s_1,
s_1)$ using the cross-correlation and the mutual information measures.
The first pair is the motor signal (order 0) and the velocity sensor
(order 1), the second one is the self-pair of the velocity sensor. The
system is designed so that the information in the velocity is
determined both by a cross-modal action and an intrinsic memory. The
memory is caused by inertia in this case. Cross-correlation fails
again to detect the nonlinear and integral relationship between action
and velocity, which mutual information is able to capture. The scan is
performed over a range of 100 timesteps and the results show that
temporal dependencies are close to the current time step and compactly
distributed. For the given window size, the dependency measure over
time converges, indicated by values close to zero for all measures
starting from ten time steps into the past.""".format(numsteps)

# scan parameters
scanstart = 0 # -100
scanstop = 100
scanlen = scanstop - scanstart

scan_plot_inputs = {'d1': {'bus': 'mi_ll0_ll0/mi', 'shape': (dim_s1, dim_m0, scanlen)}}
scan_plot_inputs['d2'] = {'bus': 'mi_ll1_ll0/mi', 'shape': (dim_s1, dim_s1, scanlen)}
scan_plot_inputs['m2s'] = make_input_matrix_ndim(id = 'xcorr_m2s', xdim = dim_m0, ydim = dim_s0, with_t = True, scan = (scanstart, scanstop))['d3']
scan_plot_inputs['s2s'] = make_input_matrix_ndim(id = 'xcorr_s2s', xdim = dim_m0, ydim = dim_s0, with_t = True, scan = (scanstart, scanstop))['d3']

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

    # plot measurement: timeseries
    ('plot0_meas_l0', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'saveplot': saveplot, 'savetype': 'pdf',
            'desc': """Timeseries of the motor values $\\hat s_0$ in
                blue and the sensor values $s_0$ in green in the top
                panel. The bottom panel contains the graph of the
                first order state variable, the velocity. The
                dissipative term of the velocity (e.g. friction) keeps
                the velocity within bounds while it is still dominated
                by the remaining inertia. The dissipation parameter is
                set to
                {{{0}}}.""".format(robot1['params']['dims']['s1']['dissipation']),
            'inputs': {
                's0': {'bus': 'robot1/s0', 'shape': (dim_s0, numsteps)},
                's0p': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)},
                's1': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)},
            },
            'subplots': [
                [
                    {
                        'input': ['s0p', 's0'],
                        'plot': [partial(timeseries, marker = 'o')] * 2,
                        'title': 'Timeseries motor and proprioception',
                        'xticks': False,
                    },
                ],
                [
                    {
                        'input': ['s1'],
                        'plot': [partial(timeseries, marker = 'o')] * 2,
                        'title': 'Timeseries exteroception'
                    },
                ],
            ],
        },
    }),
    
    # cross-correlation scan
    ('xcorr_m2s', {
        'block': XCorrBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'pre_l0/pre', 'shape': (dim_s0, numsteps)}, 'x': {'bus': 'robot1/s1', 'shape': (dim_s0, numsteps)}},
            'shift': (scanstart, scanstop),
            'outputs': {'xcorr': {'shape': (dim_m0, dim_s0, scanlen)}},
        }
    }),
    
    # cross-correlation scan
    ('xcorr_s2s', {
        'block': XCorrBlock2,
        'params': {
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'robot1/s1', 'shape': (dim_s0, numsteps)}, 'x': {'bus': 'robot1/s1', 'shape': (dim_s0, numsteps)}},
            'shift': (scanstart, scanstop),
            'outputs': {'xcorr': {'shape': (dim_m0, dim_s0, scanlen)}},
        }
    }),
    
    # motor/sensor mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loop': [
                (
                    'inputs', {'y': {'bus': 'pre_l0/pre', 'shape': (dim_m0, numsteps)}, 'x': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)}}
                ),
                (
                    'inputs', {'y': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)}, 'x': {'bus': 'robot1/s1', 'shape': (dim_s1, numsteps)}}
                ),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': True,
                    'norm_out': False,
                    'inputs': {
                        'x': {'bus': 'puppyslice/x_all'},
                        'y': {'bus': 'puppylog/y'}
                    },
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
    ('plot_xcorr', {
        'block': ImgPlotBlock2,
        'params': {
            'logging': False,
            'saveplot': saveplot,
            'debug': False,
            #           dim       col w   s  dim      row h   s
            'savesize': (dim_s1 * 1 * 4 * 3, dim_m0 * 4 * 1 * 3),
            'desc': """Results of a cross-correlation scan (top) and a
            mutual information scan (bottom). Normalized correlation
            coefficients take on values in the interval $[-1, 1]$. The
            mutual information is unnormalized in the range of $[0,
            1.6]$. The mutual information captures the interaction
            between action and velocity which is, by design, not the
            case for cross-correlation.""",
            'blocksize': numsteps,
            # 'inputs': make_input_matrix(xdim = dim_m0, ydim = dim_s0, with_t = True),
            'inputs': scan_plot_inputs,
            # 'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            'subplots': [
                # [{'input': ['d3'], 'ndslice': (i, j, ), 'xslice': (0, scanlen), 'xaxis': 't',
                #   'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(dim_m0)]
                # for i in range(dim_s0)],
                # xcorr scan
                [
                    {
                        'input': ['m2s'], 'ndslice': (slice(scanlen), j, i),
                        'shape': (dim_s0, scanlen), 'cmap': 'RdGy',
                        'title': 'Cross-correlation $m_0 \star s_1$',
                        'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                        'xticks': False, 'xlabel': None,
                        'yticks': False, 'ylabel': None,
                        # 'xaxis': range(scanstart, scanstop),
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                    } for i in range(dim_m0)] # 'seismic'
                for j in range(dim_s1)] + [
                [
                    {
                        'input': ['s2s'], 'ndslice': (slice(scanlen), j, i),
                        'shape': (dim_s1, scanlen), 'cmap': 'RdGy',
                        'title': 'Auto-correlation $s_1 \star s_1$',
                        'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                        'xticks': False, 'xlabel': None,
                        'yticks': False, 'ylabel': None,
                        # 'xaxis': range(scanstart, scanstop),
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                    } for i in range(dim_s1)] # 'seismic'
                for j in range(dim_s1)] + [
                # mutual information scan
                [
                    {
                        'input': 'd1',
                        # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                        # 'xslice': (0, 1),
                        'ndslice': (slice(scanlen), i, 0),
                        'shape': (dim_s1, scanlen), 'cmap': 'Greys',
                        'title': 'Mutual information $I_m(m_0; s_1)$',
                        # 'vmin': 0.0, 'vmax': 1.0,
                        'vaxis': 'cols',
                        # 'xticks': (np.arange(scanlen) + 0.5).tolist(),
                        # 'xlabel': 'time shift [steps]',
                        'xticks': False, 'xlabel': None,
                        'xticklabels': list(range(scanstart, scanstop)),
                        'yticks': False, 'ylabel': None,
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                    } for i in range(dim_m0)]
                for j in range(dim_s1)] + [
                # mutual information scan
                [
                    {
                        'input': 'd2',
                        # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                        # 'xslice': (0, 1),
                        'ndslice': (slice(scanlen), i, 0),
                        'shape': (dim_s1, scanlen), 'cmap': 'Greys',
                        'title': 'Self information $I_m(s_1; s_1)$',
                        # 'vmin': 0.0, 'vmax': 1.0,
                        'vaxis': 'cols',
                        'xticks': (np.arange(0, scanlen, 5) + 0.5).tolist(),
                        'xticklabels': list(range(scanstart, scanstop, 5)),
                        'xlabel': 'time shift [steps]', 'ylog': True,
                        'yticks': False, 'ylabel': None,
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                    } for i in range(dim_s1)]
                for j in range(dim_s1)],
        },
    }),

    # # plot mi matrix as image
    # ('plot2_mi', {
    #     'enable': False,
    #     'block': ImgPlotBlock2,
    #     'params': {
    #         'logging': False,
    #         'saveplot': saveplot,
    #         # 'debug': False,
    #         'wspace': 0.1,
    #         'hsapce': 0.1,
    #         'blocksize': numsteps,
    #         # 'inputs': {'d1': {'bus': 'mi_1/mi'}, 'd2': {'bus': 'infodist_1/infodist'}},
    #         'inputs': {'d1': {'bus': 'mi_ll0_ll0/mi', 'shape': (dim_s0, dim_m0, scanlen)}}, # .update(make_input_matrix_ndim(xdim = dim_m0, ydim = dim_s0, with_t = True, scan = (scanstart, scanstop))),
    #         # 'outputs': {}, #'x': {'shape': (3, 1)}},
    #         # 'subplots': [
    #         #     [
    #         #         {'input': 'd1', 'xslice': (0, (xdim + dim_s0)**2),
    #         #              'shape': (xdim+dim_s0, xdim+dim_s0), 'plot': 'bla'},
    #         #         {'input': 'd2', 'xslice': (0, (xdim + dim_s0)**2),
    #         #              'shape': (xdim+dim_s0, xdim+dim_s0), 'plot': 'bla'},
    #         #     ],
    #         # ],
    #         # 'subplots': [
    #         #     [
    #         #         {'input': 'd1', 'xslice': (0, xdim * dim_s0),
    #         #              'shape': (dim_s0, xdim), 'plot': 'bla'},
    #         #         {'input': 'd2', 'xslice': (0, xdim * dim_s0),
    #         #              'shape': (dim_s0, xdim), 'plot': 'bla'},
    #         #     ],
    #         # ],
    #         'subplots': [
                
    #             [
    #                 {'input': 'd1',
    #                  # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
    #                  # 'xslice': (0, 1),
    #                  'ndslice': (i, slice(None), slice(None)),
    #                  'shape': (dim_s0, dim_m0),
    #                  'title': 'mi-matrix', 'cmap': 'Reds',
    #                  'vaxis': 'rows',
    #                  'plot': 'bla'} for i in range(scanlen)
    #             ],
                
    #             # [
    #             #     {'input': 'd2',
    #             #      # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
    #             #      # 'xslice': (0, 1),
    #             #      'ndslice': (i, slice(None), slice(None)),
    #             #      'title': 'te-matrix', 'cmap': 'Reds',
    #             #      'vaxis': 'rows',
    #             #      'shape': (dim_s0, dim_m0),
    #             #      'plot': 'bla'} for i in range(scanlen)
    #             # ],
    #             # [
    #             #     {'input': 'd3',
    #             #      # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
    #             #      # 'xslice': (0, 1),
    #             #      'ndslice': (i, slice(None), slice(None)),
    #             #      'title': 'cte-matrix', 'cmap': 'Reds',
    #             #      'vaxis': 'rows',
    #             #      'shape': (dim_s0, dim_m0),
    #             #      'plot': 'bla'} for i in range(scanlen)
    #             # ],
    #         ],
    #     },
    # }),
    
])
