"""dm_actinf.py

Developmental model: *actinf*, smp_graphs configured for base
experiment

Variations:
- actinf variant: M1: goal and goal error, M2: goal error only, M3: ?
- data dimensionality: 1, 2, 3, ..., N?
- proprioceptive space upwards: include extero
- Temporal embedding configured with lag* parameters to system and
  models

TODO:
- FIXME: target function properties like: frequency matched to body
  (attainable / not attainable)
- FIXME: learning rate: eta
- FIXME: priors: limits, learn the sensorimotor limits
- FIXME: priors: tapping, learn tapping
- FIXME: x insert measure_mse, x insert measure_moments, x insert
  measure_accumulated_error

Log:
- renamed from dm_actinf_m1_goal_error_ND_embedding.py
- ported from actinf/active_inference_basic.py --mode m1_goal_error_nd
"""

import copy
from functools import partial

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2, MSEBlock2, MomentBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2, BhasimulatedBlock2
from smp_graphs.block_cls_ros import STDRCircularBlock2, LPZBarrelBlock2, SpheroBlock2

from smp_graphs.common import escape_backslash
from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform, f_sin_noise

from smp_graphs.utils_conf import get_systemblock

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = False # True

# experiment
commandline_args = ['numsteps']
randseed = 12348

################################################################################
# experiment variations
# - algo
# - system
# - system order
# - dimensions
# - number of modalities
    
lconf = {
    # execution and global
    'numsteps': int(10000/5),
    # system
    'sys': {
        'name': 'pm',
        'lag': 0,
        'dim_s0': 2,
        'dim_s1': 2,
    },
    # motivation
    'motivation_i': 0,
    # models
    # learning modulation
    'lm' : {
        'washout': 200,
        'thr_predict': 200,
        'fit_offset': 200,
        # start stop in ratios of episode length numsteps
        'r_washout': (0.0, 0.1),
        'r_train': (0.1, 0.8),
        'r_test': (0.8, 1.0),
    }
}
lconf['systemblock'] = get_systemblock[lconf['sys']['name']](**lconf['sys'])

numsteps = lconf['numsteps']
loopblocksize = numsteps

# learning modulation
for k in list(lconf['lm']):
    if not k.startswith('r_'): continue
    k_ = k.replace('r_', 'n_')
    r0 = lconf['lm'][k][0] * lconf['numsteps']
    r1 = lconf['lm'][k][1] * lconf['numsteps']
    # lconf['lm'][k_] = range(int(r0), int(r1))
    lconf['lm'][k_] = (int(r0), int(r1))

sysname = lconf['sys']['name']
# sysname = 'pm'
# sysname = 'sa'
# sysname = 'bha'
# sysname = 'stdr'
# sysname = 'lpzbarrel'
# sysname = 'sphero'
# dim = 3 # 2, 1
# dim = 9 # bha
    
outputs = {
    'latex': {'type': 'latex',},
}

# get systemblock
systemblock = lconf['systemblock']

# print "systemblock", systemblock
dim_s0 = systemblock['params']['dims']['s0']['dim']
print "dim_s0", dim_s0
dim_s1 = systemblock['params']['dims']['s1']['dim']
print "dim_s1", dim_s1
m_mins = np.array([systemblock['params']['m_mins']]).T
print "m_mins", m_mins
m_maxs = np.array([systemblock['params']['m_maxs']]).T
print "m_maxs", m_maxs

lag = systemblock['params']['lag']
lag_past = systemblock['params']['lag_past']
lag_future = systemblock['params']['lag_future']

dt = systemblock['params']['dt']

algo_conf = {
    'knn': {
        'name': 'k-nearest neighbors',
    },
}

algo = 'knn' #
# algo = 'gmm' #
# algo = 'igmm' #
# algo = 'hebbsom'
# algo = 'soesgp'
# algo = 'storkgp'
# algo = 'resrls'
# algo = 'homeokinesis'

# lag_past = (-21, -20)
# lag_past = (-11, -10)
# lag_past = (-7, -5)
# lag_past = (-6, -5)
# lag_past = (-5, -4)
# lag_past = (-4, -3)
# lag_past = (-3, -2)
# lag_past = (-2, -1)
# lpzbarrel non-overlapping seems important
# lag_past = (-6, -2)
# lag_future = (-1, 0)

# # pm experiment with lag = 5
# lag_past = (-7, -5)
# lag_future = (-2, 0)

minlag = 1 # -lag_future[1]
# maxlag = -lag_past[0] # + lag_future[1]
maxlag = max(20, -lag_past[0])
laglen = maxlag # - minlag

# eta = 0.99
# eta = 0.95
eta = 0.7
# eta = 0.3
# eta = 0.25
# eta = 0.15
# eta = 0.1
# eta = 0.01

# print "dm_actinf_m1_goal_error_ND_embedding.py: dim_s0 = %s" % (dim_s0, )

# motivations
from smp_graphs.utils_conf import dm_motivations
motivations = dm_motivations(m_mins, m_maxs, dim_s0, dt)
motivation_i = lconf['motivation_i']
goal = motivations[motivation_i][1]['params']['models']['goal']['type']
    
desc = """An exemplary experiment involving a single developmental
episode of {0} steps of an actinf agent. The episode starts with an
initial washout period of 1/10th of the episode length during which
the top-down prediction is applied and the model is allowed to predict
but its is only predicting from its bootstrapping state. Starting with
time step {1}, the model is being updated with the incoming
measurements, which almost immediately has the effect of bringing the
system into the state predicted by the goal. The goal or target
function is a {2} sequence of proprioceptive states. In this case,
every consecutive change of goal leads to a new mini-episode of
learning for new combinations of goal prediction and local prediction
error. This is a simple type of \\emph{{goal babbling}}, and it is
only partially completed when learning is stopped at time step
{3}. The agent able to get close to most of the goal predictions based
on the existing knowledge but the remaining residual after the goal
change transient is not being corrected anymore.""".format(
    numsteps, lconf['lm']['n_train'][0], escape_backslash(goal),
    lconf['lm']['n_test'][0])

def plot_timeseries_block(l0 = 'pre_l0', l1 = 'pre_l1', blocksize = 1):
    global partial
    global PlotBlock2, numsteps, timeseries, algo, dim_s1, dim_s0, sysname, lag, lag_past, lag_future, saveplot
    global motivations, motivation_i, goal
    # goal = motivations[motivation_i][1]['params']['models']['goal']['type']
    return {
    'block': PlotBlock2,
    'params': {
        'blocksize': blocksize,
        'saveplot': saveplot,
        # 'debug': True,
        'title': '%s\nalgo %s, sys %s(dim_p=%d), goal = %s, lag %d, tap- %s, tap+ %s' % (
            'dm actinf', algo, sysname, dim_s0, goal, lag, lag_past, lag_future),
        'desc': """An {1} agent learning to control a {0}-dimensional {2}
            system using the {3} low-level algorithm.""".format(dim_s0, 'actinf', sysname, algo),
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim_s0, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim_s0, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim_s0, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim_s0, blocksize)},
            'X_fit': {'bus': '%s/X_fit' % (l0,), 'shape': (dim_s0 * (lag_past[1] - lag_past[0]) * 2, blocksize)},
            's0': {'bus': 'robot1/s0', 'shape': (dim_s0, blocksize)},
            's1': {'bus': 'robot1/s1',  'shape': (dim_s1, blocksize)},
            'mse_s_p_accum': {'bus': 'mse_s_p_accum/Ix', 'shape': (dim_s0, blocksize)},
            },
        'hspace': 0.2,
        'subplots': [
            [
                {
                    'input': ['err',], 'plot': partial(timeseries, marker='.'),
                    'title': 'Momentary prediction error',
                    'legend': {'e(s_p)': 0},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['goals', 's0', 'pre'],
                    'plot': [partial(timeseries, marker='.')] * 3,
                    'title': 'Goal, state, and prediction',
                    'legend': {'Goal': 0, 'State': dim_s0, 'State_p': 2*dim_s0},
                    'xticks': False,
                }
            ],
            
            # [
            #     {
            #         'input': ['goals', 'pre', 's0'],
            #         'plot': partial(timeseries, marker='.')
            #     },
            # ],
            
            [
                {
                    'input': ['X_fit'], 'plot': partial(timeseries, marker='.'),
                    'title': 'Model %s input $\mathbf{X}$' % (algo),
                    'legend': {'X_pre_l1': 0, 'X_prerr_l0': dim_s0},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['tgt'], 'plot': partial(timeseries, marker='.'),
                    'title': 'Model %s input $\mathbf{Y}$' % (algo),
                    'legend': {'Y_tgt': 0},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['mse_s_p_accum',],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Accumulated error',
                    'legend': {'$\sum$ err': 0},
                }
            ],
            
            # [
            #     {'input': ['s0', 's1'], 'plot': timeseries},
            # ],
            
            # [
            #     {
            #         'input': ['hidden'], 'plot': partial(timeseries, marker = '.'),
            #         'title': 'Model %s hidden activation $\mathbf{Z}$' % (algo),
            #         # FIXME: knn particulars
            #         'legend': {'Z_dist': 0, 'E(Z_dist)': 5, 'Z_idx': 6},
            #         'xticks': False,
            #     },
            # ],
            
            # [
            #     {
            #         'input': ['wo_norm', 'wo_norm_fwd'], 'plot': partial(timeseries, marker = '.'),
            #         'title': 'Model %s parameter norm (accumulated adaptation)' % (algo),
            #         'legend': {'|W|': 0}
            #     },
            # ],
            
        ],
    }}

"""
sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepsys_steps = 40 # 6
sweepsys_input_flat = np.power(sweepsys_steps, dim_s0)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s0']['shape'] = (dim_s0, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s1']['shape']  = (dim_s1, sweepsys_input_flat)

sweepmdl_steps = 1000
sweepmdl_input_flat = sweepmdl_steps # np.power(sweepmdl_steps, dim_s0 * 2)
sweepmdl_func = f_random_uniform

# sweepmdl_steps = 3
# sweepmdl_input_flat = np.power(sweepmdl_steps, dim_s0 * 2)
# sweepmdl_func = f_meshgrid

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': sweepsys_input_flat,  # inner numsteps when used as loopblock (sideways time)
        'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
        'blockphase': [0],        # phase = 0
        'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat), 'buscopy': 'sweepsys_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s0)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
                    # 'func': f_meshgrid
                    'func': f_random_uniform,
                    },
                }),
                
                # sys to sweep
                sweepsys,

            # ('sweepsys_grid', {
            #     'block': FuncBlock2,
            #     'params': {
            #         'debug': False,
            #         'blocksize': sweepsys_input_flat,
            #         'inputs': {
            #             'ranges': {'val': np.array([[-1, 1]] * dim_s0)},
            #             'steps':  {'val': sweepsys_steps},
            #             },
            #         'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
            #         'func': f_meshgrid
            #         },
            #     }),
                
            #     # sys to sweep
            #     sweepsys,

            ]),
        }
    }

loopblock_model = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': sweepmdl_input_flat,  # inner numsteps when used as loopblock (sideways time)
        'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
        'blockphase': [0],        # phase = 0
        'outputs': {
            'pre': {
                'shape': (dim_s0 * 2, sweepmdl_input_flat),
                'buscopy': 'sweepmdl_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            # model sweep input
            ('sweepmdl_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepmdl_input_flat,
                    'inputs': {
                        # 'ranges': {'val': np.array([[-1, 1], [-1e-0, 1e-0]])},
                        'ranges': {
                            'val': np.array([[m_mins[0], m_maxs[0]]] * dim_s0 * 2)},
                            # 'val': np.vstack((
                            #     np.array([[m_mins[0], m_maxs[0]]] * dim_s0),
                            #     np.array([[-2.0,      1.0]]       * dim_s0),
                            #     ))},
                        'steps':  {'val': sweepmdl_steps},
                        },
                    'outputs': {
                        'meshgrid': {
                            'shape': (
                                dim_s0 * 2,
                                sweepmdl_input_flat,)}},
                    'func': sweepmdl_func,
                    },
                }),
                
            # # model sweep input
            # ('sweepmdl_grid_goal', {
            #     'block': FuncBlock2,
            #     'params': {
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             'ranges': {'val': np.array([[-1, 1]])},
            #             'steps':  {'val': sweepmdl_steps},
            #             },
            #         'outputs': {'pre': {'shape': (dim_s0, sweepmdl_input_flat)}},
            #         'func': f_meshgrid,
            #         },
            #     }),
                
            # # model sweep input
            # ('sweepmdl_grid_err', {
            #     'block': FuncBlock2,
            #     'params': {
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             'ranges': {'val': np.array([[-1e-0, 1e-0]])},
            #             'steps':  {'val': sweepmdl_steps},
            #             },
            #         'outputs': {'pre': {'shape': (dim_s0, sweepmdl_input_flat)}},
            #         'func': f_meshgrid,
            #         },
            #     }),

            ('sweep_slice', {
                'block': SliceBlock2,
                'params': {
                    'id': 'sweep_slice',
                    'blocksize': sweepmdl_input_flat,
                    # puppy sensors
                    'inputs': {
                        'x': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim_s0 * 2, sweepmdl_input_flat)}},
                    'slices': {
                        'x': {
                            'goals': slice(0, dim_s0),
                            'errs':  slice(dim_s0, dim_s0 * 2)},
                        },
                    # 'slices': ,
                    }
                }),

            # model sweep
            # learner: basic actinf predictor proprio space learn_proprio_base_0
            ('pre_l0_test', {
                'block': ModelBlock2,
                'params': {
                    'blocksize': sweepmdl_input_flat,
                    'blockphase': [0],
                    # 'debug': True,
                    'lag': minlag,
                    'eta': eta, # 0.3,
                    'inputs': {
                        # descending prediction
                        'pre_l1': {
                            'bus': 'sweep_slice/x_goals',
                            'shape': (dim_s0,sweepmdl_input_flat)},
                        # ascending prediction error
                        'pre_l0': {
                            'bus': 'sweep_slice/x_errs',
                            'shape': (dim_s0, sweepmdl_input_flat)},
                        # ascending prediction error
                        'prerr_l0': {
                            'bus': 'pre_l0_test/err',
                            'shape': (dim_s0, minlag+1), 'lag': minlag},
                        # measurement
                        'meas_l0': {
                            'val': np.array([[-np.inf for i in range(sweepmdl_input_flat)]] * dim_s0),
                            # 'bus': 'robot1/s0',
                            'shape': (dim_s0, sweepmdl_input_flat)}},
                    'outputs': {
                        'pre': {'shape': (dim_s0, sweepmdl_input_flat)},
                        'err': {'shape': (dim_s0, sweepmdl_input_flat)},
                        'tgt': {'shape': (dim_s0, sweepmdl_input_flat)},
                        },
                    'models': {
                        # 'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim_s0 * 2, 'odim': dim},
                        'fwd': {
                            'type': 'actinf_m1', 'algo': 'copy',
                            'copyid': 'pre_l0', 'idim': dim_s0 * 2, 'odim': dim_s0},
                        },
                    'rate': 1,
                    },
                }),


            ('pre_l0_combined', {
                'block': StackBlock2,
                'params': {
                    'inputs': {
                        # 'goals': {
                        #     'bus': 'sweepmdl_grid_goal/pre',
                        #     'shape': (dim_s0, sweepmdl_input_flat),
                        #     },
                        # 'errs':  {
                        #     'bus': 'sweepmdl_grid_err/pre',
                        #     'shape': (dim_s0, sweepmdl_input_flat),
                        #     },
                        'goalerrs': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim_s0 * 2, sweepmdl_input_flat),
                            },
                        'pres':  {
                            'bus': 'pre_l0_test/pre',
                            'shape': (dim_s0, sweepmdl_input_flat),
                            },
                        },
                    'outputs': {
                        'y': {'shape': (dim_s0 * 3, sweepmdl_input_flat)}},
                    }
                }),
                
            # # plot timeseries
            # ('plot_ts',
            #      plot_timeseries_block(
            #          l0 = 'pre_l0_test',
            #          l1 = 'sweepmdl_grid_goal',
            #          blocksize = sweepmdl_input_flat)),
                        
            # # plot model sweep 1d
            # ('plot_model_sweep', {
            #     'block': ImgPlotBlock2,
            #     'params': {
            #         'id': 'plot_model_sweep',
            #         'logging': False,
            #         'saveplot': saveplot,
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             # 'sweepin_goal': {
            #             #     'bus': 'sweepmdl_grid_goal/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             # 'sweepin_err':  {
            #             #     'bus': 'sweepmdl_grid_err/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             # 'sweepout_mdl':  {
            #             #     'bus': 'pre_l0_test/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             'all': {
            #                 'bus': 'pre_l0_combined/y',
            #                 'shape': (dim_s0 * 3, sweepmdl_input_flat),
            #                 },
            #             },
            #         'outputs': {},
            #         'wspace': 0.5,
            #         'hspace': 0.5,
            #         # with one subplot and reshape
            #         'subplots': [
            #             [
            #                 {
            #                     'input': ['all'],
            #                     'shape': (dim_s0 * 3, sweepmdl_input_flat),
            #                     'ndslice': [(slice(None), slice(None))],
            #                     # 'vmin': -1.0, 'vmax': 1.0,
            #                     # 'vmin': 0.1, 'vmax': 0.3,
            #                     'cmap': 'RdGy',
            #                     'dimstack': {
            #                         'x': range(2*dim-1, dim - 1, -1),
            #                         'y': range(dim-1,   -1     , -1)},
            #                     'digitize': {'argdims': range(0, dim_s0 * 2), 'valdim': 2*dim+i, 'numbins': 2},
            #                 } for i in range(dim_s0)],

            #             ],
            #         },
            #     }),
            ]),
        }
    }
        

# main graph
graph = OrderedDict([
    
    # # sweep system
    # ("sweepsys", {
    #     'debug': False,
    #     'block': SeqLoopBlock2,
    #     'params': {
    #         'id': 'sweepsys',
    #         # loop specification, check hierarchical block to completely
    #         # pass on the contained in/out space?
    #         'blocksize': numsteps, # execution cycle, same as global numsteps
    #         'blockphase': [1],     # execute on first time step only
    #         'numsteps':  numsteps,          # numsteps      / loopblocksize = looplength
    #         'loopblocksize': loopblocksize, # loopblocksize * looplength    = numsteps
    #         # can't do this dynamically yet without changing init passes
    #         'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
    #         'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
    #         'loopmode': 'sequential',
    #         'loopblock': loopblock,
    #     },
    # }),

    # # plot the system sweep result
    # ('plot_sweep', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'debug': False,
    #         'blocksize': numsteps, # sweepsys_input_flat,
    #         'title': 'system sweep',
    #         'inputs': {
    #             'meshgrid': {
    #                 'bus': 'sweepsys_grid/meshgrid',
    #                 'shape': (dim_s0, sweepsys_input_flat)},
    #             's0': {
    #                 'bus': 'robot0/s0',
    #                 'shape': (dim_s0, sweepsys_input_flat)},
    #             's1': {
    #                 'bus': 'robot0/s1',
    #                 'shape': (dim_s1, sweepsys_input_flat)},
    #             },
    #             'hspace': 0.2,
    #             'subplots': [
    #                 [
    #                     {'input': ['meshgrid'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s0'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s1'], 'plot': timeseries},
    #                 ],
    #                 ],
    #         }
    #     }),

    # # sns matrix plot
    # ('plot2', {
    #     'block': SnsMatrixPlotBlock2,
    #     'params': {
    #         'id': 'plot2',
    #         'logging': False,
    #         'debug': False,
    #         'saveplot': saveplot,
    #         'blocksize': numsteps,
    #         'inputs': {
    #             'meshgrid': {
    #                 'bus': 'sweepsys_grid/meshgrid',
    #                 'shape': (dim_s0, sweepsys_input_flat)},
    #             's0': {
    #                 'bus': 'robot0/s0',
    #                 'shape': (dim_s0, sweepsys_input_flat)},
    #             's1': {
    #                 'bus': 'robot0/s1',
    #                 'shape': (dim_s1, sweepsys_input_flat)},
    #             },
    #         'outputs': {},#'x': {'shape': 3, 1)}},
    #         'subplots': [
    #             [
    #                 # stack inputs into one vector (stack, combine, concat
    #                 {
    #                     'input': ['meshgrid', 's0', 's1'],
    #                     'mode': 'stack',
    #                     'plot': histogramnd
    #                 },
    #             ],
    #         ],
    #     },
    # }),
        
    # system block from definition elsewhere
    ('robot1', systemblock),

    # learning experiment
    ('brain_learn_proprio', {
        'block': Block2,
        'params': {
            'numsteps': numsteps,
            'blocksize': 1,
            'graph': OrderedDict([

                ('cnt', {
                    'block': CountBlock2,
                    'params': {
                        'blocksize': 1,
                        # 'debug': True,
                        # 'inputs': {},
                        'outputs': {'x': {'shape': (dim_s0, 1)}},
                        },
                }),

                # generic block configured with subgraph on demand
                ('pre_l1', {
                    'block': Block2,
                    'params': {
                        'numsteps': numsteps,
                        'blocksize': 1,
                        # 'subgraph': OrderedDict([motivations[0]]),
                        'subgraph': OrderedDict([motivations[motivation_i]]),
                        'subgraph_rewrite_id': False,
                    }
                }),
                
                # dev model actinf_m1: learner is basic actinf predictor proprio space learn_proprio_base_0
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'debug': True,
                        'lag': minlag,
                        'eta': eta, # 3.7,
                        'ros': ros,
                        # FIXME: relative shift = minlag, block length the maxlag
                        'inputs': {
                            # descending prediction
                            'pre_l1': {
                                'bus': 'pre_l1/pre',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag, -minlag)},
                                # FIXME: check correctness here
                                # training on past goal, prediction with current goal
                                # should be possible inside step_model to use current
                                # goal
                                'shape': (dim_s0, maxlag), 'lag': range(lag_past[0], lag_past[1])},
                            # ascending prediction error
                            'pre_l0': {
                                'bus': 'pre_l0/pre',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s0, maxlag), 'lag': range(lag_past[0] + 1, lag_past[1] + 1)},
                            # ascending prediction error
                            'prerr_l0': {
                                'bus': 'pre_l0/err',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s0, maxlag), 'lag': range(lag_past[0] + 1, lag_past[1] + 1)},
                            # measurement
                            'meas_l0': {
                                'bus': 'robot1/s0',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-laglen, 0)}
                                'shape': (dim_s0, maxlag), 'lag': range(lag_future[0], lag_future[1])}
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
                            'err': {'shape': (dim_s0, 1)},
                            'tgt': {'shape': (dim_s0, 1)},
                            'X_fit': {'shape': (dim_s0 * (lag_past[1] - lag_past[0]) * 2, 1)},
                            },
                        'models': {
                            
                            'm1': {
                                'type': 'actinf_m1',
                                'algo': algo,
                                'lag_past': lag_past,
                                'lag_future': lag_future,
                                'idim': dim_s0 * (lag_past[1] - lag_past[0]) * 2, # laglen
                                'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                                'laglen': laglen,
                                'eta': eta,
                                # 'laglen_past': lag_past[1] - lag_past[0],
                                # 'laglen_future': lag_future[1] - lag_future[0],
                                # learning modulation
                                'n_washout': lconf['lm']['n_washout'],
                                'n_train': lconf['lm']['n_train'],
                                'n_test': lconf['lm']['n_test'],
                            },
                                
                            # 'm2': {
                            #     'type': 'actinf_m2',
                            #     'algo': algo,
                            #     'idim': dim_s0 * laglen,
                            #     'odim': dim_s0 * laglen,
                            #     'laglen': laglen,
                            #     'eta': eta
                            #     },
                            
                            # 'm3': {
                            #     'type': 'actinf_m3',
                            #     'algo': algo,
                            #     'idim': dim_s0 * laglen * 2,
                            #     'odim': dim_s0 * laglen,
                            #     'laglen': laglen,
                            #     'eta': eta
                            #     },
                                
                            },
                        'rate': 1,
                        },
                    }),

                # # dev model uniform random sampler
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'inputs': {                        
                #             'lo': {'val': m_mins, 'shape': (dim_s0, 1)},
                #             'hi': {'val': m_maxs, 'shape': (dim_s0, 1)},
                #             },
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'err': {'val': np.zeros((dim_s0, 1)), 'shape': (dim_s0, 1)},
                #             'tgt': {'val': np.zeros((dim_s0, 1)), 'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                #             'goal': {'type': 'random_uniform'}
                #             },
                #         'rate': 50,
                #         },
                #     }),
                    
                # # dev model homeokinesis
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'debug': False,
                #         'lag': minlag,
                #         'eta': eta, # 3.7,
                #         'ros': ros,
                #         'inputs': {
                #             # descending prediction
                #             'pre_l1': {
                #                 'bus': 'pre_l1/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # ascending prediction error
                #             'pre_l0': {
                #                 'bus': 'pre_l0/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # ascending prediction error
                #             'prerr_l0': {
                #                 'bus': 'pre_l0/err',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # measurement
                #             'meas_l0': {
                #                 'bus': 'robot1/s0', 'shape': (dim_s0, maxlag)}},
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'err': {'shape': (dim_s0, 1)},
                #             'tgt': {'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                #             'hk': {
                #                 'type': 'homeokinesis', 'algo': algo, 'mode': 'hk', 'idim': dim_s0, 'odim': dim_s0,
                #                 'minlag': minlag, 'maxlag': maxlag, 'laglen': laglen, 'm_mins': m_mins, 'm_maxs': m_maxs,
                #                 'epsA': 0.01, 'epsC': 0.03, 'creativity': 0.8},
                #             },
                #         'rate': 1,
                #         },
                #     }),
                    
                # learn_proprio_e2p2e

                # measure and introspect
                # total MSE goal - state
                ('mse_s_p', {
                    'block': MSEBlock2,
                    'params': {
                        'blocksize': 1, # numsteps,
                        'inputs': {
                            'x': {'bus': 'pre_l1/pre', 'shape': (dim_s0, 1)},
                            'x_': {'bus': 'robot1/s0', 'shape': (dim_s0, 1)},
                            },
                        'outputs': {
                            'y': {'shape': (dim_s0, 1)},
                            },
                        },
                    }),
                    
                # accumulated MSE goal - state over time
                ('mse_s_p_accum', {
                    'block': IBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'inputs': {
                            'x': {'bus': 'mse_s_p/y', 'shape': (dim_s0, numsteps)},
                            },
                        'outputs': {
                            'Ix': {'shape': (dim_s0, numsteps)},
                            },
                        },
                    }),
                    
                
                # end brain
                ]),
            }
        }),

    ################################################################################
    # use a sequential loop block to insert probes running orthogonally in time
    # blocksize:  seqloops blocksize from the outside, same as overall experiment
    # blockphase: points in the cnt % numsteps space when to execute
    # numsteps:      
    # loopblocksize: number of loop iterations = numsteps/loopblocksize 
    # ("sweepmodel", {
    #     'debug': False,
    #     'block': SeqLoopBlock2,
    #     'params': {
    #         'id': 'sweepmodel',
    #         'blocksize': numsteps, # execution cycle, same as global numsteps
    #         #                        execution phase, on first time step only
    #         # 'blockphase': [numsteps/2, numsteps-10],
    #         # 'blockphase': [int(i * numsteps)-1 for i in np.linspace(1.0/2, 1, 2)],
    #         'blockphase': [int(i * numsteps)-1 for i in np.linspace(1.0/1, 1, 1)],
    #         # 'blockphase': [0],
    #         'numsteps':  1, # numsteps,          # numsteps      / loopblocksize = looplength
    #         'loopblocksize': 1, #loopblocksize, # loopblocksize * looplength    = numsteps
    #         # can't do this dynamically yet without changing init passes
    #         'outputs': {'pre': {'shape': (dim_s0 * 2, sweepmdl_input_flat)}},
    #         'loop': [('none', {}) for i in range(2)], # lambda ref, i, obj: ('none', {}),
    #         'loopmode': 'sequential',
    #         'loopblock': loopblock_model,
    #     },
    # }),        
        
    # plot timeseries
    ('plot_ts', plot_timeseries_block(l0 = 'pre_l0', blocksize = numsteps)),

    # # plot / print measures
    # ('plot_meas', {
    #     'block': PlotBlock2,
    #     'params': {
            
    #         },
    #     }),
    
    ])
