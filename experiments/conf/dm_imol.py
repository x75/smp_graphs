"""dm_imol - developmental model (dm) using internal model online learning (imol)

.. moduleauthor:: Oswald Berthold, 2018

smp_graphs configurables, converted from smp/imol

Experiment variations:
- algo
- system
- system order
- dimensions
- number of modalities

TODO:
- Embedding: implement embedding at block boundaries
- robustness
- fix: target properties like frequency matched to body (attainable / not attainable)
- fix: eta
- fix: motor / sensor limits, how does the model learn the limits
- fix: priors: timing, limits, ...
"""

import copy
from functools import partial

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2, BhasimulatedBlock2
from smp_graphs.block_cls_ros import STDRCircularBlock2, LPZBarrelBlock2, SpheroBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform, f_sin_noise
from smp_graphs.utils_conf import get_systemblock

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = False

outputs = {
    'latex': {'type': 'latex',},
}

# experiment
commandline_args = ['numsteps']
randseed = 12355

lconf = {
    # execution and global
    'numsteps': int(10000/4),
    
    # system
    'sys': {
        'name': 'pm',
        'lag': 2,
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
    
"""system block
"""

# def get_systemblock_pm(dim_s0 = 2, dim_s1 = 2, dt = 0.1):
#     global np, PointmassBlock2, meas
#     return {
#         'block': PointmassBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'systype': 2,
#             'sysdim': dim_s0,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s0 * 3, 1)),
#             # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1':  {'shape': (dim_s1, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s0 * 3,
#             'dt': dt,
#             'mass': 1.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.01,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s0': dim_s0,
#             'length_ratio': 3./2., # gain curve?
#             'm_mins': [-1.] * dim_s0,
#             'm_maxs': [ 1.] * dim_s0,
#             'dim_s1': dim_s1,
#             'lag': 3,
#             'order': 2,
#             'coupling_sigma': 1e-3,
#             'transfer': 0,
#             'anoise_mean': 0.0,
#             'anoise_std': 1e-2,
#             }
#         }

# def get_systemblock_sa(dim_s0 = 2, dim_s1 = 2, dt = 0.1):
#     global np, SimplearmBlock2
#     return {
#         'block': SimplearmBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'sysdim': dim_s0,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s0 * 3, 1)),
#             # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1':  {'shape': (dim_s1,  1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s0 * 3,
#             'dt': dt,
#             'mass': 1.0/3.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.001,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s0': dim_s0,
#             'length_ratio': 3./2.0,
#             'm_mins': [-1.] * dim_s0,
#             'm_maxs': [ 1.] * dim_s0,
#             # 's_mins': [-1.00] * 9,
#             # 's_maxs': [ 1.00] * 9,
#             # 'm_mins': -1,
#             # 'm_maxs': 1,
#             'dim_s1': dim_s1,
#             'minlag': 1,
#             'maxlag': 2, # 5
#             }
#         }

# def get_systemblock_bha(dim_s0 = 9, dim_s1 = 3, dt = 0.1):
#     global np, BhasimulatedBlock2
#     return {
#         'block': BhasimulatedBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'sysdim': dim_s0,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s0 * 3, 1)),
#             # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1':  {'shape': (dim_s1,  1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s0 * 3,
#             'dt': dt,
#             'mass': 1.0/3.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.001,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s0': dim_s0,
#             # 'length_ratio': 3./2.,
#             # 'm_mins': 0.05, # 0.1
#             # 'm_maxs': 0.4,  # 0.3
#             'dim_s1': 3,
#             'numsegs': 3,
#             'segradii': np.array([0.1,0.093,0.079]),
#             'm_mins': [ 0.10] * dim_s0,
#             'm_maxs': [ 0.30] * dim_s0,
#             's_mins': [ 0.10] * dim_s0, # fixme all sensors
#             's_maxs': [ 0.30] * dim_s0,
#             'doplot': False,
#             'minlag': 1,
#             'maxlag': 3 # 5
#             }
#         }
    
# # ROS system STDR
# def get_systemblock_stdr(dim_s0 = 2, dim_s1 = 3, dt = 0.1):
#     global np, STDRCircularBlock2
#     return {
#         'block': STDRCircularBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1': {'shape': (dim_s1, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'ros': True,
#             'dt': dt,
#             'm_mins': [-0.3, 0], # [-0.1] * dim_s0,
#             'm_maxs': [0.3, np.pi/4.0],    # [ 0.1] * dim_s0,
#             'dim_s0': dim_s0, 
#             'dim_s1': dim_s1,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 1, # ha
#             'maxlag': 4, # 5
#             }
#         }

# # ROS system using lpzrobots' roscontroller to interact with the 'Barrel'
# def get_systemblock_lpzbarrel(dim_s0 = 2, dim_s1 = 1, dt = 0.01):
#     global LPZBarrelBlock2
#     systemblock_lpz = {
#         'block': LPZBarrelBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1': {'shape': (dim_s1, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'ros': True,
#             'dt': dt,
#             'm_mins': [-1.] * dim_s0,
#             'm_maxs': [ 1.] * dim_s0,
#             'dim_s0': dim_s0, 
#             'dim_s1': dim_s1,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 2, # 1, # 5, 4
#             'maxlag': 6, # 2,
#             }
#         }
#     return systemblock_lpz

# # systemblock_lpzbarrel = get_systemblock_lpzbarrel(dt = dt)

# # ROS system using the Sphero
# def get_systemblock_sphero(dim_s0 = 2, dim_s1 = 1, dt = 0.05):
#     global SpheroBlock2
#     systemblock_sphero = {
#         'block': SpheroBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's0': {'shape': (dim_s0, 1)},
#                 's1': {'shape': (dim_s1, 1)}
#                 }, # , 's_all': [(9, 1)]},
#                 'ros': True,
#                 'dt': dt,
#             'm_mins': [-0.75] * dim_s0,
#             'm_maxs': [ 0.75] * dim_s0,
#             'dim_s0': dim_s0, 
#             'dim_s1': dim_s1,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 2, # 2, # 4, # 2
#             'maxlag': 5,
#             }
#         }
#     return systemblock_sphero    

# systemblock   = systemblock_lpzbarrel
# lag = 6 # 5, 4, 2 # 2 or 3 worked with lpzbarrel, dt = 0.05
# systemblock   = get_systemblock[sysname](lag = 2)
# lag           = 1
systemblock = lconf['systemblock']

# systemblock['params']['anoise_std'] = 0.0
# systemblock['params']['sysname'] = 0.0
dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1 = systemblock['params']['dims']['s1']['dim']
m_mins = np.array([systemblock['params']['m_mins']]).T
m_maxs = np.array([systemblock['params']['m_maxs']]).T

# minlag = systemblock['params']['minlag']
# maxlag = systemblock['params']['maxlag']

dt = systemblock['params']['dt']

algo_conf = {
    'knn': {
        'name': 'k-nearest neighbors',
    },
}

algo = 'knn' # ok
# algo = 'gmm' # ok
# algo = 'igmm' # ok, fix deprecation, inference output conf
# algo = 'hebbsom' # fix
# algo = 'soesgp'
# algo = 'storkgp'
# algo = 'resrls'

# algo = 'homeokinesis'

# pm
algo_fwd = algo
algo_inv = algo

if algo == 'knn':
    dim_s_hidden_debug = 5 * 2 + 1
    modelsize_fwd = int(numsteps*0.8)
    modelsize_inv = int(numsteps*0.8) # 1000
else:
    dim_s_hidden_debug = 20
    modelsize_fwd = 100
    modelsize_inv = 200

# lag_past   = (-1, 0)
# lag_past   = (-2, -1)
# lag_past   = (-3, -2)
# lag_past   = (-4, -3)
# lag_past   = (-5, -4)
# lag_past   = (-6, -5)
# lag_future = (-1, 0)

lag_future = systemblock['params']['lag_future'] # (-1, 0)
lag_past = systemblock['params']['lag_past'] # (-1, 0)
lag_gap_f2p = lag_future[0] - lag_past[1]
lag_off_f2p = lag_future[1] - lag_past[1]

minlag = 1 # -lag_future[1]
# maxlag = -lag_past[0] # + lag_future[1]
maxlag = max(20, -lag_past[0])
laglen = maxlag # - minlag
# minlag = max(1, -max(lag_past[1], lag_future[1]))
# maxlag = 1 - min(lag_past[0], lag_future[0])
# maxlag_x = 120
# laglen = maxlag - minlag

# lag_past = (-11, -10)
# lag_future = (-5, 0)
# lag_past = (-4, -3)
# lag_future = (-1, 0)

# # lpzbarrel non-overlapping seems important
# lag_past = (-6, -2)
# lag_future = (-1, 0)

# minlag = 1
# maxlag = 20 # -lag_past[0] + lag_future[1]
# laglen = maxlag - minlag

# eta = 0.99
# eta = 0.95
# eta = 0.7
# eta = 0.3
# eta = 0.25
eta = 0.15
# eta = 0.1
# eta = 0.05

devmodel = 'imol'
################################################################################
# experiment description after all configurables have been set
desc = """This is a basic version of the forward / inverse internal
model pair online learning (imol) algorithm put into an actual agent
experiment. The low-level learning algorithm in this particular case
is {0} ({1}). The episode lasts for {2} timesteps. The developmental
schedule within that episode consists of bootstrapping the low-level
models on initialization (uniform random), a warm-up phase (200 time
steps), an actual learning phase (1600 time steps), and a consecutive
testing phase (another 200 time
steps). During the learning phase, the following steps are repeated:
1) compute the (inverse) prediction error using the incoming
measurement, this prediction error is the difference of actual state
and some goal state; 2) the inverse model is fitted to the current
error and the corresponding past input, which is still lingering in
local memory; 3) predict the next motor command from current goal and
state inputs based on the updated model; 4) compute the forward
prediction error; 5) fit the forward model with the forward pe; 6)
make new forward prediction using current state and current motor
prediction. There is no explicit exploration noise involved and only
the cumulative noise of embodiment and model uncertainty is present in
the system. Also, only the inverse model is effective in this
configuration. What is interesting and which can be observed from the
different behaviour of the green and yellow traces in the second row
panel, is the difference in output for identical initializations of
low-level predictors, caused only by their
inputs. """.format(algo_conf[algo]['name'], algo, numsteps)

from smp_graphs.utils_conf import dm_motivations
motivations = dm_motivations(m_mins, m_maxs, dim_s0, dt)
motivation_i = lconf['motivation_i']

def plot_timeseries_block(l0 = 'pre_l0', l1 = 'pre_l1', blocksize = 1):
    global partial, OrderedDict
    global PlotBlock2, timeseries
    global numsteps, saveplot
    global dim_s1, dim_s0, dim_s_hidden_debug
    global devmodel, algo, lag_past, lag_future
    global motivations, motivation_i
    goal = motivations[motivation_i][1]['params']['models']['goal']['type']
    global sysname
    return {
    'block': PlotBlock2,
    'params': {
        'saveplot': saveplot,
        'blocksize': numsteps, # min(numsteps, numsteps), # 1000, # blocksize
        'title': 'Learning episode timeseries: dev-model = %s, algo = %s, system %s, goal = %s' % (devmodel, algo, sysname, goal),
        'desc': """An %s agent learning to control a %d-dimensional %s
            system using the %s low-level algorithm. Please refer to the
            main text of the dm-imol experiment for the detailed
            description.""" % (devmodel, dim_s0, sysname, algo),
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim_s0, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim_s0, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim_s0, blocksize)},
            'prerr_avg': {'bus': '%s/prerr_avg' % (l0,), 'shape': (dim_s0, blocksize)},
            'prerr_rms_avg': {'bus': '%s/prerr_rms_avg' % (l0,), 'shape': (1, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim_s0, blocksize)},
            's0':    {'bus': 'robot1/s0', 'shape': (dim_s0, blocksize)},
            's1':     {'bus': 'robot1/s1',  'shape': (dim_s1, blocksize)},
            'X': {'bus': '%s/X' % (l0, ), 'shape': (dim_s0 * (lag_past[1] - lag_past[0]) * 3, blocksize)},
            'Y': {'bus': '%s/Y' % (l0, ),  'shape': (dim_s0 * (lag_future[1] - lag_future[0]), blocksize)},
            'hidden': {'bus': '%s/hidden' % (l0, ),  'shape': (dim_s_hidden_debug, blocksize)},
            'wo_norm': {'bus': '%s/wo_norm' % (l0, ),  'shape': (dim_s0, blocksize)},
            'pre_fwd': {'bus': '%s/pre_fwd' % (l0,), 'shape': (dim_s0 * (lag_future[1] - lag_future[0]) * 1, blocksize)},
            'prerr_avg_fwd': {'bus': '%s/prerr_avg_fwd' % (l0,), 'shape': (dim_s0, blocksize)},
            'prerr_rms_avg_fwd': {'bus': '%s/prerr_rms_avg_fwd' % (l0,), 'shape': (1, blocksize)},
            'wo_norm_fwd': {'bus': '%s/wo_norm_fwd' % (l0, ),  'shape': (dim_s0, blocksize)},
            },
        'hspace': 0.2,
        'subplots': [
            
            [
                {
                    'input': ['err', 'prerr_rms_avg', 'prerr_rms_avg_fwd'],
                    'plot': [partial(timeseries, marker='.', alpha=0.07), partial(timeseries, marker='.'), partial(timeseries, marker='.')],
                    'title': 'Momentary and time averaged inverse (goal) and forward errors',
                    'legend': {'Error_t': 0, 'E(Error_i_t)': dim_s0, 'E(Error_f_t)': dim_s0 + 1},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['goals', 's0', 'pre', 'pre_fwd'],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Goal, state, inverse, andd forward predictions',
                    # 'legend': OrderedDict([('State', 0), ('State_p', dim_s0)]),
                    'legend': {'Goal': 0, 'State': dim_s0, 'State_p_i': 2*dim_s0, 'State_p_f': 3*dim_s0},
                    'xticks': False,
                },
            ],
            
            # [
            #     {
            #         'input': ['goals', 's0'], 'plot': partial(timeseries, marker='.'),
            #         'title': 'Goal and state',
            #         'legend': {'Goal': 0, 'State': dim_s0},
            #         'xticks': False,
            #     },
            # ],
            
            # [
            #     {
            #         'input': ['goals', 'pre'], 'plot': partial(timeseries, marker='.'),
            #         'title': 'Goal and prediction',
            #         'xticks': False,
            #     },
            # ],
                        
            [
                {
                    'input': ['X'], 'plot': partial(timeseries, marker = '.'),
                    'title': 'Model %s input $\mathbf{X}$' % (algo),
                    'legend': {'X_pre_l1': 0, 'X_meas_l0': dim_s0, 'X_err_l0': 2*dim_s0},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['Y'], 'plot': partial(timeseries, marker = '.'),
                    'title': 'Model %s input $\mathbf{Y}$' % (algo),
                    'legend': {'Y_pre_l0': 0},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['hidden'], 'plot': partial(timeseries, marker = '.'),
                    'title': 'Model %s hidden activation $\mathbf{Z}$' % (algo),
                    # FIXME: knn particulars
                    'legend': {'Z_dist': 0, 'E(Z_dist)': 5, 'Z_idx': 6},
                    'xticks': False,
                },
            ],
            
            [
                {
                    'input': ['wo_norm', 'wo_norm_fwd'], 'plot': partial(timeseries, marker = '.'),
                    'title': 'Model %s parameter norm (accumulated adaptation)' % (algo),
                    'legend': {'|W|': 0}
                },
            ],
            ]
        }
    }

"""
sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepsys_steps = 6
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
                    'func': f_meshgrid
                    },
                }),
                
                # sys to sweep
                sweepsys,

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
                    'debug': False, # True,
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
    #             'meshgrid':     {'bus': 'sweepsys_grid/meshgrid', 'shape': (dim_s0, sweepsys_input_flat)},
    #             's0':    {'bus': 'robot0/s0', 'shape': (dim_s0, sweepsys_input_flat)},
    #             's1':     {'bus': 'robot0/s1', 'shape': (dim_s1, sweepsys_input_flat)},
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

    # system block from definition elsewhere
    ('robot1', systemblock),
    
    # learning experiment
    ('brain_learn_proprio', {
        'block': Block2,
        'params': {
            'graph': OrderedDict([

                ('cnt', {
                    'block': CountBlock2,
                    'params': {
                        'blocksize': 1,
                        'debug': False,
                        'inputs': {},
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
                
                # dev model imol: learner is basic imol inverse predictor
                #     in proprio space
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'debug': False,
                        # 'debug_trace_callgraph': True,
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
                                'shape': (dim_s0, maxlag), 'lag': range(lag_future[0], lag_future[1])
                            },
                            'pre_l0_fwd': {
                                'bus': 'pre_l0/pre_fwd',
                                'shape': (dim_s0 * (lag_future[1] - lag_future[0]), maxlag),
                                'lag': range(lag_past[0] + 1, lag_past[1] + 1),
                            },
                            'prerr_l0_fwd': {
                                'bus': 'pre_l0/prerr_fwd',
                                'shape': (dim_s0 * (lag_future[1] - lag_future[0]), maxlag),
                                'lag': range(lag_past[0] + 1, lag_past[1] + 1),
                            },
                        },
                        'outputs': {
                            # legacy
                            'pre': {'shape': (dim_s0, 1)},
                            'err': {'shape': (dim_s0, 1)},
                            'prerr_avg': {'shape': (dim_s0, 1)},
                            'prerr_rms_avg': {'shape': (1, 1)},
                            'tgt': {'shape': (dim_s0, 1)},
                            'X': {'shape': (dim_s0 * (lag_past[1] - lag_past[0]) * 3, 1)},
                            'Y': {'shape': (dim_s0 * (lag_future[1] - lag_future[0]), 1)},
                            'hidden': {'shape': (dim_s_hidden_debug, 1)},
                            'wo_norm': {'shape': (dim_s0, 1)},
                            # inv out
                            'pre_inv': {'shape': (dim_s0, 1)},
                            'err_inv': {'shape': (dim_s0, 1)},
                            'prerr_avg_inv': {'shape': (dim_s0, 1)},
                            'prerr_rms_avg_inv': {'shape': (1, 1)},
                            'wo_norm_inv': {'shape': (dim_s0, 1)},
                            # fwd out
                            'pre_fwd': {'shape': (dim_s0 * (lag_future[1] - lag_future[0]), 1)},
                            'prerr_fwd': {'shape': (dim_s0 * (lag_future[1] - lag_future[0]), 1)},
                            'prerr_avg_fwd': {'shape': (dim_s0, 1)},
                            'prerr_rms_avg_fwd': {'shape': (1, 1)},
                            'wo_norm_fwd': {'shape': (dim_s0, 1)},
                        },
                        
                        'models': {
                            'imol': {
                                # imol model, requires 'fwd', 'inv' full model subconfs
                                'type': 'imol',
                                'algo': algo,
                                
                                # forward config
                                'fwd': {
                                    'type': 'imol',
                                    'algo': algo_fwd,
                                    # tapping X_fit
                                    'tap_X_fit_vars': ['pre_l0', 'meas_l0', 'prerr_l0'], # 'prerr_l0_fwd'],
                                    'tap_X_fit_taps': ['lag_past'] * 3,
                                    'tap_X_fit_offs': [1, 0, 1],
                                    'tap_X_fit_srcs': ['inputs'] * 3,
                                    # tapping Y_fit
                                    'tap_Y_fit_vars': ['meas_l0'],
                                    'tap_Y_fit_taps': ['lag_future'],
                                    'tap_Y_fit_offs': [0],
                                    'tap_Y_fit_srcs': ['inputs'],
                                    # tapping X_pre
                                    'tap_X_pre_vars': ['pre_l0_inv', 'meas_l0', 'prerr_l0_inv'], # 'prerr_l0_fwd'],
                                    'tap_X_pre_taps': ['lag_past'] * 3,
                                    # 'tap_X_pre_offs': [lag_off_f2p - 1] + [lag_off_f2p] * 2,
                                    'tap_X_pre_offs': [lag_off_f2p] * 3,
                                    'tap_X_pre_srcs': ['attr', 'inputs', 'attr'],
                                    # legacy lag
                                    'lag_past': lag_past,
                                    'lag_future': lag_future,
                                    'idim': dim_s0 * (lag_past[1] - lag_past[0]) * 3, # laglen
                                    'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                                    'laglen': laglen,
                                    # 'laglen_past': lag_past[1] - lag_past[0],
                                    # 'laglen_future': lag_future[1] - lag_future[0],
                                    'eta': eta,
                                    'modelsize': modelsize_fwd,
                                    # learning modulation
                                    'n_washout': lconf['lm']['n_washout'],
                                    'n_train': lconf['lm']['n_train'],
                                    'n_test': lconf['lm']['n_test'],
                                },
                                
                                # inverse config
                                'inv': {
                                    'type': 'imol',
                                    'algo': algo_inv,
                                    # tapping X_fit
                                    # 'tap_X_fit_vars': [('pre_l1', 'meas_l0'), 'meas_l0', 'prerr_l0'],
                                    # 'tap_X_fit_offs': [lag_off_f2p, 0, 1],
                                    # 'tap_X_fit_srcs': ['inputs'] * 3,
                                    'tap_X_fit_taps': ['lag_past'] * 3,
                                    'tap_X_fit_vars': [('pre_l1', 'meas_l0'), 'meas_l0', 'prerr_l0_invfit'],
                                    'tap_X_fit_offs': [lag_off_f2p, 0, lag_off_f2p],
                                    'tap_X_fit_srcs': ['inputs'] * 2 + ['attr'],
                                    # tapping Y_fit
                                    'tap_Y_fit_vars': ['pre_l0'],
                                    'tap_Y_fit_taps': ['lag_future'],
                                    'tap_Y_fit_offs': [-lag_off_f2p + 1],
                                    'tap_Y_fit_srcs': ['inputs'],
                                    # tapping X_pre
                                    'tap_X_pre_vars': ['pre_l1', 'meas_l0', 'prerr_l0_%s' % 'inv'],
                                    'tap_X_pre_taps': ['lag_past'] * 3,
                                    'tap_X_pre_offs': [lag_off_f2p] * 3,
                                    'tap_X_pre_srcs': ['inputs', 'inputs', 'attr'],
                                    # legacy lag
                                    'lag_past': lag_past,
                                    'lag_future': lag_future,
                                    'idim': dim_s0 * (lag_past[1] - lag_past[0]) * 3, # laglen
                                    'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                                    'laglen': laglen,
                                    # 'laglen_past': lag_past[1] - lag_past[0],
                                    # 'laglen_future': lag_future[1] - lag_future[0],
                                    'eta': eta,
                                    'memory': maxlag,
                                    'w_input': 1.0,
                                    'w_bias': 0.2,
                                    'multitau': False, # True,
                                    'theta': 0.01,
                                    # soesgp
                                    'modelsize': modelsize_inv,
                                    # 'spectral_radius': 1.5, # 0.01,
                                    # FORCE / pm
                                    # 'modelsize': 300,
                                    'theta_state': 0.1,
                                    'lrname': 'FORCE',
                                    'alpha': 50.0, # 10.0,
                                    'spectral_radius': 0.99, # 0.01,
                                    'tau': 0.85, # 0.8, # 0.05, # 1.0,
                                    # 'tau': 0.4, # 0.8, # 0.05, # 1.0,
                                    'wgt_thr': 2.3,
                                    # RLS / barrel
                                    # 'modelsize': 322,
                                    # 'theta_state': 0.1, # for RLS
                                    # 'lrname': 'RLS',
                                    # 'spectral_radius': 0.999, # 0.01,
                                    # 'tau': 0.08, # 0.8, # 0.05, # 1.0,
                                    # 'wgt_thr': 0.5,
                                    'mixcomps': 3,
                                    'oversampling': 1,
                                    'visualize': False,
                                    # learning modulation
                                    'n_washout': lconf['lm']['n_washout'],
                                    'n_train': lconf['lm']['n_train'],
                                    'n_test': lconf['lm']['n_test'],
                                }
                            },
                        },
                        'rate': 1,
                    },
                }),

                # dev model uniform random sampler baseline / comparison
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'inputs': {                        
                #             'lo': {'val': np.array([m_mins]).T, 'shape': (dim_s0, 1)},
                #             'hi': {'val': np.array([m_maxs]).T, 'shape': (dim_s0, 1)},
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
    
])
