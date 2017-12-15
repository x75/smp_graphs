"""dm_imol (developmental model using internal model online learning

smp_graphs config for experiment

    imol

from smp/imol 

Embedding: implement embedding at block boundaries
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

# robustness
# fix: target properties like frequency matched to body (attainable / not attainable)
# fix: eta
# fix: motor / sensor limits, how does the model learn the limits

# priors: timing, limits

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = False

# experiment
commandline_args = ['numsteps']
randseed = 12355
numsteps = int(10000/10)
loopblocksize = numsteps
sysname = 'pm'
# sysname = 'sa'
# sysname = 'bha'
# sysname = 'stdr'
# sysname = 'lpzbarrel'
# sysname = 'sphero'
# dim = 3 # 2, 1
# dim = 9 # bha

"""system block
 - a robot
"""
def get_systemblock_pm(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1):
    global np, PointmassBlock2, meas
    return {
        'block': PointmassBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'systype': 2,
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero, 1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'mass': 1.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.01,
            'sysnoise': 1e-2,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            'length_ratio': 3./2., # gain curve?
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            'dim_s_extero': dim_s_extero,
            'lag': 3,
            'order': 2,
            'coupling_sigma': 1e-3,
            'transfer': 0,
            'anoise_mean': 0.0,
            'anoise_std': 1e-2,
            }
        }

def get_systemblock_sa(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1):
    global np, SimplearmBlock2
    return {
        'block': SimplearmBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero,  1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'mass': 1.0/3.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.001,
            'sysnoise': 1e-2,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            'length_ratio': 3./2.0,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            # 's_mins': [-1.00] * 9,
            # 's_maxs': [ 1.00] * 9,
            # 'm_mins': -1,
            # 'm_maxs': 1,
            'dim_s_extero': dim_s_extero,
            'minlag': 1,
            'maxlag': 2, # 5
            }
        }

def get_systemblock_bha(dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1):
    global np, BhasimulatedBlock2
    return {
        'block': BhasimulatedBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero,  1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'mass': 1.0/3.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.001,
            'sysnoise': 1e-2,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            # 'length_ratio': 3./2.,
            # 'm_mins': 0.05, # 0.1
            # 'm_maxs': 0.4,  # 0.3
            'dim_s_extero': 3,
            'numsegs': 3,
            'segradii': np.array([0.1,0.093,0.079]),
            'm_mins': [ 0.10] * dim_s_proprio,
            'm_maxs': [ 0.30] * dim_s_proprio,
            's_mins': [ 0.10] * dim_s_proprio, # fixme all sensors
            's_maxs': [ 0.30] * dim_s_proprio,
            'doplot': False,
            'minlag': 1,
            'maxlag': 3 # 5
            }
        }
    
# ROS system STDR
def get_systemblock_stdr(dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1):
    global np, STDRCircularBlock2
    return {
        'block': STDRCircularBlock2,
        'params': {
            'id': 'robot1',
            'debug': False,
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero': {'shape': (dim_s_extero, 1)}
                }, # , 's_all': [(9, 1)]},
            'ros': True,
            'dt': dt,
            'm_mins': [-0.3, 0], # [-0.1] * dim_s_proprio,
            'm_maxs': [0.3, np.pi/4.0],    # [ 0.1] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            'minlag': 1, # ha
            'maxlag': 4, # 5
            }
        }

# ROS system using lpzrobots' roscontroller to interact with the 'Barrel'
def get_systemblock_lpzbarrel(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.01):
    global LPZBarrelBlock2
    systemblock_lpz = {
        'block': LPZBarrelBlock2,
        'params': {
            'id': 'robot1',
            'debug': False,
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero': {'shape': (dim_s_extero, 1)}
                }, # , 's_all': [(9, 1)]},
            'ros': True,
            'dt': dt,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            'minlag': 2, # 1, # 5, 4
            'maxlag': 6, # 2,
            }
        }
    return systemblock_lpz

# systemblock_lpzbarrel = get_systemblock_lpzbarrel(dt = dt)

# ROS system using the Sphero
def get_systemblock_sphero(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.05):
    global SpheroBlock2
    systemblock_sphero = {
        'block': SpheroBlock2,
        'params': {
            'id': 'robot1',
            'debug': False,
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero': {'shape': (dim_s_extero, 1)}
                }, # , 's_all': [(9, 1)]},
                'ros': True,
                'dt': dt,
            'm_mins': [-0.75] * dim_s_proprio,
            'm_maxs': [ 0.75] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            'minlag': 2, # 2, # 4, # 2
            'maxlag': 5,
            }
        }
    return systemblock_sphero

# systemblock_sphero = get_systemblock_sphero()

get_systemblock = {
    'pm': partial(get_systemblock_pm, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
    'sa': partial(get_systemblock_sa, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
    'bha': partial(get_systemblock_bha, dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1),
    'lpzbarrel': partial(get_systemblock_lpzbarrel, dim_s_proprio = 2, dim_s_extero = 1, dt = 1.0/12.5), # 2.0/92.0), # 0.025),
    'stdr': partial(get_systemblock_stdr, dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1),
    'sphero': partial(get_systemblock_sphero, dim_s_proprio = 2, dim_s_extero = 1, dt = 0.0167),
    }
    

################################################################################
# experiment variations
# - algo
# - system
# - system order
# - dimensions
# - number of modalities
    
# systemblock   = systemblock_lpzbarrel
# lag = 6 # 5, 4, 2 # 2 or 3 worked with lpzbarrel, dt = 0.05
systemblock   = get_systemblock[sysname]()
# lag           = 1

dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
dim_s_hidden_debug = 20
m_mins = np.array([systemblock['params']['m_mins']]).T
m_maxs = np.array([systemblock['params']['m_maxs']]).T

minlag = systemblock['params']['minlag']
maxlag = systemblock['params']['maxlag']

dt = systemblock['params']['dt']

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
# lag_past   = (-1, 0)
# lag_past   = (-2, -1)
# lag_past   = (-3, -2)
lag_past   = (-4, -3)
# lag_past   = (-5, -4)
# lag_past   = (-6, -5)
lag_future = (-1, 0)

# lag_past = (-11, -10)
# lag_future = (-5, 0)
# lag_past = (-4, -3)
# lag_future = (-1, 0)

# # lpzbarrel non-overlapping seems important
# lag_past = (-6, -2)
# lag_future = (-1, 0)

minlag = 1
maxlag = 20 # -lag_past[0] + lag_future[1]
laglen = maxlag - minlag

# eta = 0.99
# eta = 0.95
# eta = 0.7
# eta = 0.3
# eta = 0.25
eta = 0.15
# eta = 0.1
# eta = 0.05

def plot_timeseries_block(l0 = 'pre_l0', l1 = 'pre_l1', blocksize = 1):
    global partial
    global PlotBlock2, numsteps, timeseries, dim_s_extero, dim_s_proprio, dim_s_hidden_debug, lag_past, lag_future
    return {
    'block': PlotBlock2,
    'params': {
        'blocksize': min(numsteps, 1000), # blocksize, # numsteps
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim_s_proprio, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            's_proprio':    {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, blocksize)},
            's_extero':     {'bus': 'robot1/s_extero',  'shape': (dim_s_extero, blocksize)},
            'X': {'bus': 'pre_l0/X', 'shape': (dim_s_proprio * (lag_past[1] - lag_past[0]) * 3, blocksize)},
            'Y': {'bus': 'pre_l0/Y',  'shape': (dim_s_proprio * (lag_future[1] - lag_future[0]), blocksize)},
            'hidden': {'bus': 'pre_l0/hidden',  'shape': (dim_s_hidden_debug, blocksize)},
            'wo_norm': {'bus': 'pre_l0/wo_norm',  'shape': (1, blocksize)},
            },
        'hspace': 0.2,
        'subplots': [
            [
                {'input': ['goals', 's_proprio'], 'plot': partial(timeseries, marker='.')},
            ],
            [
                {'input': ['goals', 'pre'], 'plot': partial(timeseries, marker='.')},
            ],
            # [
            #     {'input': ['s_proprio'], 'plot': partial(timeseries, marker='.')},
            # ],
            # [
            #     {'input': ['pre'], 'plot': partial(timeseries, marker='.')},
            # ],
            # [
            #     {'input': ['pre'], 'plot': timeseries},
            # ],
            [
                {'input': ['err', 'tgt',], 'plot': partial(timeseries, marker='.')},
            ],
            # [
            #     {'input': ['tgt'], 'plot': partial(timeseries, marker='.')},
            # ],
            [
                {'input': ['X'], 'plot': partial(timeseries, marker = '.')},
            ],
            [
                {'input': ['Y'], 'plot': partial(timeseries, marker = '.')},
            ],
            [
                {'input': ['hidden'], 'plot': partial(timeseries, marker = '.')},
            ],
            [
                {'input': ['wo_norm'], 'plot': partial(timeseries, marker = '.')},
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
sweepsys_input_flat = np.power(sweepsys_steps, dim_s_proprio)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s_proprio']['shape'] = (dim_s_proprio, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s_extero']['shape']  = (dim_s_extero, sweepsys_input_flat)

sweepmdl_steps = 1000
sweepmdl_input_flat = sweepmdl_steps # np.power(sweepmdl_steps, dim_s_proprio * 2)
sweepmdl_func = f_random_uniform

# sweepmdl_steps = 3
# sweepmdl_input_flat = np.power(sweepmdl_steps, dim_s_proprio * 2)
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
        'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat), 'buscopy': 'sweepsys_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s_proprio)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
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
                'shape': (dim_s_proprio * 2, sweepmdl_input_flat),
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
                            'val': np.array([[m_mins[0], m_maxs[0]]] * dim_s_proprio * 2)},
                            # 'val': np.vstack((
                            #     np.array([[m_mins[0], m_maxs[0]]] * dim_s_proprio),
                            #     np.array([[-2.0,      1.0]]       * dim_s_proprio),
                            #     ))},
                        'steps':  {'val': sweepmdl_steps},
                        },
                    'outputs': {
                        'meshgrid': {
                            'shape': (
                                dim_s_proprio * 2,
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
            #         'outputs': {'pre': {'shape': (dim_s_proprio, sweepmdl_input_flat)}},
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
            #         'outputs': {'pre': {'shape': (dim_s_proprio, sweepmdl_input_flat)}},
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
                            'shape': (dim_s_proprio * 2, sweepmdl_input_flat)}},
                    'slices': {
                        'x': {
                            'goals': slice(0, dim_s_proprio),
                            'errs':  slice(dim_s_proprio, dim_s_proprio * 2)},
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
                            'shape': (dim_s_proprio,sweepmdl_input_flat)},
                        # ascending prediction error
                        'pre_l0': {
                            'bus': 'sweep_slice/x_errs',
                            'shape': (dim_s_proprio, sweepmdl_input_flat)},
                        # ascending prediction error
                        'prerr_l0': {
                            'bus': 'pre_l0_test/err',
                            'shape': (dim_s_proprio, minlag+1), 'lag': minlag},
                        # measurement
                        'meas_l0': {
                            'val': np.array([[-np.inf for i in range(sweepmdl_input_flat)]] * dim_s_proprio),
                            # 'bus': 'robot1/s_proprio',
                            'shape': (dim_s_proprio, sweepmdl_input_flat)}},
                    'outputs': {
                        'pre': {'shape': (dim_s_proprio, sweepmdl_input_flat)},
                        'err': {'shape': (dim_s_proprio, sweepmdl_input_flat)},
                        'tgt': {'shape': (dim_s_proprio, sweepmdl_input_flat)},
                        },
                    'models': {
                        # 'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim_s_proprio * 2, 'odim': dim},
                        'fwd': {
                            'type': 'actinf_m1', 'algo': 'copy',
                            'copyid': 'pre_l0', 'idim': dim_s_proprio * 2, 'odim': dim_s_proprio},
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
                        #     'shape': (dim_s_proprio, sweepmdl_input_flat),
                        #     },
                        # 'errs':  {
                        #     'bus': 'sweepmdl_grid_err/pre',
                        #     'shape': (dim_s_proprio, sweepmdl_input_flat),
                        #     },
                        'goalerrs': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim_s_proprio * 2, sweepmdl_input_flat),
                            },
                        'pres':  {
                            'bus': 'pre_l0_test/pre',
                            'shape': (dim_s_proprio, sweepmdl_input_flat),
                            },
                        },
                    'outputs': {
                        'y': {'shape': (dim_s_proprio * 3, sweepmdl_input_flat)}},
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
            #             #     'shape': (dim_s_proprio, sweepmdl_input_flat)},
            #             # 'sweepin_err':  {
            #             #     'bus': 'sweepmdl_grid_err/pre',
            #             #     'shape': (dim_s_proprio, sweepmdl_input_flat)},
            #             # 'sweepout_mdl':  {
            #             #     'bus': 'pre_l0_test/pre',
            #             #     'shape': (dim_s_proprio, sweepmdl_input_flat)},
            #             'all': {
            #                 'bus': 'pre_l0_combined/y',
            #                 'shape': (dim_s_proprio * 3, sweepmdl_input_flat),
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
            #                     'shape': (dim_s_proprio * 3, sweepmdl_input_flat),
            #                     'ndslice': [(slice(None), slice(None))],
            #                     # 'vmin': -1.0, 'vmax': 1.0,
            #                     # 'vmin': 0.1, 'vmax': 0.3,
            #                     'cmap': 'RdGy',
            #                     'dimstack': {
            #                         'x': range(2*dim-1, dim - 1, -1),
            #                         'y': range(dim-1,   -1     , -1)},
            #                     'digitize': {'argdims': range(0, dim_s_proprio * 2), 'valdim': 2*dim+i, 'numbins': 2},
            #                 } for i in range(dim_s_proprio)],

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
    #         'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
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
    #             'meshgrid':     {'bus': 'sweepsys_grid/meshgrid', 'shape': (dim_s_proprio, sweepsys_input_flat)},
    #             's_proprio':    {'bus': 'robot0/s_proprio', 'shape': (dim_s_proprio, sweepsys_input_flat)},
    #             's_extero':     {'bus': 'robot0/s_extero', 'shape': (dim_s_extero, sweepsys_input_flat)},
    #             },
    #             'hspace': 0.2,
    #             'subplots': [
    #                 [
    #                     {'input': ['meshgrid'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s_proprio'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s_extero'], 'plot': timeseries},
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

                # goal sampler (motivation) sample_discrete_uniform_goal
                ('pre_l1', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'ros': ros,
                        'inputs': {                        
                            'lo': {'val': m_mins * 1.0, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': m_maxs * 1.0, 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
                        'models': {
                            'goal': {'type': 'random_uniform'}
                            },
                        'rate': 40,
                        },
                    }),

                # ('cnt', {
                #     'block': CountBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'debug': False,
                #         'inputs': {},
                #         'outputs': {'x': {'shape': (dim_s_proprio, 1)}},
                #         },
                #     }),

                # # a random number generator, mapping const input to hi
                # ('pre_l1', {
                #     'block': FuncBlock2,
                #     'params': {
                #         'id': 'pre_l1',
                #         'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
                #         'debug': False,
                #         'ros': ros,
                #         'blocksize': 1,
                #         # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #         # recurrent connection
                #         'inputs': {
                #             'x': {'bus': 'cnt/x'},
                #             # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
                #             # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.45]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.225]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7

                #             # barrel
                #             # 'f': {'val': np.array([[0.23539]]).T * 10.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 7.23 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 3.2 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 2.9 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 1.25 * dt}, # good with soesgp and eta = 0.7
                            
                #             # 'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, # good with soesgp and eta = 0.7
                #             'f': {'val': np.array([[0.23539]]).T * 0.05 * dt}, # good with soesgp and eta = 0.7
                            
                #             # 'f': {'val': np.array([[0.23539, 0.2348, 0.14]]).T * 1.25 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.14, 0.14]]).T * 1.0},
                #             # 'f': {'val': np.array([[0.82, 0.82]]).T},
                #             # 'f': {'val': np.array([[0.745, 0.745]]).T},
                #             # 'f': {'val': np.array([[0.7, 0.7]]).T},
                #             # 'f': {'val': np.array([[0.65, 0.65]]).T},
                #             # 'f': {'val': np.array([[0.39, 0.39]]).T},
                #             # 'f': {'val': np.array([[0.37, 0.37]]).T},
                #             # 'f': {'val': np.array([[0.325, 0.325]]).T},
                #             # 'f': {'val': np.array([[0.31, 0.31]]).T},
                #             # 'f': {'val': np.array([[0.19, 0.19]]).T},
                #             # 'f': {'val': np.array([[0.18, 0.181]]).T},
                #             # 'f': {'val': np.array([[0.171, 0.171]]).T},
                #             # 'f': {'val': np.array([[0.161, 0.161]]).T},
                #             # 'f': {'val': np.array([[0.151, 0.151]]).T},
                #             # 'f': {'val': np.array([[0.141, 0.141]]).T},
                #             # stay in place
                #             # 'f': {'val': np.array([[0.1, 0.1]]).T},
                #             # 'f': {'val': np.array([[0.24, 0.24]]).T},
                #             # 'sigma': {'val': np.array([[0.001, 0.002]]).T}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #             'sigma': {'val': np.random.uniform(0, 0.01, (dim_s_proprio, 1))},
                #             'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
                #             'amp': {'val': (m_maxs - m_mins)/2.0},
                #         }, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #         'func': f_sin_noise,
                #     },
                # }),
                
                # dev model actinf_m1: learner is basic actinf predictor proprio space learn_proprio_base_0
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'debug': False,
                        'lag': minlag,
                        'eta': eta, # 3.7,
                        'ros': ros,
                        # FIXME: relative shift = minlag, block length the maxlag
                        'inputs': {
                            # descending prediction
                            'pre_l1': {
                                'bus': 'pre_l1/pre',
                                # 'shape': (dim_s_proprio, maxlag), 'lag': range(-maxlag, -minlag)},
                                # FIXME: check correctness here
                                # training on past goal, prediction with current goal
                                # should be possible inside step_model to use current
                                # goal
                                'shape': (dim_s_proprio, maxlag), 'lag': list(range(lag_past[0], lag_past[1]))},
                            # ascending prediction error
                            'pre_l0': {
                                'bus': 'pre_l0/pre',
                                # 'shape': (dim_s_proprio, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s_proprio, maxlag), 'lag': list(range(lag_past[0] + 1, lag_past[1] + 1))},
                            # ascending prediction error
                            'prerr_l0': {
                                'bus': 'pre_l0/err',
                                # 'shape': (dim_s_proprio, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s_proprio, maxlag), 'lag': list(range(lag_past[0] + 1, lag_past[1] + 1))},
                            # measurement
                            'meas_l0': {
                                'bus': 'robot1/s_proprio',
                                # 'shape': (dim_s_proprio, maxlag), 'lag': range(-laglen, 0)}
                                'shape': (dim_s_proprio, maxlag), 'lag': list(range(lag_future[0], lag_future[1]))}
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                            'err': {'shape': (dim_s_proprio, 1)},
                            'tgt': {'shape': (dim_s_proprio, 1)},
                            'X': {'shape': (dim_s_proprio * (lag_past[1] - lag_past[0]) * 3, 1)},
                            'Y': {'shape': (dim_s_proprio * (lag_future[1] - lag_future[0]), 1)},
                            'hidden': {'shape': (dim_s_hidden_debug, 1)},
                            'wo_norm': {'shape': (1, 1)},
                            },
                        'models': {
                            'imol': {
                                'type': 'imol',
                                'algo': algo,
                                'fwd': {
                                    'type': 'imol',
                                    'algo': algo_fwd,
                                    'lag_past': lag_past,
                                    'lag_future': lag_future,
                                    'idim': dim_s_proprio * (lag_past[1] - lag_past[0]) * 2, # laglen
                                    'odim': dim_s_proprio * (lag_future[1] - lag_future[0]), # laglen,
                                    'laglen': laglen,
                                    # 'laglen_past': lag_past[1] - lag_past[0],
                                    # 'laglen_future': lag_future[1] - lag_future[0],
                                    'eta': eta,
                                },
                                'inv': {
                                    'type': 'imol',
                                    'algo': algo_inv,
                                    'lag_past': lag_past,
                                    'lag_future': lag_future,
                                    'idim': dim_s_proprio * (lag_past[1] - lag_past[0]) * 3, # laglen
                                    'odim': dim_s_proprio * (lag_future[1] - lag_future[0]), # laglen,
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
                                    'modelsize': 200,
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
                                }
                            },
                        },
                        'rate': 1,
                    },
                }),

                # dev model uniform random sampler
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'inputs': {                        
                #             'lo': {'val': np.array([m_mins]).T, 'shape': (dim_s_proprio, 1)},
                #             'hi': {'val': np.array([m_maxs]).T, 'shape': (dim_s_proprio, 1)},
                #             },
                #         'outputs': {
                #             'pre': {'shape': (dim_s_proprio, 1)},
                #             'err': {'val': np.zeros((dim_s_proprio, 1)), 'shape': (dim_s_proprio, 1)},
                #             'tgt': {'val': np.zeros((dim_s_proprio, 1)), 'shape': (dim_s_proprio, 1)},
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
    #         'outputs': {'pre': {'shape': (dim_s_proprio * 2, sweepmdl_input_flat)}},
    #         'loop': [('none', {}) for i in range(2)], # lambda ref, i, obj: ('none', {}),
    #         'loopmode': 'sequential',
    #         'loopblock': loopblock_model,
    #     },
    # }),        
        
    # plot timeseries
    ('plot_ts', plot_timeseries_block(l0 = 'pre_l0', blocksize = numsteps)),
    
    ])
