"""actinf_m1_goal_error_ND.py

smp_graphs config for experiment

    active inference
    model type 1 (goal error)
    simple n-dimensioal system (point mass, simple n-joint arm)

from actinf/active_inference_basic.py --mode m1_goal_error_nd

Model variant M1, n-dimensional data, proprioceptive space

Embedding: implement embedding at block boundaries
"""

import copy

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2, BhasimulatedBlock2
from smp_graphs.block_cls_ros import STDRCircularBlock2, LPZBarrelBlock2, SpheroBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform, f_sin_noise

# execution
saveplot = False
recurrent = True
debug = False
showplot = True

# experiment
commandline_args = ['numsteps']
randseed = 12345
numsteps = 1000
# dim = 3 # 2, 1
# dim = 9 # bha

# for pm
dim_s_proprio = 2 # acc two dof
dim_s_extero = 2  # vel two dof
dt = 0.1
lag = 6 # 1
eta = 0.715

# # for sa
# dim_s_proprio = 2 # two joint angles
# dim_s_extero = 2  # 2D end-effector
# dt = 0.1
# lag = 1
# eta = 0.715

# # for bha
# dim_s_proprio = 9 # nine actuated strings
# dim_s_extero =  3 # 3D end-effector
# dt = 0.1
# lag = 1
# eta = 0.715

# # for stdr
# dim_s_proprio = 2 # linear, angular (twist)
# dim_s_extero =  3 # three sonar rangers?
# dt = 0.1
# lag = 2
# eta = 0.715

# # for lpzbarrel
# dim_s_proprio = 2 # two weight positions / strut angles
# dim_s_extero  = 1 # just zero dummy
# dt = 0.025
# lag = 6 # 5, 4
# eta = 0.15

# # for sphero, depending on mode
# dim_s_proprio = 2 # linear, angular
# # dim_s_proprio = 2 # cartesian x, y
# # dim_s_proprio = 2 # two raw motors
# dim_s_extero  = 1 # don't know
# dt = 0.05
# lag = 4 # 2?

# goal func, target freq

dim = dim_s_proprio
motors = dim
loopblocksize = numsteps

"""system block
 - a robot
"""
systemblock_pm = {
    'block': PointmassBlock2,
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
            's_extero':  {'shape': (dim_s_extero, 1)}
            }, # , 's_all': [(9, 1)]},
        # "class": PointmassRobot2, # SimpleRandomRobot,
        # "type": "explauto",
        # "name": make_robot_name(expr_id, "pm", 0),
        # "numsteps": numsteps,
        # "control": "force",
        # "ros": False,
        "statedim": dim_s_proprio * 3,
        "dt": dt,
        "mass": 1.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.01,
        "sysnoise": 5e-2,
        'debug': False,
        'dim_s_proprio': dim_s_proprio,
        'length_ratio': 3./2., # gain curve?
        'm_mins': [-1.] * dim_s_proprio,
        'm_maxs': [ 1.] * dim_s_proprio,
        'dim_s_extero': dim_s_extero,
        }
    }

systemblock_sa = {
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
        "statedim": dim_s_proprio * 3,
        "dt": dt,
        "mass": 1.0/3.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        'debug': False,
        'dim_s_proprio': dim_s_proprio,
        'length_ratio': 3./2.,
        'm_mins': [-1.] * dim_s_proprio,
        'm_maxs': [ 1.] * dim_s_proprio,
        # 's_mins': [-1.00] * 9,
        # 's_maxs': [ 1.00] * 9,
        # 'm_mins': -1,
        # 'm_maxs': 1,
        'dim_s_extero': 2,
        }
    }

systemblock_bha = {
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
        "statedim": dim_s_proprio * 3,
        "dt": dt,
        "mass": 1.0/3.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        'debug': False,
        'dim_s_proprio': dim_s_proprio,
        # 'length_ratio': 3./2.,
        'm_mins': 0.05, # 0.1
        'm_maxs': 0.4,  # 0.3
        'dim_s_extero': 3,
        'numsegs': 3,
        'segradii': np.array([0.1,0.093,0.079]),
        'm_mins': [ 0.10] * dim_s_proprio,
        'm_maxs': [ 0.30] * dim_s_proprio,
        's_mins': [-1.00] * dim_s_proprio, # fixme all sensors
        's_maxs': [ 1.00] * dim_s_proprio,
        'doplot': False,
        }
    }
    
# ROS system STDR
systemblock_stdr = {
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
        "ros": True,
        "dt": dt,
        'm_mins': [-1.] * dim_s_proprio,
        'm_maxs': [ 1.] * dim_s_proprio,
        'dim_s_proprio': dim_s_proprio, 
        'dim_s_extero': dim_s_extero,   
        'outdict': {},
        'smdict': {},
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
                "ros": True,
                "dt": dt,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            }
        }
    return systemblock_lpz

systemblock_lpzbarrel = get_systemblock_lpzbarrel(dt = dt)

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
                "ros": True,
                "dt": dt,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            }
        }
    return systemblock_sphero

systemblock_sphero = get_systemblock_sphero()

################################################################################
# experiment variations
# - algo
# - system
# - system order
# - dimensions
# - number of modalities
    
algo = 'knn' #  'homeokinesis' 'soesgp' 'storkgp'

# systemblock   = systemblock_lpzbarrel
# lag = 6 # 5, 4, 2 # 2 or 3 worked with lpzbarrel, dt = 0.05
systemblock   = systemblock_pm
# lag           = 1

dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
m_mins = systemblock['params']['m_mins']
m_maxs = systemblock['params']['m_maxs']

def plot_timeseries_block(l0 = 'pre_l0', l1 = "pre_l1", blocksize = 1):
    global PlotBlock2, dim, numsteps, timeseries, dim_s_extero, dim_s_proprio
    return {
    'block': PlotBlock2,
    'params': {
        'blocksize': blocksize,
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim_s_proprio, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim_s_proprio, blocksize)},
            's_proprio':    {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, blocksize)},
            's_extero':     {'bus': 'robot1/s_extero',  'shape': (dim_s_extero, blocksize)},
            },
        'hspace': 0.2,
        'subplots': [
            [
                {'input': ['goals', 's_proprio'], 'plot': timeseries},
            ],
            [
                {'input': ['goals', 'pre'], 'plot': timeseries},
            ],
            # [
            #     {'input': ['pre'], 'plot': timeseries},
            # ],
            [
                {'input': ['err',], 'plot': timeseries},
            ],
            # [
            #     {'input': ['tgt'], 'plot': timeseries},
            # ],
            # [
            #     {'input': ['s_proprio', 's_extero'], 'plot': timeseries},
            # ],
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
                    'lag': lag,
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
                            'shape': (dim_s_proprio, lag+1), 'lag': lag},
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
                            'copyid': 'pre_l0', 'idim': dim_s_proprio * 2, 'odim': dim},
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
                # # goal sampler (motivation) sample_discrete_uniform_goal
                # ('pre_l1', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'inputs': {                        
                #             'lo': {'val': np.array([m_mins]).T, 'shape': (dim_s_proprio, 1)},
                #             'hi': {'val': np.array([m_maxs]).T, 'shape': (dim_s_proprio, 1)},
                #             },
                #         'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
                #         'models': {
                #             'goal': {'type': 'random_uniform'}
                #             },
                #         'rate': 25,
                #         },
                #     }),

                ('cnt', {
                    'block': CountBlock2,
                    'params': {
                        'blocksize': 1,
                        'debug': False,
                        'inputs': {},
                        'outputs': {'x': {'shape': (dim_s_proprio, 1)}},
                        },
                    }),

                # a random number generator, mapping const input to hi
                ('pre_l1', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'pre_l1',
                        'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
                        'debug': False,
                        'blocksize': 1,
                        # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                        # recurrent connection
                        'inputs': {
                            'x': {'bus': 'cnt/x'},
                            # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
                            # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, # good with soesgp and eta = 0.7
                            # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                            'f': {'val': np.array([[0.23539]]).T * 1.25 * dt}, # good with soesgp and eta = 0.7
                            # 'f': {'val': np.array([[0.14, 0.14]]).T * 1.0},
                            # 'f': {'val': np.array([[0.82, 0.82]]).T},
                            # 'f': {'val': np.array([[0.745, 0.745]]).T},
                            # 'f': {'val': np.array([[0.7, 0.7]]).T},
                            # 'f': {'val': np.array([[0.65, 0.65]]).T},
                            # 'f': {'val': np.array([[0.39, 0.39]]).T},
                            # 'f': {'val': np.array([[0.37, 0.37]]).T},
                            # 'f': {'val': np.array([[0.325, 0.325]]).T},
                            # 'f': {'val': np.array([[0.31, 0.31]]).T},
                            # 'f': {'val': np.array([[0.19, 0.19]]).T},
                            # 'f': {'val': np.array([[0.18, 0.181]]).T},
                            # 'f': {'val': np.array([[0.171, 0.171]]).T},
                            # 'f': {'val': np.array([[0.161, 0.161]]).T},
                            # 'f': {'val': np.array([[0.151, 0.151]]).T},
                            # 'f': {'val': np.array([[0.141, 0.141]]).T},
                            # stay in place
                            # 'f': {'val': np.array([[0.1, 0.1]]).T},
                            # 'f': {'val': np.array([[0.24, 0.24]]).T},
                            # 'sigma': {'val': np.array([[0.001, 0.002]]).T}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                            'sigma': {'val': np.random.uniform(0, 0.01, (dim_s_proprio, 1))}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                            'func': f_sin_noise,
                        },
                    }),
                
                # dev model actinf_m1: learner is basic actinf predictor proprio space learn_proprio_base_0
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'debug': False,
                        'lag': lag,
                        'eta': eta, # 3.7,
                        'ros': True,
                        # the shift is minlag, the length is the maxlag
                        'inputs': {
                            # descending prediction
                            'pre_l1': {
                                'bus': 'pre_l1/pre',
                                'shape': (dim_s_proprio, lag+1), 'lag': range(-5, -1)}, # lag},
                            # ascending prediction error
                            'pre_l0': {
                                'bus': 'pre_l0/pre',
                                'shape': (dim_s_proprio, lag+1), 'lag': range(-5, -1)}, # lag},
                            # ascending prediction error
                            'prerr_l0': {
                                'bus': 'pre_l0/err',
                                'shape': (dim_s_proprio, lag+1), 'lag': range(-4, 0)}, # lag},
                            # measurement
                            'meas_l0': {
                                'bus': 'robot1/s_proprio',
                                'shape': (dim_s_proprio, lag+1), 'lag': range(-4, 0)}
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                            'err': {'shape': (dim_s_proprio, 1)},
                            'tgt': {'shape': (dim_s_proprio, 1)},
                            },
                        'models': {
                            'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim_s_proprio * 2 * 4, 'odim': dim * 4, 'eta': eta},
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
                    
                # # dev model homeokinesis
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'debug': False,
                #         'lag': lag,
                #         'eta': eta, # 3.7,
                #         'ros': True,
                #         'inputs': {
                #             # descending prediction
                #             'pre_l1': {
                #                 'bus': 'pre_l1/pre',
                #                 'shape': (dim_s_proprio, lag+1), 'lag': lag},
                #             # ascending prediction error
                #             'pre_l0': {
                #                 'bus': 'pre_l0/pre',
                #                 'shape': (dim_s_proprio, lag+1), 'lag': lag},
                #             # ascending prediction error
                #             'prerr_l0': {
                #                 'bus': 'pre_l0/err',
                #                 'shape': (dim_s_proprio, lag+1), 'lag': lag},
                #             # measurement
                #             'meas_l0': {
                #                 'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, lag+1)}},
                #         'outputs': {
                #             'pre': {'shape': (dim_s_proprio, 1)},
                #             'err': {'shape': (dim_s_proprio, 1)},
                #             'tgt': {'shape': (dim_s_proprio, 1)},
                #             },
                #         'models': {
                #             'hk': {'type': 'homeokinesis', 'algo': algo, 'idim': dim_s_proprio, 'odim': dim_s_proprio},
                #             },
                #         'rate': 1,
                #         },
                #     }),
                    
                # learn_proprio_e2p2e
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