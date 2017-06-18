"""actinf_m1_goal_error_pm1d.py

smp_graphs config for experiment

active inference
model type 1 (goal error)
pointmass 1D system

from actinf/active_inference_basic.py --mode m1_goal_error_1d

Model variant M1, 1-dimensional data, proprioceptive space

"""

import copy

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl

# execution
saveplot = True
recurrent = True
debug = False
showplot = True

# experiment
randseed = 12345
numsteps = 1000
dim = 1
motors = dim
dt = 0.1
loopblocksize = numsteps
algo = 'soesgp' # 'knn', 'soesgp', 'storkgp'

"""system block
 - a robot
"""
systemblock = {
    'block': PointmassBlock2,
    'params': {
        'id': 'robot1',
        'blocksize': 1, # FIXME: make pm blocksize aware!
        'sysdim': motors,
        # initial state
        'x0': np.random.uniform(-0.3, 0.3, (motors * 3, 1)),
        # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
        'inputs': {'u': {'bus': 'pre_l0/pre'}},
        'outputs': {
            's_proprio': {'shape': (1, 1)},
            's_extero': {'shape': (1, 1)}
            }, # , 's_all': [(9, 1)]},
        # "class": PointmassRobot2, # SimpleRandomRobot,
        # "type": "explauto",
        # "name": make_robot_name(expr_id, "pm", 0),
        # "numsteps": numsteps,
        # "control": "force",
        # "ros": False,
        "statedim": motors * 3,
        "dt": dt,
        "mass": 1.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        'debug': False,
        }
    }

def plot_timeseries_block(l0 = 'pre_l0', l1 = "pre_l1", blocksize = 1):
    global PlotBlock2, dim, numsteps, timeseries
    return {
    'block': PlotBlock2,
    'params': {
        'blocksize': blocksize,
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim, blocksize)},
            },
        'hspace': 0.2,
        'subplots': [
            [
                {'input': ['goals'], 'plot': timeseries},
            ],
            [
                {'input': ['pre'], 'plot': timeseries},
            ],
            [
                {'input': ['err'], 'plot': timeseries},
            ],
            [
                {'input': ['tgt'], 'plot': timeseries},
            ],
            ]
        }
    }

"""
sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepsys_steps = 21
sweepsys_input_flat = np.power(sweepsys_steps, dim)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s_proprio']['shape'] = (dim, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s_extero']['shape']  = (dim, sweepsys_input_flat)

sweepmdl_steps = 21
sweepmdl_input_flat = np.power(sweepmdl_steps, dim * 2)

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
        'outputs': {'meshgrid': {'shape': (dim, sweepsys_input_flat), 'buscopy': 'sweepsys_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]])},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim, sweepsys_input_flat)}},
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
                'shape': (dim * 2, sweepmdl_input_flat),
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
                        'ranges': {'val': np.array([[-1, 1], [-1e-0, 1e-0]])},
                        'steps':  {'val': sweepmdl_steps},
                        },
                    'outputs': {
                        'meshgrid': {
                            'shape': (
                                dim * 2,
                                sweepmdl_input_flat,)}},
                    'func': f_meshgrid,
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
            #         'outputs': {'pre': {'shape': (dim, sweepmdl_input_flat)}},
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
            #         'outputs': {'pre': {'shape': (dim, sweepmdl_input_flat)}},
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
                            'shape': (dim * 2, sweepmdl_input_flat)}},
                    'slices': {
                        'x': {
                            'goals': slice(0, dim),
                            'errs':  slice(dim, dim*2)},
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
                    'inputs': {
                        # descending prediction
                        'pre_l1': {
                            'bus': 'sweep_slice/x_goals',
                            'shape': (dim,sweepmdl_input_flat)},
                        # ascending prediction error
                        'pre_l0': {
                            'bus': 'sweep_slice/x_errs',
                            'shape': (dim, sweepmdl_input_flat)},
                        # measurement
                        'meas_l0': {
                            'val': np.array([[-np.inf for i in range(sweepmdl_input_flat)]]),
                            # 'bus': 'robot1/s_proprio',
                            'shape': (dim, sweepmdl_input_flat)}},
                    'outputs': {
                        'pre': {'shape': (dim, sweepmdl_input_flat)},
                        'err': {'shape': (dim, sweepmdl_input_flat)},
                        'tgt': {'shape': (dim, sweepmdl_input_flat)},
                        },
                    'models': {
                        # 'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim * 2, 'odim': dim},
                        'fwd': {
                            'type': 'actinf_m1', 'algo': 'copy',
                            'copyid': 'pre_l0', 'idim': dim * 2, 'odim': dim},
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
                        #     'shape': (dim, sweepmdl_input_flat),
                        #     },
                        # 'errs':  {
                        #     'bus': 'sweepmdl_grid_err/pre',
                        #     'shape': (dim, sweepmdl_input_flat),
                        #     },
                        'goalerrs': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim * 2, sweepmdl_input_flat),
                            },
                        'pres':  {
                            'bus': 'pre_l0_test/pre',
                            'shape': (dim, sweepmdl_input_flat),
                            },
                        },
                    'outputs': {
                        'y': {'shape': (dim * 3, sweepmdl_input_flat)}},
                    }
                }),
                
            # OBSOLETE: old cloning approach, didn't work
            # # model sweep model
            # ('test_pre_l0', {
            #     'block': Block2,
            #     'params': {
            #         'id': 'test_pre_l0',
            #         'numsteps':  sweepmdl_input_flat, # np.power(21, dim),
            #         'blocksize': sweepmdl_input_flat, # numsteps,
            #         'blockphase': [0], # [i * numsteps/2 for i in range(2)],
            #         'inputs': {
            #             # descending prediction
            #             'pre_l1': {
            #                 'bus': 'sweepmdl_grid_goal/pre', 'shape': (dim,sweepmdl_input_flat)},
            #             # ascending prediction error
            #             'pre_l0': {
            #                 'bus': 'sweepmdl_grid_err/pre',  'shape': (dim, sweepmdl_input_flat)},
            #             # measurement
            #             'meas_l0': {
            #                 'val': np.array([[-np.inf for i in range(sweepmdl_input_flat)]]),
            #                 'shape': (dim, 1)}},
            #         'outputs': {
            #             'pre': {'shape': (dim, sweepmdl_input_flat)},
            #             'err': {'shape': (dim, sweepmdl_input_flat)},
            #             'tgt': {'shape': (dim, sweepmdl_input_flat)},
            #             },
            #         'graph': 'id:pre_l0',
            #         }
            #     }),

            # # plot timeseries
            # ('plot_ts',
            #      plot_timeseries_block(
            #          l0 = 'pre_l0_test',
            #          l1 = 'sweepmdl_grid_goal',
            #          blocksize = sweepmdl_input_flat)),
                        
            # plot model sweep 1d
            ('plot_model_sweep', {
                'block': ImgPlotBlock2,
                'params': {
                    'id': 'plot_model_sweep',
                    'logging': False,
                    'saveplot': saveplot,
                    'debug': True,
                    'blocksize': sweepmdl_input_flat,
                    'inputs': {
                        # 'sweepin_goal': {
                        #     'bus': 'sweepmdl_grid_goal/pre',
                        #     'shape': (dim, sweepmdl_input_flat)},
                        # 'sweepin_err':  {
                        #     'bus': 'sweepmdl_grid_err/pre',
                        #     'shape': (dim, sweepmdl_input_flat)},
                        # 'sweepout_mdl':  {
                        #     'bus': 'pre_l0_test/pre',
                        #     'shape': (dim, sweepmdl_input_flat)},
                        'all': {
                            'bus': 'pre_l0_combined/y',
                            'shape': (dim * 3, sweepmdl_input_flat),
                            },
                        },
                    'outputs': {},
                    'wspace': 0.5,
                    'hspace': 0.5,
                    # with one subplot and reshape
                    'subplots': [
                        [
                            {
                                'input': ['all'],
                                'shape': (sweepmdl_steps, sweepmdl_steps),
                                'ndslice': [(slice(None), slice(2, 3))],
                                'vmin': -1, 'vmax': 1, 'cmap': 'RdGy',
                                }
                            # {
                            #     'input': ['sweepin_goal'],
                            #     # 'ndslice': (slice(None), slice(None), slice(None)),
                            #     'shape': (dim, sweepmdl_input_flat), 'cmap': 'RdGy',
                            #     # 'dimstack': {'x': [2, 1], 'y': [0]},
                            # },
                            # {
                            #     'input': ['sweepin_err'],
                            #     # 'ndslice': (slice(None), slice(None), slice(None)),
                            #     'shape': (dim, sweepmdl_input_flat), 'cmap': 'RdGy',
                            #     # 'dimstack': {'x': [2, 1], 'y': [0]},
                            # },
                            # {
                            #     'input': ['sweepout_mdl'],
                            #     # 'ndslice': (slice(None), slice(None), slice(None)),
                            #     'shape': (dim, sweepmdl_input_flat), 'cmap': 'RdGy',
                            #     # 'dimstack': {'x': [2, 1], 'y': [0]},
                            # },
                        ],
                        ],
                    },
                }),
            ]),
        }
    }
        

# main graph
graph = OrderedDict([
    # sweep system
    ("sweepsys", {
        'debug': False,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'sweepsys',
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # execution cycle, same as global numsteps
            'blockphase': [1],     # execute on first time step only
            'numsteps':  numsteps,          # numsteps      / loopblocksize = looplength
            'loopblocksize': loopblocksize, # loopblocksize * looplength    = numsteps
            # can't do this dynamically yet without changing init passes
            'outputs': {'meshgrid': {'shape': (dim, sweepsys_input_flat)}},
            'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),

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
                        'inputs': {'x': {'val': 1, 'shape': (1,1)}},
                        'outputs': {'pre': {'shape': (dim, 1)}},
                        'models': {
                            'goal': {'type': 'random_uniform'}
                            },
                        'rate': 50,
                        },
                    }),
                    
                # learner: basic actinf predictor proprio space learn_proprio_base_0
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'inputs': {
                            # descending prediction
                            'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim,1)},
                            # ascending prediction error
                            'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim, 1)},
                            # measurement
                            'meas_l0': {'bus': 'robot1/s_proprio', 'shape': (dim, 1)}},
                        'outputs': {
                            'pre': {'shape': (dim, 1)},
                            'err': {'shape': (dim, 1)},
                            'tgt': {'shape': (dim, 1)},
                            },
                        'models': {
                            'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim * 2, 'odim': dim},
                            },
                        'rate': 1,
                        },
                    }),
                # learn_proprio_e2p2e
                ]),
            }
        }),

    # sequenced model evaluation
    # 1 make a block executing every n steps
    # 2 execute: search a node and clone it
    # 3 execute: run the clone according cloner's internal time (sideways or not)
    
    # # standalone clone test
    # ('test_pre_l0', {
    #     'block': Block2,
    #     'params': {
    #         'id': 'test_pre_l0',
    #         'numsteps': np.power(21, dim),
    #         'blocksize': numsteps,
    #         'blockphase': [i * numsteps/2 for i in range(2)],
    #         'graph': 'id:pre_l0',
    #         }
    #     }),

    ################################################################################
    # use a sequential loop block to insert probes running orthogonally in time
    # blocksize:  seqloops blocksize from the outside, same as overall experiment
    # blockphase: points in the cnt % numsteps space when to execute
    # numsteps:      
    # loopblocksize: number of loop iterations = numsteps/loopblocksize 
    ("sweepmodel", {
        'debug': False,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'sweepmodel',
            'blocksize': numsteps, # execution cycle, same as global numsteps
            #                        execution phase, on first time step only
            # 'blockphase': [numsteps/2, numsteps-10],
            'blockphase': [100, 200, 950],
            # 'blockphase': [0],
            'numsteps':  1, # numsteps,          # numsteps      / loopblocksize = looplength
            'loopblocksize': 1, #loopblocksize, # loopblocksize * looplength    = numsteps
            # can't do this dynamically yet without changing init passes
            'outputs': {'pre': {'shape': (dim * 2, sweepmdl_input_flat)}},
            'loop': [('none', {}) for i in range(2)], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock_model,
        },
    }),
    
    # system block from definition elsewhere
    ('robot1', systemblock),
    
    # plot the system sweep result
    ('plot_sweep', {
        'block': PlotBlock2,
        'params': {
            'debug': False,
            'blocksize': numsteps, # sweepsys_input_flat,
            'title': 'system sweep',
            'inputs': {
                'meshgrid':     {'bus': 'sweepsys_grid/meshgrid', 'shape': (dim, sweepsys_input_flat)},
                's_proprio':    {'bus': 'robot0/s_proprio', 'shape': (dim, sweepsys_input_flat)},
                's_extero':     {'bus': 'robot0/s_extero', 'shape': (dim, sweepsys_input_flat)},
                },
                'hspace': 0.2,
                'subplots': [
                    [
                        {'input': ['meshgrid'], 'plot': timeseries},
                    ],
                    [
                        {'input': ['s_proprio'], 'plot': timeseries},
                    ],
                    [
                        {'input': ['s_extero'], 'plot': timeseries},
                    ],
                    ],
            }
        }),
              
    # plot timeseries
    ('plot_ts', plot_timeseries_block(l0 = 'pre_l0', blocksize = numsteps)),
    
])
