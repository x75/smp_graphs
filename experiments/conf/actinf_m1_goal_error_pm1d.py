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

from smp_graphs.funcs import f_meshgrid

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
sweepsteps = 21
algo = 'knn'

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

plot_timeseries_block = {
    'block': PlotBlock2,
    'params': {
        'blocksize': numsteps,
        'inputs': {
            'goals':    {'bus': 'pre_l1/pre_l1', 'shape': (dim, numsteps)},
            'pre_l0':   {'bus': 'pre_l0/pre', 'shape': (dim, numsteps)},
            'err_l0': {'bus': 'pre_l0/err', 'shape': (dim, numsteps)},
            'tgt_l0': {'bus': 'pre_l0/tgt', 'shape': (dim, numsteps)},
            },
            'hspace': 0.2,
            'subplots': [
                [
                    {'input': ['goals'], 'plot': timeseries},
                ],
                [
                    {'input': ['pre_l0'], 'plot': timeseries},
                ],
                [
                    {'input': ['err_l0'], 'plot': timeseries},
                ],
                [
                    {'input': ['tgt_l0'], 'plot': timeseries},
                ],
                ]
        }
    }

# plot_sweep_block =     
    
"""sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepinflat = np.power(sweepsteps, dim)
sweepsystem = ('robot0', copy.deepcopy(systemblock))
sweepsystem[1]['params']['blocksize'] = sweepinflat
sweepsystem[1]['params']['debug'] = True
sweepsystem[1]['params']['inputs'] = {'u': {'bus': 'sweepinf/meshgrid'}}
sweepsystem[1]['params']['outputs']['s_proprio']['shape'] = (dim, sweepinflat)
sweepsystem[1]['params']['outputs']['s_extero']['shape']  = (dim, sweepinflat)

loopblock = {
        'block': Block2,
        'params': {
            'id': 'bhier',
            'debug': False,
            'topblock': False,
            'logging': False,
            'numsteps': sweepinflat,  # inner numsteps when used as loopblock (sideways time)
            'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
            'blockphase': [0],        # phase = 0
            'outputs': {'meshgrid': {'shape': (dim, sweepinflat), 'buscopy': 'sweepinf/meshgrid'}},
            # contains the subgraph specified in this config file
            'graph': OrderedDict([
                # ('sweepin', {
                #     'block': sweepBlock2,
                #     'params': {
                #         'ranges': [],
                #         'steps': 21,
                #         'mode': 'lin',
                #         },
                #     }),
                ('sweepinf', {
                    'block': FuncBlock2,
                    'params': {
                        'debug': True,
                        'blocksize': sweepinflat,
                        'inputs': {
                            'ranges': {'val': np.array([[-1, 1]])},
                            'steps':  {'val': sweepsteps},
                            },
                        'outputs': {'meshgrid': {'shape': (dim, sweepinflat)}},
                        'func': f_meshgrid
                        },
                    }),
                # sys to sweep
                sweepsystem,

            ]),
        }
    }


graph = OrderedDict([
    # sweep system
    ("sweepsystem", {
        'debug': True,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'sweepsystem',
            # loop specification, check hierarchical block to completely pass on the contained in/out space?
            'blocksize': numsteps, # execution cycle, same as global numsteps
            'blockphase': [1],     # execute on first time step only
            'numsteps':  numsteps,          # numsteps      / loopblocksize = looplength
            'loopblocksize': loopblocksize, # loopblocksize * looplength    = numsteps
            # can't do this dynamically yet without changing init passes
            'outputs': {'meshgrid': {'shape': (dim, sweepinflat)}},
            # 'loop': [('inputs', {
            #     'lo': {'val': np.random.uniform(-i, 0, (3, 1)), 'shape': (3, 1)}, 'hi': {'val': np.random.uniform(0.1, i, (3, 1)), 'shape': (3, 1)}}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            # 'loop': [('inputs', {'x': {'val': np.random.uniform(np.pi/2, 3*np.pi/2, (3,1))]}) for i in range(1, numsteps+1)],
            # 'loop': partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
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
                        'outputs': {'pre_l1': {'shape': (dim, 1)}},
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
                            'pre_l1': {'bus': 'pre_l1/pre_l1', 'shape': (dim,1)},
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

        # system block from definition elsewhere
        ('robot1', systemblock),
        
        # plot the result
        ('plot_sweep', {
            'block': PlotBlock2,
            'params': {
                'debug': True,
                'blocksize': numsteps, # sweepinflat,
                'title': 'system sweep',
                'inputs': {
                    'meshgrid':     {'bus': 'sweepinf/meshgrid', 'shape': (dim, sweepinflat)},
                    's_proprio':    {'bus': 'robot0/s_proprio', 'shape': (dim, sweepinflat)},
                    's_extero':     {'bus': 'robot0/s_extero', 'shape': (dim, sweepinflat)},
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
        ('plot_ts', plot_timeseries_block),
    ])
