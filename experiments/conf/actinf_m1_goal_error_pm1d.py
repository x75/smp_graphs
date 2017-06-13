"""actinf_m1_goal_error_pm1d.py

smp_graphs config for experiment

active inference
model type 1 (goal error)
pointmass 1D system

from actinf/active_inference_basic.py --mode m1_goal_error_1d

Model variant M1, 1-dimensional data, proprioceptive space

"""


from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2

# execution
saveplot = True
recurrent = True
debug = False
showplot = True

# experiment
randseed = 12345
numsteps = 4000
dim = 1
motors = dim
dt = 0.1

graph = OrderedDict([
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
                        'outputs': {'x_p_pre_l1': {'shape': (dim, 1)}},
                        'models': {
                            'goal': {'type': 'random_uniform'}
                            },
                        'rate': 50,
                        },
                    }),
                    
                # learner basic actinf predictor proprio space learn_proprio_base_0
                ('pre_l0_proprio', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'inputs': {
                            # descending prediction
                            'pre_l1': {'bus': 'pre_l1/x_p_pre_l1', 'shape': (dim,1)},
                            # ascending prediction error
                            'pre_l0': {'bus': 'pre_l0_proprio/x_p_pre_l0', 'shape': (dim, 1)},
                            # measurement
                            'meas_l0': {'bus': 'robot1/s_proprio', 'shape': (dim, 1)}},
                        'outputs': {
                            'x_p_pre_l0':   {'shape': (dim, 1)},
                            'x_p_prerr_l0': {'shape': (dim, 1)},
                            'x_p_target': {'shape': (dim, 1)},
                            },
                        'models': {
                            'fwd': {'type': 'actinf_m1', 'algo': 'knn', 'idim': dim * 2, 'odim': dim},
                            },
                        'rate': 1,
                        },
                    }),
                # learn_proprio_e2p2e
                ]),
            }
        }),

        # a robot
        ('robot1', {
            'block': PointmassBlock2,
            'params': {
                'id': 'robot1',
                'blocksize': 1, # FIXME: make pm blocksize aware!
                'sysdim': motors,
                # initial state
                'x0': np.random.uniform(-0.3, 0.3, (motors * 3, 1)),
                # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
                'inputs': {'u': {'bus': 'pre_l0_proprio/x_p_pre_l0'}},
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
                "sysnoise": 1e-3,
                'debug': False,
                }
            }),

        # plot timeseries
        ('plot_ts', {
            'block': PlotBlock2,
            'params': {
                'blocksize': numsteps,
                'inputs': {
                    'goals':    {'bus': 'pre_l1/x_p_pre_l1', 'shape': (dim, numsteps)},
                    'pre_l0':   {'bus': 'pre_l0_proprio/x_p_pre_l0', 'shape': (dim, numsteps)},
                    'prerr_l0': {'bus': 'pre_l0_proprio/x_p_prerr_l0', 'shape': (dim, numsteps)},
                    'target_l0': {'bus': 'pre_l0_proprio/x_p_target', 'shape': (dim, numsteps)},
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
                        {'input': ['prerr_l0'], 'plot': timeseries},
                        ],
                    [
                        {'input': ['target_l0'], 'plot': timeseries},
                        ],
                    ]
                }
            
            })
    ])
