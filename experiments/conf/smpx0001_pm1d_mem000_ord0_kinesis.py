"""smp_graphs configuration

kinesis on an n-dimensional system

Oswald Berthold 2017

porting from smq/experiments/conf2/kinesis_pm_1d.py

components for kinesis (fom smq):
 - world (identity)
 - robot (pointmass, simplearm)
 - motivation: distance to goal
 - action: activity modulated proportionally by distance to goal

now: motivation and action are of the same kind (a prediction), but
placed on different levels. can we make the levels very general and
self-organizing?

start with innermost (fundamental drives) and outermost (raw sensors)
layers and start to grow connecting pathways
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/1
recurrent = True
debug = False
dim = 1
motors = dim
dt = 0.1
showplot = True
saveplot = True
randseed = 126

from smp_graphs.utils_conf import get_systemblock
from smp_graphs.utils_conf import get_systemblock_pm
from smp_graphs.utils_conf import get_systemblock_sa

systemblock   = get_systemblock['pm'](dim_s_proprio = dim)
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio

# TODO
# 1. loop over randseed
# 2. loop over budget vs. density (space limits, distance threshold), hyperopt
# 3. loop over randseed with fixed optimized parameters
# 4. loop over kinesis variants [bin, cont] and system variants ord [0, 1, 2, 3?] and ndim = [1,2,3,4,8,16,...,ndim_max]

# TODO low-level
# experiment sig, make hash, store config and logfile with that hash
# compute experiment hash: if exists, use logfile, else compute
# compute experiment/model_i hash: if exists, use pickled model i, else train
# pimp smp_graphs graph visualisation

# graph
graph = OrderedDict([
    # brainxs first
    # brain a) write down into graph directly
    ('braina', {
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'braina',
            # open loop sinewave brain ;)
            'graph': OrderedDict([
                # test signal
                # ('sin', {
                #     'block': FuncBlock2,
                #     'params': {
                #         'id': 'sin',
                #         'inputs': {'x': {'bus': 'sin/cnt'}, 'f': {'val': np.array([[0.1, 0.2, 0.3]]).T}},
                #         'outputs': {'cnt': {'shape': (1,1)}, 'y': {'shape': (3,1)}},
                #         'func': f_sin,
                #         'debug': False,
                #     },
                # }),
                # ('sin2', {
                #     'block': FuncBlock2,
                #     'params': {
                #         'id': 'sin2',
                #         'inputs': {'x': {'bus': 'sin2/cnt'}, 'f': {'val': np.array([[0.11, 0.191, 0.313]]).T}},
                #         'outputs': {'cnt': {'shape': (1, 1)}, 'y': {'shape': (3, 1)}},
                #         'func': f_sin, # f_pulse,
                #         'debug': False,
                #     },
                # }),
                
                # more randomness
                ('pre_l1', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'ros': ros,
                        'credit': np.ones((1, 1)) * 100,
                        'inputs': {                        
                            'lo': {'val': -0.2, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': 0.2 * 1.0, 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                            'credit': {'shape': (1,1)}},
                        'models': {
                            'goal': {'type': 'random_uniform_modulated'}
                            },
                        'rate': 1,
                        },
                    }),
                    
                # ('goal', {
                #     'block': UniformRandomBlock2,
                #     'params': {
                #         'id': 'goal',
                #         'rate': numsteps/8, # 500,
                #         'lo': -0.2,
                #         'hi': 0.2,
                #         'inputs': {
                #             'lo': {'val': -0.2, 'shape': (1, 1)},
                #             'hi': {'val': 0.2, 'shape': (1, 1)}
                #         },
                #         'outputs': {
                #             'x': {'shape': (dim_s_goal, 1)}}
                #         },
                #     }),
                # kinesis, goal detector, output is binary or continuous, on-goal/off-goal, d(goal)
                ('motivation', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'motivation',
                        # 'inputs': {'x': {'bus': 'robot1/s_extero'}, 'x_': {'val': np.random.uniform(-0.05, 0.05, (3,1))}},
                        'inputs': {
                            # 'x':   {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, 1)},
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            'x_':  {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, 1)},
                            # 'x__': {'val': dim_s_proprio, 'shape': (dim_s_proprio, 1)}
                            },
                        'outputs': {
                            'y': {'shape': (dim_s_proprio, 1)},
                            'y1': {'shape': (dim_s_proprio, 1)},
                            'x_': {'shape': (dim_s_goal, 1)}},
                        'func': f_motivation_bin,
                        # 'func': f_motivation,
                        'debug': False,
                    },
                }),
                # kinesis, search, noise, motor babbling - modulated by goal detector
                # ('search', {
                ('pre_l0', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {
                            # 'lo': {'bus': 'motivation/y1'},
                            # 'hi': {'bus': 'motivation/y'}},
                            'lo': {'val': -0.2},
                            'hi': {'val': 0.2}},
                        'outputs': {
                            # 'x': {'shape': (dim_s_proprio, 1)},
                            'pre': {'shape': (dim_s_proprio, 1)},
                            }
                        },
                    }),
            ]),
            # 'outputs': {'cnt': {'shape': (1, 1)}, 'y': {'shape': (3, 1)}}
        }
    }),
    
    # brain b) include subgraph config
    # ('brainb': {}),
    
    # # brain c) primblock
    # ('brainc', {
    #     'block': PredBrainBlock2,
    #     'params': {
    #         'id': 'brainc',
            
    #         },
    # }),
        
    # a robot
    ('robot1', systemblock),
    
    # plotting
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'blocksize': numsteps,
            'saveplot': saveplot,
            'inputs': {
                's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'goal': {'bus': 'motivation/x_', 'shape': (dim_s_goal, numsteps)},
                'd4': {'bus': 'sin/y', 'shape': (dim_s_proprio, numsteps)},
                'd5': {'bus': 'motivation/y', 'shape': (dim_s_proprio, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1_credit': {'bus': 'pre_l1/credit', 'shape': (dim_s_goal, numsteps)},
                },
            'subplots': [
                [
                    {'input': ['s_p', 'goal'], 'plot': timeseries},
                    {'input': ['s_p', 'goal'], 'plot': histogram},
                ],
                # [
                #     {'input': ['s_e', 'goal'], 'plot': timeseries},
                #     {'input': ['s_e', 'goal'], 'plot': histogram},
                # ],
                # [
                #     {'input': 'goal', 'plot': timeseries},
                #     {'input': 'goal', 'plot': histogram},
                # ],
                # [
                #     {'input': 'd5', 'plot': timeseries},
                #     {'input': 'd5', 'plot': histogram},
                # ],
                [
                    {'input': 'pre_l1', 'plot': timeseries},
                    {'input': 'pre_l1', 'plot': histogram},
                ],
                [
                    {'input': 'pre_l1_credit', 'plot': timeseries},
                    {'input': 'pre_l1_credit', 'plot': histogram},
                ]
            ],
        },
    })
])
