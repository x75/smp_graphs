"""smp_graphs config

kinesis on a 1D point mass system

porting from smq/experiments/conf2/kinesis_pm_1d.py

components for kinesis:
 - world (identity)
 - robot (pointmass)
 - motivation: goal state
 - action: activity modulated proportionally by distance to goal

now: motivation and action are of the same kind (a prediction), but
placed on different levels. can we make the levels very general and
self-organizing?

start with innermost (fundamental drives) and outermost (raw sensors)
layers and start to grow connecting pathways

rename brain to smloop
 
"""

from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block import FuncBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

# reuse
numsteps = 10000
recurrent = True
debug = False
dim = 3
motors = dim
dt = 0.1
showplot = True
randseed = 124

systemblock_pm = {
        'block': PointmassBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': motors,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (motors * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'search/x'}},
            'outputs': {
                's_proprio': {'shape': (motors, 1)},
                's_extero': {'shape': (motors, 1)}
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
            'dim_s_motor': dim,
            'dim_s_extero': dim,
        }
    }

systemblock_sa = {
    'block': SimplearmBlock2,
    'params': {
        'id': 'robot1',
        'blocksize': 1, # FIXME: make pm blocksize aware!
        'sysdim': motors,
        # initial state
        'x0': np.random.uniform(-0.3, 0.3, (dim * 3, 1)),
        # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
        'inputs': {'u': {'bus': 'search/x'}},
        'outputs': {
            's_proprio': {'shape': (dim, 1)},
            's_extero': {'shape': (2, 1)}
            }, # , 's_all': [(9, 1)]},
        "statedim": motors * 3,
        "dt": dt,
        "mass": 1.0/3.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        'debug': False,
        'dim_s_motor': dim,
        'length_ratio': 3./2.,
        'm_mins': -1,
        'm_maxs': 1,
        'dim_s_extero': 2,
        }
    }    

systemblock = systemblock_sa
dim_s_motor  = systemblock['params']['dim_s_motor']
dim_s_extero = systemblock['params']['dim_s_extero']
dim_s_goal   = dim_s_extero

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
                ('mot0', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'mot0',
                        'rate': 500,
                        'inputs': {'lo': {'val': -0.2, 'shape': (1, 1)}, 'hi': {'val': 0.2, 'shape': (1, 1)}},
                        'outputs': {'x': {'shape': (dim_s_goal, 1)}}
                        },
                    }),
                # kinesis, goal detector, output is binary or continuous, on-goal/off-goal, d(goal)
                ('motivation', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'motivation',
                        # 'inputs': {'x': {'bus': 'robot1/s_extero'}, 'x_': {'val': np.random.uniform(-0.05, 0.05, (3,1))}},
                        'inputs': {
                            'x':   {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, 1)},
                            # 'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_motor, 1)},
                            'x_':  {'bus': 'mot0/x', 'shape': (dim_s_goal, 1)},
                            'x__': {'val': dim_s_motor, 'shape': (dim_s_motor, 1)}},
                        'outputs': {
                            'y': {'shape': (dim_s_motor, 1)},
                            'y1': {'shape': (dim_s_motor, 1)},
                            'x_': {'shape': (dim_s_goal, 1)}},
                        # 'func': f_motivation_bin,
                        'func': f_motivation,
                        'debug': False,
                    },
                }),
                # kinesis, search, noise, motor babbling - modulated by goal detector
                ('search', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {'lo': {'bus': 'motivation/y1'}, 'hi': {'bus': 'motivation/y'}},
                        'outputs': {'x': {'shape': (dim_s_motor, 1)}}
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
            'inputs': {
                'd1': {'bus': 'robot1/s_proprio', 'shape': (dim_s_motor, numsteps)},
                'd2': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'd3': {'bus': 'motivation/x_', 'shape': (dim_s_goal, numsteps)},
                'd4': {'bus': 'sin/y', 'shape': (dim_s_motor, numsteps)},
                'd5': {'bus': 'motivation/y', 'shape': (dim_s_motor, numsteps)},
                'd6': {'bus': 'mot0/x', 'shape': (dim_s_goal, numsteps)}
                },
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                    {'input': 'd1', 'plot': histogram},
                ],
                [
                    {'input': ['d2', 'd3'], 'plot': timeseries},
                    {'input': ['d2', 'd3'], 'plot': histogram},
                ],
                # [
                #     {'input': 'd3', 'plot': timeseries},
                #     {'input': 'd3', 'plot': histogram},
                # ],
                [
                    {'input': 'd5', 'plot': timeseries},
                    {'input': 'd5', 'plot': histogram},
                ],
                [
                    {'input': 'd6', 'plot': timeseries},
                    {'input': 'd6', 'plot': histogram},
                ]
            ],
        },
    })
])
