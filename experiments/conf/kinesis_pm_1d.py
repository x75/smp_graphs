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

from smp_graphs.block_cls import PointmassBlock2
from smp_graphs.block import FuncBlock2

from smp_graphs.funcs import f_sin, f_motivation

# reuse
numsteps = 1000
debug = False
motors = 3
dt = 0.1
showplot = True

# graph
graph = OrderedDict([
    # brain first
    # brain a) write down into graph directly
    ('braina', {
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'braina',
            # open loop sinewave brain ;)
            'graph': OrderedDict([
                # test signal
                ('sin', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'sin',
                        'inputs': {'x': ['sin/cnt'], 'f': [np.array([[0.1, 0.2, 0.3]]).T]},
                        'outputs': {'cnt': [(1,1)], 'y': [(3,1)]},
                        'func': f_sin,
                        'debug': False,
                    },
                }),
                ('sin2', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'sin2',
                        'inputs': {'x': ['sin2/cnt'], 'f': [np.array([[0.11, 0.191, 0.313]]).T]},
                        'outputs': {'cnt': [(1,1)], 'y': [(3,1)]},
                        'func': f_sin,
                        'debug': False,
                    },
                }),
                # kinesis, goal detector, output is binary or continuous, on-goal/off-goal, d(goal)
                ('motivation', {
                    'block': FuncBlock2,
                    'params': {
                        'id': 'motivation',
                        'inputs': {'x': ['robot1/s_extero'], 'x_': [np.array([[0.11, 0.191, 0.313]]).T]},
                        'outputs': {'y': [(3,1)], 'y1': [(3,1)], 'x_': [(3,1)]},
                        'func': f_motivation,
                        'debug': False,
                    },
                }),
                # kinesis, search, noise, motor babbling - modulated by goal detector
                ('search', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {'lo': ['motivation/y1'], 'hi': ['motivation/y']},
                        'outputs': {'x': [(3,1)]}
                        },
                    })
            ]),
            # 'outputs': {'cnt': [(1,1)], 'y': [(3,1)]}
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
    ('robot1', {
        'block': PointmassBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': motors,
            # 'inputs': {'u': [np.random.uniform(-1, 1, (3, numsteps))]},
            'inputs': {'u': ['search/x']},
            'outputs': {'s_proprio': [(3,1)], 's_extero': [(3,1)]}, # , 's_all': [(9, 1)]},
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
            "friction": 0.01,
            "sysnoise": 1e-3,
            'debug': False,
        }
    }),
    # plotting
    ('plot', {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'plot',
            'blocksize': numsteps,
            'inputs': {'d1': ['robot1/s_proprio'], 'd2': ['robot1/s_extero'],
                    'd3': ['motivation/x_'],
                    'd4': ['sin/y']},
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
                    {'input': 'd4', 'plot': timeseries},
                    {'input': 'd4', 'plot': histogram},
                ]
            ],
        },
    })
])
