"""smp_graphs config

kinesis on a 1D point mass system

see also smq/experiments/conf2/kinesis_pm_1d.py


"""

from smp_graphs.block_systems import PointmassBlock2

# reuse
numsteps = 1000
debug = False
motors = 3
dt = 0.1
showplot = True

# graph
graph = OrderedDict([
    # a robot
    ('robot1', {
        'block': PointmassBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': numsteps, # FIXME: make pm blocksize aware!
            'sysdim': motors,
            'inputs': {'u': [np.random.uniform(-1, 1, (3, numsteps))]},
            'outputs': {'s_proprio': [(3,1)], 's_extero': [(3,1)], 's_all': [(9, 1)]},
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
            'inputs': {'d1': ['robot1/s_proprio'], 'd2': ['robot1/s_extero'], 'd3': ['robot1/s_all']},
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                    {'input': 'd1', 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'plot': timeseries},
                    {'input': 'd2', 'plot': histogram},
                ],
                [
                    {'input': 'd3', 'plot': timeseries},
                    {'input': 'd3', 'plot': histogram},
                ]
            ],
        },
    })
])
