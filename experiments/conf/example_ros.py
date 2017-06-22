"""smp_graphs config

Example usage of a ROS based robot
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2, BhasimulatedBlock2

from smp_graphs.block_cls_ros import STDRCircularBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = True

# experiment
commandline_args = ['numsteps']
randseed = 12345
numsteps = 100 # ten seconds
dt = 0.1
dim_s_proprio = 2 # linear, angular
dim_s_extero = 3  # three sonar rangers

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

m_mins = systemblock_stdr['params']['m_mins']
m_maxs = systemblock_stdr['params']['m_maxs']

graph = OrderedDict([
    # goal sampler (motivation) sample_discrete_uniform_goal unconditioned
    ('pre_l0', {
        'block': ModelBlock2,
        'params': {
            'blocksize': 1,
            'blockphase': [0],
            'inputs': {                        
                'lo': {'val': np.array([m_mins]).T, 'shape': (dim_s_proprio, 1)},
                'hi': {'val': np.array([m_maxs]).T, 'shape': (dim_s_proprio, 1)},
                },
            'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
            'models': {
                'goal': {'type': 'random_uniform'}
                },
                'rate': 2,
            },
        }),

    # ROS robot
    ('robot1', systemblock_stdr),

    ('plot_robot1', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'inputs': {
                'pre_l0':    {'bus': 'pre_l0/pre',       'shape': (dim_s_proprio, numsteps)},
                's_proprio': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_extero':  {'bus': 'robot1/s_extero',  'shape': (dim_s_extero,  numsteps)},
                },
                'hspace': 0.2,
                'subplots': [
                    [
                        {'input': ['pre_l0', 's_proprio'], 'plot': timeseries},
                    ],
                    [
                        {'input': ['s_extero'], 'plot': timeseries},
                    ],
                    ],
            },
            })
    ])
