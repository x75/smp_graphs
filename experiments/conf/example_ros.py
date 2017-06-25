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

# the ros system blocks
from smp_graphs.block_cls_ros import STDRCircularBlock2, LPZBarrelBlock2, SpheroBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform, f_sin_noise

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = True

# experiment
commandline_args = ['numsteps']
randseed = 12345
numsteps = 10000 # ten seconds

# sys = "stdr"
# dim_s_proprio = 2 # linear, angular
# dim_s_extero = 3  # three sonar rangers
# dt = 0.1

sys = "lpzbarrel"
dim_s_proprio = 2 # linear, angular
dim_s_extero = 1  # three sonar rangers
dt = 0.01

# sys = "sphero"
# dim_s_proprio = 2 # linear, angular
# dim_s_extero = 1  # three sonar rangers
# dt = 0.05

# ROS system using Simple Two-Dimensional Robot Simulator (STDR) with basic circular
# configuration and 3 sonar rangers
def get_systemblock_stdr(dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1):
    global STDRCircularBlock2
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
    return systemblock_stdr

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

def get_systemblock(sys = "stdr", dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1):
    global get_systemblock_stdr, get_systemblock_lpzbarrel, get_systemblock_sphero
    if sys == "stdr":
        return get_systemblock_stdr(
            dim_s_proprio = dim_s_proprio, dim_s_extero = dim_s_extero, dt = dt)
    elif sys == "lpzbarrel":
        return get_systemblock_lpzbarrel(
            dim_s_proprio = dim_s_proprio, dim_s_extero = dim_s_extero, dt = dt)
    elif sys == "sphero":
        return get_systemblock_sphero(
            dim_s_proprio = dim_s_proprio, dim_s_extero = dim_s_extero, dt = dt)

################################################################################
systemblock = get_systemblock(sys = sys, dim_s_proprio = dim_s_proprio, dim_s_extero = dim_s_extero, dt = dt)
    
m_mins = systemblock['params']['m_mins']
m_maxs = systemblock['params']['m_maxs']

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
                'rate': 100,
            },
        }),

    # ('cnt', {
    #     'block': CountBlock2,
    #     'params': {
    #         'blocksize': 1,
    #         'debug': False,
    #         'inputs': {},
    #         'outputs': {'x': {'shape': (2, 1)}},
    #     },
    # }),

    # # a random number generator, mapping const input to hi
    # ('pre_l0', {
    #     'block': FuncBlock2,
    #     'params': {
    #         'id': 'pre_l0',
    #         'outputs': {'pre': {'shape': (dim_s_proprio, 1)}},
    #         'debug': False,
    #         'blocksize': 1,
    #         # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
    #         # recurrent connection
    #         'inputs': {'x': {'bus': 'cnt/x'},
    #                    'f': {'val': np.array([[0.03, 0.07]]).T},
    #                    'sigma': {'val': np.array([[0.1, 0.06]]).T}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
    #         'func': f_sin_noise,
    #     },
    # }),
        
    # ROS robot
    ('robot1', systemblock),

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
