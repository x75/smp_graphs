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
numsteps = 10000/10
recurrent = True
debug = False
dim = 1
motors = dim
dt = 0.1
showplot = True
saveplot = False
randseed = 126

from smp_graphs.utils_conf import get_systemblock
from smp_graphs.utils_conf import get_systemblock_pm
from smp_graphs.utils_conf import get_systemblock_sa

systemblock   = get_systemblock['pm'](dim_s_proprio = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio

budget = 510
lim = 1.0

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
    # # robot
    # ('robot1', systemblock),
        
    # brain
    ('braina', {
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'braina',
            'graph': OrderedDict([
                # uniformly dist. random goals, triggered when error < goalsize
                ('pre_l1', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'ros': ros,
                        'credit': np.ones((1, 1)) * 510,
                        'goalsize': 0.01, # area of goal
                        'inputs': {                        
                            'lo': {'val': -lim, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': lim, 'shape': (dim_s_proprio, 1)},
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
                    
                # uniformly distributed random action, no modulation
                ('pre_l0', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {
                            'lo': {'val': -lim},
                            'hi': {'val': lim}},
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                            }
                        },
                    }),
            ]),
        }
    }),
    
    # robot
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
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1_credit': {'bus': 'pre_l1/credit', 'shape': (dim_s_goal, numsteps)},
                },
            'subplots': [
                [
                    {'input': ['pre_l0', 's_p', 'pre_l1'], 'plot': [partial(timeseries, linewidth = 1.0), timeseries, timeseries]},
                    {'input': ['pre_l0', 's_p', 'pre_l1'], 'plot': histogram},
                ],
                [
                    {'input': 'pre_l1_credit', 'plot': timeseries},
                    {'input': 'pre_l1_credit', 'plot': histogram},
                ]
            ],
        },
    })
])
