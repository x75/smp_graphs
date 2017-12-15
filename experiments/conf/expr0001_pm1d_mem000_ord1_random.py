"""smp_graphs configuration

baseline behaviour - open-loop uniform random search in finite isotropic space

id:thesis_smpx0001

Oswald Berthold 2017

special case of kinesis with coupling = 0 between measurement and action
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MomentBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/1
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

lconf = {
    'dim': 2,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000,
    'lim': 1.0,
    'order': 1,
}
    
dim = lconf['dim']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0
lag = lconf['lag'] # 1.0
lim_s1 = 0.5
order = lconf['order']

histnormed = False

systemblock   = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = lag, order = order,
    dims = {'s1': {'dim': dim, 'dissipation': 0.03}})

# fo saturation

# diss * velx = motor_max * dt?

systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio

print("sysblock", systemblock['params']['dim_s_proprio'])

# TODO
# 1. loop over randseed
# 2. loop over budget vs. density (space limits, distance threshold), hyperopt
# 3. loop over randseed with fixed optimized parameters
# 4. loop over kinesis variants [bin, cont] and system variants ord [0, 1, 2, 3?] and ndim = [1,2,3,4,8,16,...,ndim_max]

# TODO low-level
# block groups
# experiment sig, make hash, store config and logfile with that hash
# compute experiment hash: if exists, use logfile, else compute
# compute experiment/model_i hash: if exists, use pickled model i, else train
# pimp smp_graphs graph visualisation

# graph
graph = OrderedDict([
    # robot
    ('robot1', systemblock),
        
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
                        'credit': np.ones((1, 1)) * budget,
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {                        
                            'lo': {'val': -lim_s1, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': lim_s1, 'shape': (dim_s_proprio, 1)},
                            # 'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'robot1/s1', 'shape': (dim_s_proprio, 1)},
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
                            'lo': {'val': -lim}, # lim - 1e-3},
                            'hi': {'val':  lim}},
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                            }
                        },
                    }),
            ]),
        }
    }),
    
    # # robot
    # ('robot1', systemblock),

    # measures
    ('measure', {
        'block': MomentBlock2,
        'params': {
            'id': 'measure',
            'blocksize': numsteps/10,
            'inputs': {
                'credit': {'bus': 'pre_l1/credit', 'shape': (1, numsteps)},
            },
            'outputs': {
                'credit_mu': {'shape': (1, 1)},
                'credit_var': {'shape': (1, 1)},
                'credit_min': {'shape': (1, 1)},
                'credit_max': {'shape': (1, 1)},
            },
        },
    }),
    
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
                'credit_l1': {'bus': 'pre_l1/credit', 'shape': (1, numsteps)},
                },
            'hspace': 0.2,
            'wspace': 0.2,
            'desc': 'Baseline agent single episode 1D predictions and budget',
            'subplots': [
                [
                    {
                        'input': ['pre_l0', 's_p', 'pre_l1'],
                        'plot': [
                            partial(timeseries, linewidth = 1.0),
                            timeseries,
                            partial(timeseries, linewidth = 2.0, xticks = False)]},
                    {
                        'input': ['pre_l0', 's_p', 'pre_l1'],
                        'plot': partial(
                            histogram, orientation = 'horizontal', histtype = 'step',
                            xticks = False, yticks = False, normed = histnormed),
                        'mode': 'stack'},
                ],
                [
                    {'input': ['pre_l1', 's_e'], 'plot': partial(timeseries, xticks = False)},
                    {'input': ['pre_l1', 's_e'], 'plot': partial(
                        histogram, orientation = 'horizontal', histtype = 'step',
                        xticks = False, yticks = False, normed = histnormed)},
                ],
                [
                    {'input': 'credit_l1', 'plot': timeseries},
                    {'input': 'credit_l1', 'plot': partial(
                        histogram, orientation = 'horizontal', histtype = 'step',
                        yticks = False, normed = histnormed)},
                ]
            ],
        },
    })
])
