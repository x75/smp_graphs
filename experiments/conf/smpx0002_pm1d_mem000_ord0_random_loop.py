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
graph1 = OrderedDict([
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
                    {'input': ['pre_l0', 's_p', 'pre_l1'], 'plot': [timeseries, partial(timeseries, linewidth = 1.0), timeseries]},
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


# loopblock = {
#     'block': Block2,
#     'params': {
#         'id': 'bhier',
#         'debug': False,
#         'topblock': False,
#         'logging': False,
#         'numsteps': sweepsys_input_flat,  # inner numsteps when used as loopblock (sideways time)
#         'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
#         'blockphase': [0],        # phase = 0
#         'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat), 'buscopy': 'sweepsys_grid/meshgrid'}},
#         # subgraph
#         'graph': OrderedDict([
#             ('sweepsys_grid', {
#                 'block': FuncBlock2,
#                 'params': {
#                     'debug': False,
#                     'blocksize': sweepsys_input_flat,
#                     'inputs': {
#                         'ranges': {'val': np.array([[-1, 1]] * dim_s_proprio)},
#                         'steps':  {'val': sweepsys_steps},
#                         },
#                     'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
#                     'func': f_meshgrid
#                     },
#                 }),
                
#                 # sys to sweep
#                 sweepsys,

#             ]),
#         }
#     }

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'numsteps': numsteps,
        'subgraph_rewrite_id': True,
        # contains the subgraph specified in this config file
        'subgraph': 'conf/smpx0001_pm1d_mem000_ord0_random.py'
        # 'graph': graph1,
    },
}

graph = OrderedDict([
    # # concurrent loop
    # ('b3', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'b3',
    #         'logging': False,
    #         'loop': [('randseed', 1000 + i) for i in range(1, 4)],
    #         'loopmode': 'parallel',
    #         'numsteps': numsteps,
    #         # graph dictionary: (id-key, {config dict})
    #         'loopblock': loopblock,
    #         # 'outputs': {'x': {'shape': (3, 1)}},
    #     },
    # }),

    # sequential loop
    ("b4", {
        'block': SeqLoopBlock2,
        'params': {
            'id': 'b4',
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'numsteps':  numsteps,
            'loopblocksize': numsteps/3, # loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {
                # 'x': {'shape': (3, numsteps)},
                # 'y': {'shape': (1, numsteps)}
            },
            # 'loop': [('inputs', {
            #     'lo': {'val': np.random.uniform(-i, 0, (3, 1)), 'shape': (3, 1)}, 'hi': {'val': np.random.uniform(0.1, i, (3, 1)), 'shape': (3, 1)}}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            # 'loop': [('inputs', {'x': {'val': np.random.uniform(np.pi/2, 3*np.pi/2, (3,1))]}) for i in range(1, numsteps+1)],
            'loop': [('randseed', 1000 + i) for i in range(0, 3)], # partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),
    
])
