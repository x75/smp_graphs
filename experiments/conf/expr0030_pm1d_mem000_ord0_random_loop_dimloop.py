"""smp_graphs configuration

id:thesis_smpx0002

Baseline statistics for randseed start ... start + numloop of id:thesis_smpx0001

Oswald Berthold 2017

Multiple runs of baseline behaviour varying seed
"""

# from collections import OrderedDict

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/10
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

from smp_graphs.utils_conf import get_systemblock
from smp_graphs.utils_conf import get_systemblock_pm
from smp_graphs.utils_conf import get_systemblock_sa

numloop = 5
numloop_inner = 10

desc = "Experiment expr0030 computes the budget statistics over {0} episodes for sensorimotor dimension $d = {1}$".format(
    numloop_inner,
    list(range(1, numloop + 1)),
)
outputs = {
    'latex': {'type': 'latex',},
}

# dim = 1
# motors = dim
# dt = 0.1

# systemblock   = get_systemblock['pm'](dim_s_proprio = dim, lag = 1)
# systemblock['params']['sysnoise'] = 0.0
# systemblock['params']['anoise_std'] = 0.0
# dim_s_proprio = systemblock['params']['dim_s_proprio']
# dim_s_extero  = systemblock['params']['dim_s_extero']
# # dim_s_goal   = dim_s_extero
# dim_s_goal    = dim_s_proprio

# budget = 510
# lim = 1.0

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

# # graph
# graph1 = OrderedDict([
#     # robot
#     ('robot1', systemblock),
    
#     # brain
#     ('braina', {
#         'block': Block2,
#         'params': {
#             'numsteps': 1, # numsteps,
#             'id': 'braina',
#             'graph': OrderedDict([
#                 # uniformly dist. random goals, triggered when error < goalsize
#                 ('pre_l1', {
#                     'block': ModelBlock2,
#                     'params': {
#                         'blocksize': 1,
#                         'blockphase': [0],
#                         'ros': ros,
#                         'credit': np.ones((1, 1)) * 510,
#                         'goalsize': 0.01, # area of goal
#                         'inputs': {                        
#                             'lo': {'val': -lim, 'shape': (dim_s_proprio, 1)},
#                             'hi': {'val': lim, 'shape': (dim_s_proprio, 1)},
#                             'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
#                             },
#                         'outputs': {
#                             'pre': {'shape': (dim_s_proprio, 1)},
#                             'credit': {'shape': (1,1)}},
#                         'models': {
#                             'goal': {'type': 'random_uniform_modulated'}
#                             },
#                         'rate': 1,
#                         },
#                     }),
                    
#                 # uniformly distributed random action, no modulation
#                 ('pre_l0', {
#                     'block': UniformRandomBlock2,
#                     'params': {
#                         'id': 'search',
#                         'inputs': {
#                             'lo': {'val': -lim},
#                             'hi': {'val': lim}},
#                         'outputs': {
#                             'pre': {'shape': (dim_s_proprio, 1)},
#                             }
#                         },
#                     }),
#             ]),
#         }
#     }),
    
#     # plotting
#     ('plot', {
#         'block': PlotBlock2,
#         'params': {
#             'id': 'plot',
#             'blocksize': numsteps,
#             'saveplot': saveplot,
#             'inputs': {
#                 's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
#                 's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
#                 'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
#                 'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
#                 'pre_l1_credit': {'bus': 'pre_l1/credit', 'shape': (dim_s_goal, numsteps)},
#                 },
#             'subplots': [
#                 [
#                     {'input': ['pre_l0', 's_p', 'pre_l1'], 'plot': [timeseries, partial(timeseries, linewidth = 1.0), timeseries]},
#                     {'input': ['pre_l0', 's_p', 'pre_l1'], 'plot': histogram},
#                 ],
#                 [
#                     {'input': 'pre_l1_credit', 'plot': timeseries},
#                     {'input': 'pre_l1_credit', 'plot': histogram},
#                 ]
#             ],
#         },
#     })
# ])


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
#         'outputs': {
#             'meshgrid': {
#                 'shape': (dim_s_proprio, sweepsys_input_flat),
#                 'buscopy': 'sweepsys_grid/meshgrid'}},
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

# # for stats
# numloop = 5
# loopblock = {
#     'block': Block2,
#     'params': {
#         'id': 'bhier',
#         'debug': False,
#         'logging': False,
#         'topblock': False,
#         'numsteps': numsteps,
#         'randseed': 1,
#         # subcomponent?
#         'robot1/dim': 2,
#         'lconf': {
#             'dim': 2,
#             'dt': 0.1,
#             'lag': 1,
#             'budget': 1000,
#             'lim': 1.0,
#             'order': 0,
#         },
#         # contains the subgraph specified in this config file
#         'subgraph': 'conf/expr0001_pm1d_mem000_ord0_random.py',
#         'subgraph_rewrite_id': True,
#         'subgraph_ignore_nodes': ['plot'],
#         'subgraphconf': {
#             # 'plot/active': False
#             # 'robot1/sysdim': 1,
#             },
#         # 'graph': graph1,
#         'outputs': {
#             'credit_min': {'shape': (1, 1), 'buscopy': 'measure/bcredit_min'},
#             'credit_max': {'shape': (1, 1), 'buscopy': 'measure/bcredit_max'},
#             'credit_mu': {'shape': (1, 1), 'buscopy': 'measure/bcredit_mu'},
#         }
#     },
# }
# loop = [('randseed', 1000 + i) for i in range(0, numloop)]

# for dims
loopblock = {
# loopblock = OrderedDict([('bhier', {
        'block': Block2,
        'params': {
            'id': 'bhier',
            'debug': False,
            'logging': False,
            'topblock': False,
            'numsteps': numsteps,
            'randseed': 1,
            # subcomponent?
            # 'robot1/dim': 2,
            # 'lconf': {
            #     'dim': 2,
            #     'dt': 0.1,
            #     'lag': 1,
            #     'budget': 1000,
            #     'lim': 1.0,
            #     'order': 0
            # },
            'lconf': {},
            # contains the subgraph specified in this config file
            # 'subgraph': 'conf/expr0001_pm1d_mem000_ord0_random.py',
            'subgraph': 'conf/expr0020_pm1d_mem000_ord0_random_loop.py',
            'subgraph_rewrite_id': True,
            'subgraph_ignore_nodes': ['plot'],
            'subgraphconf': {
                # 'plot/active': False
                # 'robot1/sysdim': 1,
            },
            # 'graph': graph1,
            'outputs': {
                # 'credit_min': {'shape': (1, 1), 'buscopy': 'measure/bcredit_min'},
                # 'credit_max': {'shape': (1, 1), 'buscopy': 'measure/bcredit_max'},
                # 'credit_mu': {'shape': (1, 1), 'buscopy': 'measure/bcredit_mu'},
                'credit_min': {'shape': (1, numloop_inner), 'buscopy': 'b4/credit_min'},
                'credit_max': {'shape': (1, numloop_inner), 'buscopy': 'b4/credit_max'},
                'credit_mu': {'shape': (1, numloop_inner), 'buscopy': 'b4/credit_mu'},
            }
        },
    }
# )])
        
# loop = [('dim', i) for i in range(1, numloop + 1)]
loop = [('lconf', {
            'dim': i + 1,
            'dt': 0.1,
            'lag': 1,
            'budget': 1000,
            'lim': 1.0,
            'order': 0,
            'numloop': numloop_inner,
        }) for i in range(numloop)]
print("loop", loop)

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
    #         'outputs': {'credit_min': {'shape': (1, numloop)}},
    #     },
    # }),

    # sequential loop
    ("b5", {
        'block': SeqLoopBlock2,
        'params': {
            'id': 'b5',
            'debug': False,
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'numsteps':  numsteps,
            'loopblocksize': numsteps/numloop, # loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {
                'credit_min': {'shape': (1, numloop * numloop_inner), 'buscopy': 'b4/credit_min'},
                'credit_max': {'shape': (1, numloop * numloop_inner), 'buscopy': 'b4/credit_max'},
                'credit_mu': {'shape': (1, numloop * numloop_inner), 'buscopy': 'b4/credit_mu'},
                # 'credit_min': {'shape': (1, numloop)},
                # 'credit_max': {'shape': (1, numloop)},
                # 'credit_mu': {'shape': (1, numloop)},
                # 'x': {'shape': (3, numsteps)},
                # 'y': {'shape': (1, numsteps)}
            },

            # single dim config statistics
            'loop': loop,
            
            # # loop over dims
            # 'loop': [
            #     ('lconf', {
            #         'dim': (i + 1),
            #         'dt': 0.1,
            #         'lag': 1,
            #         'budget': 1000,
            #         'lim': 1.0,
            #         }) for i in range(0, numloop)],

            # # failed attempt looping subgraphconf
            # 'loop': [
            #     ('subgraphconf', {
            #         'robot1/sysdim': i + 2,
            #         'robot1/statedim': (i + 2) * 3,
            #         'robot1/dim_s_proprio': (i + 2),
            #         'robot1/dim_s_extero': (i + 2),
            #         'robot1/outputs': {
            #             's_proprio': {'shape': ((i + 2), 1)},
            #             's_extero': {'shape': ((i + 2), 1)},
            #             }
            #         }) for i in range(0, numloop)],

            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),

    # plotting
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'debug': True,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.35,
            'hspace': 0.25,
            'inputs': {
                'mins_s': {'bus': 'b5/credit_min', 'shape': (1, numloop * numloop_inner)},
                'maxs_s': {'bus': 'b5/credit_max', 'shape': (1, numloop * numloop_inner)},
                'mus_s':  {'bus': 'b5/credit_mu',  'shape': (1, numloop * numloop_inner)},
                # 'mins_p': {'bus': 'b3/credit_min', 'shape': (1, numloop)},
                },
            'subplots': [
                [
                    {
                        'input': ['mins_s', 'maxs_s', 'mus_s'],
                        'xslice': (i * numloop_inner, (i+1) * numloop_inner),
                        'plot': partial(
                            timeseries,
                            ylim = (-30, 1030),
                            yscale = 'linear',
                            linestyle = 'none',
                            marker = 'o')
                    },
                    {
                        'input': ['mins_s', 'maxs_s', 'mus_s'],
                        'xslice': (i * numloop_inner, (i+1) * numloop_inner),
                        'plot': partial(
                            histogram,
                            title = 'mean/min budget hist',
                            ylim = (-30, 1030),
                            yscale = 'linear',
                            orientation = 'horizontal')
                    },]
                for i in range(numloop)
        ],
        },
    })
    
])
