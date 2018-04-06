"""expr0400 dm_actinf.py variations

Variations:
- episode length
- target: pre_l1
- learner: pre_l0
  - tapping: pre_l0
- system: robot1


FIXME: looping + subgraph
"""
from smp_graphs.utils_conf import get_systemblock
from smp_graphs.funcs import f_sin_noise

# reuse
graphconf = 'conf/dm_imol.py'
debug = False

# add latex output
outputs = {
    'latex': {'type': 'latex',},
}

# local configuration
cnf = {
    'numsteps': 2000, # 5000
}
    
numsteps = cnf['numsteps']

lconf = {
    # execution and global
    'numsteps': int(10000/5),
    
    # system
    'sys': {
        'name': 'pm',
        'lag': 2,
        'dim_s0': 2,
        'dim_s1': 2,
        'lag_past': (-4, -3),
        'lag_future': (-1, 0),
    },
    
    # motivation
    'motivation_i': 1,
    
    # models
    
    # learning modulation
    'lm' : {
        'washout': 200,
        'thr_predict': 200,
        'fit_offset': 200,
        # start stop in ratios of episode length numsteps
        'r_washout': (0.0, 0.1),
        'r_train': (0.1, 0.8),
        'r_test': (0.8, 1.0),
    }
    
}
systemblock = get_systemblock[lconf['sys']['name']](**lconf['sys'])

desc = """Variation 1 of dm\_imol using a different target function.""".format()

# execution
saveplot = True
recurrent = True
debug = False
showplot = True

# experiment
randseed = 12345
numsteps = int(10000/5)
dim = 1
motors = dim
dt = 0.1

m_mins = np.array([systemblock['params']['m_mins']]).T
m_maxs = np.array([systemblock['params']['m_maxs']]).T
dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1 = systemblock['params']['dims']['s1']['dim']

# motivations = [
#     ('random_uniform_fixed_rate', ),
#     ('periodic_function_fixed_rate', ),
#     ('intrinsic_motivation', ),
#     ('error_based_resampling', ),
# ]

# looparray = [('file', {'filename': fname, 'filetype': 'puppy', 'offset': random.randint(0, 500), 'length': numsteps}) for fname in filearray]
# models = ['knn', 'igmm', 'soesgp'] # , 'storkgp',]

# motivations = OrderedDict([
#     ('random_uniform', {
#         'models': {
#             'goal': {'type': 'random_uniform'}
#         },
#         'inputs': {
#             'lo': {'val': m_mins, 'shape': (dim_s0, 1)},
#             'hi': {'val': m_maxs, 'shape': (dim_s0, 1)},
#         },
#         'rate': 40,
#     }),
    
#     ('sinusoid', {
#         'inputs': {                        
#             'x': {'bus': 'cnt/x'},
#             # pointmass, good with soesgp and eta = 0.7
#             'f': {'val': np.array([[0.23539]]).T * 0.2 * dt},
#             'sigma': {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
#             'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
#             'amp': {'val': (m_maxs - m_mins)/2.0},
#         },
#         'models': {
#             'goal': {
#                 'type': 'function_generator',
#                 'func': f_sin_noise,
#             }
#         },
#         'rate': 1,
#     })
# ])

# looparray = [('subgraphconf', dict(
#         [('pre_l1/%s' % k, v) for k, v in motivations[motk].items()]
#     ))
#     # 'pre_l0/models': {
#     #     'm1': {
#     #         'type': 'actinf_m1',
#     #         'algo': model,
#     #         # 'type': 'actinf_m1',
#     #         # 'algo': algo,
#     #         # 'lag_past': lag_past,
#     #         # 'lag_future': lag_future,
#     #         # 'idim': dim_s_proprio * (lag_past[1] - lag_past[0]) * 2, # laglen
#     #         # 'odim': dim_s_proprio * (lag_future[1] - lag_future[0]), # laglen,
#     #         # 'laglen': laglen,
#     #         # 'eta': eta,
#     #     },
#     # },
#     # 'plot_ts/title': 'actinf_m1 1D with %s' % (model, )
#     for (i, motk) in enumerate(motivations)]

# print "looparray", looparray
# print "looparray[0][1]", looparray[0][1]

motivations = [
    # goal sampler (motivation) sample_discrete_uniform_goal
    ('pre_l1/subgraph', {
        'block': ModelBlock2,
        'params': {
            'blocksize': 1,
            'blockphase': [0],
            'inputs': {                        
                'lo': {'val': m_mins, 'shape': (dim_s0, 1)},
                'hi': {'val': m_maxs, 'shape': (dim_s0, 1)},
            },
            'outputs': {'pre': {'shape': (dim_s0, 1)}},
            'models': {
                'goal1': {'type': 'random_uniform'}
            },
            'rate': 40,
        },
    }),

    # goal sampler (motivation) sample_function_generator sinusoid
    ('pre_l1/subgraph', {
        'block': ModelBlock2,
        'params': {
            'blocksize': 1,
            'blockphase': [0],
            'inputs': {
                'x': {'bus': 'cnt/x'},
                # pointmass
                'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, # good with soesgp and eta = 0.7
                'sigma': {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
                'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
                'amp': {'val': (m_maxs - m_mins)/2.0},
            },
            'outputs': {'pre': {'shape': (dim_s0, 1)}},
            'models': {
                'goal': {
                    'type': 'function_generator',
                    'func': f_sin_noise,
                }
            },
            'rate': 1,
        },
    }),

    # a random number generator, mapping const input to hi
    ('pre_l1/subgraph', {
        'block': FuncBlock2,
        'params': {
            'id': 'pre_l1',
            'outputs': {'pre': {'shape': (dim_s0, 1)}},
            'debug': False,
            'ros': ros,
            'blocksize': 1,
            # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            # recurrent connection
            'inputs': {
                'x': {'bus': 'cnt/x'},
                # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
                # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.45]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.225]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                
                # barrel
                # 'f': {'val': np.array([[0.23539]]).T * 10.0 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.23539]]).T * 7.23 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.23539]]).T * 2.9 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.23539]]).T * 1.25 * dt}, # good with soesgp and eta = 0.7
                
                # pointmass
                'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, # good with soesgp and eta = 0.7
            
                # 'f': {'val': np.array([[0.23539, 0.2348, 0.14]]).T * 1.25 * dt}, # good with soesgp and eta = 0.7
                # 'f': {'val': np.array([[0.14, 0.14]]).T * 1.0},
                # 'f': {'val': np.array([[0.82, 0.82]]).T},
                # 'f': {'val': np.array([[0.745, 0.745]]).T},
                # 'f': {'val': np.array([[0.7, 0.7]]).T},
                # 'f': {'val': np.array([[0.65, 0.65]]).T},
                # 'f': {'val': np.array([[0.39, 0.39]]).T},
                # 'f': {'val': np.array([[0.37, 0.37]]).T},
                # 'f': {'val': np.array([[0.325, 0.325]]).T},
                # 'f': {'val': np.array([[0.31, 0.31]]).T},
                # 'f': {'val': np.array([[0.19, 0.19]]).T},
                # 'f': {'val': np.array([[0.18, 0.181]]).T},
                # 'f': {'val': np.array([[0.171, 0.171]]).T},
                # 'f': {'val': np.array([[0.161, 0.161]]).T},
                # 'f': {'val': np.array([[0.151, 0.151]]).T},
                # 'f': {'val': np.array([[0.141, 0.141]]).T},
                # stay in place
                # 'f': {'val': np.array([[0.1, 0.1]]).T},
                # 'f': {'val': np.array([[0.24, 0.24]]).T},
                # 'sigma': {'val': np.array([[0.001, 0.002]]).T}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                'sigma': {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
                'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
                'amp': {'val': (m_maxs - m_mins)/2.0},
            }, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'func': f_sin_noise,
        },
    })
]
    

# 'pre_l0_test/models': {'fwd': {'type': 'actinf_m1', 'algo': 'copy', 'copyid': 'pre_l0', 'idim': dim * 2, 'odim': dim},},

# del lconf['systemblock']

loopblock = {
    'block': Block2,
    'params': {
        # 'id': 'actinf_m1_loop',
        'debug': False,
        'blocksize': 1,
        # 'numsteps': numsteps,
        'lconf': lconf,
        'subgraph': graphconf,
        # 'subgraphconf': looparray[0][1],
        # 'subgraphconf': OrderedDict([motivations[0]]),
        'subgraph_rewrite_id': False, # True,
        'outputs': {},
    },
}

# # graph
# graph = OrderedDict([
#     # puppy data
#     ('dm_actinf', {
#         'block': LoopBlock2,
#         # 'blocksize': 1,
#         # 'numsteps': numsteps,
#         'params': {
#             # 'numsteps': numsteps,
#             # required
#             'loop': looparray,
#             'loopmode': 'parallel',
#             'loopblock': loopblock,
#         },
#     }),
# ])

# graph from expr0300
graph = OrderedDict([
    ('n1', loopblock),
    # ("reuser", {
    #     'block': Block2,
    #     'params': {
    #         'debug': True,
    #         'topblock': False,
    #         'numsteps': 1,
    #         'lconf': lconf,
    #         # points to config file containing the subgraph specification
    #         'subgraph': graphconf,
    #         # dynamic id rewriting on instantiation
    #         'subgraph_rewrite_id': False,
    #         # override initial config with local config
    #         'subgraphconf': {
    #             # 'puppylog/file': {'filename': cnf['logfile']},
    #             # 'puppylog/type': cnf['logtype'],
    #             # 'puppylog/blocksize': numsteps,
    #         }
    #     },
    # }),
])


# print "graph", graph
