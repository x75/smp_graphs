"""smp_graphs configuration

baseline behaviour - open-loop uniform random search in finite isotropic space

id:thesis_smp_expr0050

Oswald Berthold 2017

special case of kinesis with coupling = 0 between measurement and action
"""

from smp_graphs.block import FuncBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/10
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

################################################################################
# raw loop params

# for stats
# FIXME: construct systematically, include linear amp 1.0 ... 0.0
l_as = [1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.99, 0.9, 0.85, 0.8, 0.6, 0.5, 1.0, 1.00, 1.00, 1.0, 1.0, 1.0, ]
d_as = [0.0, 0.5, 0.8, 0.9, 1.0, 1.0, 0.00, 0.0, 0.00, 0.0, 0.0, 0.0, 1.0, 0.00, 0.00, 0.0, 0.0, 0.0, ]
d_ss = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.00, 1.0, 0.90, 0.6, 0.3, 0.1, 0.1, 1.00, 1.00, 0.9, 0.6, 0.3, ]
s_as = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.1, 0.15, 0.2, 0.4, 0.5, 0.0, 0.00, 0.00, 0.0, 0.0, 0.0, ]
s_fs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.00, 1.8, 2.00, 2.5, 3.0, 4.0, 0.0, 0.00, 3.00, 2.5, 2.0, 1.0, ]
es   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.00, 0.0, 0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, ]
numloop_types = 3
numloop_param = 6

# # for testing
# l_as = [1.0, 0.5, 0.5, 0.0, ]
# d_as = [0.0, 0.5, 0.5, 1.0, ]
# d_ss = [1.0, 1.0, 0.5, 0.3, ]
# s_as = [0.0, 0.0, 0.0, 0.0, ]
# s_fs = [0.0, 0.0, 0.0, 0.0, ]
# es   = [0.0, 0.0, 0.0, 0.0, ]
# numloop_types = 2
# numloop_param = 2

numloop = len(l_as)

# looper lconf
lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'numloop': numloop,
    'loop_pre_l2_numelem': 1001,
}
    
dim = lconf['dim']
dim_s_proprio = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0
# numloop = lconf['numloop'] # 1.0

# m_inputs = OrderedDict([('m_sum_div', {'xtwin': False,}), ('m_mi', {'xtwin': True}), ('m_di', {'xtwin': True})])
# m_inputs = OrderedDict([('m_sum_div', {'xtwin': False,}), ('m_di', {'xtwin': True})])
# m_inputs = OrderedDict([('m_di', {'xtwin': False})])
m_inputs = OrderedDict(
    [
        ('m_sum_div', {'xtwin': False,}),
        ('m_budget_mu', {'xtwin': False,}),
        ('m_rmse', {'xtwin': True,}),
        ('m_di', {'xtwin': True}),
    ]
)

desc = """This experiment is looping expr0045 over the parameters that control the information distance between to spaces."""

outputs = {
    'latex': {'type': 'latex',},
}

# final: random sampling in d space with loopblock lconf
loop = [('lconf', {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': lconf['loop_pre_l2_numelem'],
        'l_a': l_as[i],
        'd_a': d_as[i],
        'd_s': d_ss[i],
        's_a': s_as[i],
        's_f': s_fs[i],
        'e': es[i],
    },
    'div_meas': 'chisq',
}) for i in range(numloop)]

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        # 'debug': True,
        'logging': True,
        'topblock': False,
        'numsteps': numsteps,
        'randseed': 1,
        # subcomponent?
        # 'robot1/dim': 2,
        'lconf': {},
        # contains the subgraph specified in this config file
        'subgraph': 'conf/expr0045_pm1d_mem000_ord0_random_infodist_id.py',
        'subgraph_rewrite_id': True,
        'subgraph_ignore_nodes': ['plot_infodist'],
        'subgraphconf': {
            # 'plot/active': False
            # 'robot1/sysdim': 1,
            },
        # 'graph': graph1,
        'outputs': {
            'gm_mi': {'shape': (1, 1, 1), 'buscopy': 'm_mi/mi'},
            'gm_di': {'shape': (1, 1, 1), 'buscopy': 'm_di/infodist'},
            'gm_sum_div': {'shape': (1, 1), 'buscopy': 'm_sum_div/y'},
            'gm_rmse': {'shape': (1, 1), 'buscopy': 'm_rmse/y'},
            'gm_budget_mu': {'shape': (1, 1), 'buscopy': 'm_budget/y_mu'},
            'pre_l2_h': {'shape': (dim_s_proprio, lconf['loop_pre_l2_numelem']), 'buscopy': 'pre_l2/h'},
            # 'credit_min': {'shape': (1, 1), 'buscopy': 'measure/bcredit_min'},
            # 'credit_max': {'shape': (1, 1), 'buscopy': 'measure/bcredit_max'},
            # 'credit_mu': {'shape': (1, 1), 'buscopy': 'measure/bcredit_mu'},
        }
    },
}
# loop = [('randseed', 1000 + i) for i in range(0, numloop)]
# loop = [('randseed', 1000 + i) for i in range(0, numloop)]

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
    ("b4", {
        'block': SeqLoopBlock2,
        # 'block': LoopBlock2,
        'params': {
            'id': 'b4',
            'debug': True,
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'numsteps':  numsteps,
            'loopblocksize': numsteps/numloop, # loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {
                'gm_mi': {'shape': (1, 1, numloop)},
                'gm_di': {'shape': (1, 1, numloop)},
                'gm_sum_div': {'shape': (1, numloop)},
                'gm_rmse': {'shape': (1, numloop)},
                'gm_budget_mu': {'shape': (1, numloop)},
                'pre_l2_h': {'shape': (dim_s_proprio, lconf['loop_pre_l2_numelem'] * numloop)},
                # 'credit_min': {'shape': (1, numloop)},
                # 'credit_max': {'shape': (1, numloop)},
                # 'credit_mu': {'shape': (1, numloop)},
                # 'x': {'shape': (3, numsteps)},
                # 'y': {'shape': (1, numsteps)}
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
            'hspace': 0.2,
            'wspace': 0.2,
            'inputs': {
                'm_mi': {'bus': 'b4/gm_mi', 'shape': (1, 1, numloop)},
                'm_di': {'bus': 'b4/gm_di', 'shape': (1, 1, numloop)},
                'm_sum_div': {'bus': 'b4/gm_sum_div', 'shape': (1, numloop)},
                'm_rmse': {'bus': 'b4/gm_rmse', 'shape': (1, numloop)},
                'm_budget_mu': {'bus': 'b4/gm_budget_mu', 'shape': (1, numloop)},
                'pre_l2_h': {'bus': 'b4/pre_l2_h', 'shape': (dim_s_proprio, lconf['loop_pre_l2_numelem'] * numloop)},
                # 'mins_s': {'bus': 'b4/credit_min', 'shape': (1, numloop)},
                # 'maxs_s': {'bus': 'b4/credit_max', 'shape': (1, numloop)},
                # 'mus_s': {'bus': 'b4/credit_mu', 'shape': (1, numloop)},
                # 'mins_p': {'bus': 'b3/credit_min', 'shape': (1, numloop)},
            },
            'subplots': [
                [
                    {
                        'input': ['pre_l2_h'] * numloop_param,
                        'xslice': [(
                            (j * numloop_param + i) * lconf['loop_pre_l2_numelem'],
                            (j * numloop_param + i + 1) * lconf['loop_pre_l2_numelem']
                        ) for i in range(numloop_param)],
                        # shape overrides xslice
                        # 'shape': (dim_s_proprio, lconf['loop_pre_l2_numelem']),
                        'plot': timeseries,
                        'title': 'transfer functions $h_i$', 'aspect': 1.0,
                        'xaxis': np.linspace(-1, 1, lconf['loop_pre_l2_numelem']), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend_loc': 'right',
                    },

                    {
                        'input': list(m_inputs.keys()),
                        'ndslice': [
                            (slice(j * numloop_param, (j + 1) * numloop_param), slice(None)),
                            (slice(j * numloop_param, (j + 1) * numloop_param), slice(None)),
                            (slice(j * numloop_param, (j + 1) * numloop_param), slice(None)),
                            (slice(j * numloop_param, (j + 1) * numloop_param), slice(None), 0),
                            # (slice(j * numloop_param, (j + 1) * numloop_param), slice(None), 0),
                        ],
                        'shape': (1, numloop_param),
                        'title': 'measures series',
                        # 'ylim': (-30, 1030),
                        'plot': [
                            partial(
                                timeseries,
                                yscale = 'linear',
                                linestyle = 'none', marker = 'o'
                            ) for k, v in list(m_inputs.items())],
                        'xtwin': [v['xtwin'] for k, v in list(m_inputs.items())],
                    },
                    
                    # {
                    #     'input': m_inputs.keys(),
                    #     'ndslice': [
                    #         (slice(j * numloop_param, (j + 1) * numloop_param), slice(None)),
                    #         (slice(j * numloop_param, (j + 1) * numloop_param), slice(None), 0),
                    #         (slice(j * numloop_param, (j + 1) * numloop_param), slice(None), 0),
                    #     ],
                    #     # 'xslice': [
                    #     #     (j * numloop_param, (j + 1) * numloop_param),
                    #     #     (j * numloop_param, (j + 1) * numloop_param),
                    #     #     (j * numloop_param, (j + 1) * numloop_param),
                    #     # ],
                    #     'shape': (1, numloop_param),
                    #     # 'ylim': (-30, 1030),
                    #     'title': 'measures hist',
                    #     'plot': partial(
                    #         histogram,
                    #         yscale = 'linear',
                    #         orientation = 'horizontal',
                    #     ),
                    #     'xtwin': [v['xtwin'] for k, v in m_inputs.items()],
                    # }
                    
                ] for j in range(numloop_types)
                
            ],
        },
    })
    
])
