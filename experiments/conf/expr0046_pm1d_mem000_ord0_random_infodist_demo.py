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
numsteps = int(10000/1)
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 127

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 2000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'numloop': 4,
}
    
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0
numloop = lconf['numloop'] # 1.0

expr_number=6
expr_name = "Experiment {0}".format(expr_number)
desc = """This experiment is using expr0045 as a starting point and
iterating system configuration parameters controlling the shape of the
transfer function. The shape produces an effect that is reflected in
the error, the divergence of $X$ and $Y$ distributions, and also in the
information distance between the two distribution."""

outputs = {
    'latex': {'type': 'latex',},
}

global l_names, l_as, d_as, d_ss, s_as, s_fs, es
# configuration name
l_names = ['wide sigmoid', 'narrow sigmoid', '1/f transfer noise', 'dynamic noise']
# linear component mixture amplitude
l_as = [0.5, 0.0, 0.9, 0.75]
# gaussian component mixture amplitude
d_as = [0.5, 1.0, 0.0, 0.0]
# gaussian component spread amplitude (sigma)
d_ss = [0.8, 0.5, 1.0, 1.0]
# noise component mixture amplitude
s_as = [0.0, 0.0, 0.1, 0.0]
# noise component color (1/color)
s_fs = [0.0, 0.0, 1.0, 0.0]
# external entropy
es   = [0.0, 0.0, 0.0, 0.5]

# final: random sampling in d space
loop = [('lconf', {
    'expr_number': 6, # 'expr0046-{0}'.format(i+1),
    'expr_name': "Experiment 6-{0}".format(i),
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': 1001,
        'l_a': l_as[i],
        'd_a': d_as[i],
        'd_s': d_ss[i],
        's_a': s_as[i],
        's_f': s_fs[i],
        'e': es[i],
    },
    'div_meas': 'chisq',
    'plot_desc': """Motor to sensor prediction divergence for the {0}
        configuration of the transfer function $h$. Please refer to
        \\autoref{{fig:smp-expr0045-pm1d-mem000-ord0-random-infodist-id-plot}}
        for a full description of the structure of this
        experiment.""".format(l_names[i]),
}) for i in range(numloop)]

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
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
        'subgraph_ignore_nodes': [], # ['plot'],
        'subgraphconf': {
            # 'plot/active': False
            # 'robot1/sysdim': 1,
            },
        # 'graph': graph1,
        'outputs': {
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
        'params': {
            'id': 'b4',
            # 'debug': True,
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'numsteps':  numsteps,
            'loopblocksize': numsteps/numloop, # loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {
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

    # # plotting
    # ('plot', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'id': 'plot',
    #         'blocksize': numsteps,
    #         'saveplot': saveplot,
    #         'savetype': 'pdf',
    #         'hspace': 0.2,
    #         'wspace': 0.2,
    #         'inputs': {
    #             'mins_s': {'bus': 'b4/credit_min', 'shape': (1, numloop)},
    #             'maxs_s': {'bus': 'b4/credit_max', 'shape': (1, numloop)},
    #             'mus_s': {'bus': 'b4/credit_mu', 'shape': (1, numloop)},
    #             # 'mins_p': {'bus': 'b3/credit_min', 'shape': (1, numloop)},
    #             },
    #         'subplots': [
    #             [
    #                 {
    #                 'input': ['mins_s', 'maxs_s', 'mus_s'],
    #                 'plot': partial(
    #                     timeseries,
    #                     ylim = (-30, 1030),
    #                     yscale = 'linear',
    #                     linestyle = 'none',
    #                     marker = 'o')},
    #                 {'input': ['mins_s', 'maxs_s', 'mus_s'], 'plot': partial(
    #                     histogram,
    #                     title = 'mean/min budget hist',
    #                     ylim = (-30, 1030),
    #                     yscale = 'linear',
    #                     orientation = 'horizontal')}
    #             ],
    #         ],
    #     },
    # })
    
])
