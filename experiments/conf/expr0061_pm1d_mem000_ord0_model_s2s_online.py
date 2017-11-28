"""smp_graphs expr0061

.. moduleauthor:: Oswald Berthold 2017

From :mod:`expr0060_pm1d_mem000_ord0_model_s2s`
 - take 0060 and add online learning

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

lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'numloop': 1,
}
    
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0
numloop = lconf['numloop'] # 1.0

desc = """This experiment is looping expr0045 over the parameters that
control the information distance between to spaces illustrating each
parameter's effects on information distance."""

outputs = {
    'latex': {'type': 'latex',},
}

# for stats
l_as = [0.5, 0.0, 0.5, 0.75]
d_as = [0.5, 1.0, 0.0, 0.0]
d_ss = [0.8, 0.5, 1.0, 1.0]
s_as = [0.0, 0.0, 0.2, 0.0]
s_fs = [0.0, 0.0, 1.0, 0.0]
es   = [0.0, 0.0, 0.0, 0.5]

# local conf dict for looping
lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
}

# lconf['model_s2s'] = model_s2s_sklearn

p_vars = ['robot1/s0']
# measured variables
# m_vars = ['robot1/s0']
m_vars = ['pre_l2/y']

dim_s0 = lconf['dim']
numelem = 1001





# imol
algo = 'knn' # ok
# algo = 'gmm' # ok
# algo = 'igmm' # ok, fix deprecation, inference output conf
# algo = 'hebbsom' # fix
# algo = 'soesgp'
# algo = 'storkgp'
# algo = 'resrls'
# algo = 'homeokinesis'

# pm
algo_fwd = algo
algo_inv = algo
lag_past   = (-1, -0)
lag_future = (-1, 0)
minlag = 1
maxlag = 20 # -lag_past[0] + lag_future[1]
laglen = maxlag - minlag
eta = 0.15

# final: random sampling in d space
loop = [('lconf', {
    'dim': dim_s0,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    # 'infodistgen': {
    #     'type': 'random_lookup',
    #     'numelem': numelem,
    #     'l_a': l_as[i],
    #     'd_a': d_as[i],
    #     'd_s': d_ss[i],
    #     's_a': s_as[i],
    #     's_f': s_fs[i],
    #     'e': es[i],
    # },
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': 1001, # sampling grid
        'l_a': 0.0,
        'd_a': 0.98,
        'd_s': 0.5,
        's_a': 0.02,
        's_f': 2.0,
        'e': 0.0,
    },
    'div_meas': 'chisq', # 'kld'
    # 'model_s2s': {
    #     'block': repr(ModelBlock2),
    'model_s2s_params': {
            'debug': True,
            'blocksize': numsteps,
            'models': {
                # from top config
                # 'pre_l2_2_robot1_s0': shln,
                'pre_l2_2_robot1_s0': {
                    'type': 'sklearn',
                    'skmodel': 'linear_model.LinearRegression',
                    'skmodel_params': {}, # {'alpha': 1.0},
                    # 'skmodel': 'kernel_ridge.KernelRidge',
                    # 'skmodel_params': {'alpha': 0.1, 'kernel': 'rbf'},
                    # 'skmodel': 'gaussian_process.GaussianProcessRegressor',
                    # 'skmodel_params': {'kernel': ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)},
                    # 'skmodel': 'gaussian_process.kernels.WhiteKernel, ExpSineSquared',
                    # 'skmodel': model_selection.GridSearchCV
                },
                # 'pre_l2_2_robot1_s0': {
                #     'type': 'imol',
                #     'algo': algo,
                #     'fwd': {
                #         'type': 'imol',
                #         'algo': algo_fwd,
                #         'lag_past': lag_past,
                #         'lag_future': lag_future,
                #         'idim': dim_s0 * (lag_past[1] - lag_past[0]) * 2, # laglen
                #         'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                #         'laglen': laglen,
                #         # 'laglen_past': lag_past[1] - lag_past[0],
                #         # 'laglen_future': lag_future[1] - lag_future[0],
                #         'eta': eta,
                #     },
                #                 'inv': {
                #                     'type': 'imol',
                #                     'algo': algo_inv,
                #                     'lag_past': lag_past,
                #                     'lag_future': lag_future,
                #                     'idim': dim_s0 * (lag_past[1] - lag_past[0]) * 3, # laglen
                #                     'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                #                     'laglen': laglen,
                #                     # 'laglen_past': lag_past[1] - lag_past[0],
                #                     # 'laglen_future': lag_future[1] - lag_future[0],
                #                     'eta': eta,
                #                     'memory': maxlag,
                #                     'w_input': 1.0,
                #                     'w_bias': 0.2,
                #                     'multitau': False, # True,
                #                     'theta': 0.01,
                #                     # soesgp
                #                     'modelsize': 200,
                #                     # 'spectral_radius': 1.5, # 0.01,
                #                     # FORCE / pm
                #                     # 'modelsize': 300,
                #                     'theta_state': 0.1,
                #                     'lrname': 'FORCE',
                #                     'alpha': 50.0, # 10.0,
                #                     'spectral_radius': 0.99, # 0.01,
                #                     'tau': 0.85, # 0.8, # 0.05, # 1.0,
                #                     # 'tau': 0.4, # 0.8, # 0.05, # 1.0,
                #                     'wgt_thr': 2.3,
                #                     # RLS / barrel
                #                     # 'modelsize': 322,
                #                     # 'theta_state': 0.1, # for RLS
                #                     # 'lrname': 'RLS',
                #                     # 'spectral_radius': 0.999, # 0.01,
                #                     # 'tau': 0.08, # 0.8, # 0.05, # 1.0,
                #                     # 'wgt_thr': 0.5,
                #                     'mixcomps': 3,
                #                     'oversampling': 1,
                #                     'visualize': False,
                #                 }
                # },
            },
            'inputs': {
                # input
                'x_in': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
                # target
                'x_tg': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
            },
            'outputs': {
                'y': {'shape': (dim_s0, numsteps)},
                'h': {'shape': (dim_s0, numelem), 'trigger': 'trig_ll%d_ll%d/pre_l2_t1' % (i, i, )},
            },
        },
}) for i in range(numloop)]

print "loop", loop
    
loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': True,
        'logging': True,
        'topblock': False,
        'numsteps': numsteps,
        'randseed': 1,
        # subcomponent?
        # 'robot1/dim': 2,
        'lconf': {},
        # contains the subgraph specified in this config file
        'subgraph': 'conf/expr0060_pm1d_mem000_ord0_model_s2s.py',
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
            'debug': True,
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
            #         'robot1/dim_s0': (i + 2),
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
