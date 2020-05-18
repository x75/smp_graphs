"""dm_eh.py

smp_graphs config for experiment

    exploratory hebbian learner
    model type reservoir EH
    simple n-dimensioal system (point mass, simple n-joint arm)

EH legacy
 - smp/neural/esn_reward_MSO.py: early EH learning with sinewave target
 - smp/neural/esn_reward_MSO_EH.py: early EH learning with sinewave target and more analyis, hyperopt

# robustness
# fix: target properties like frequency matched to body (attainable / not attainable)
# fix: eta
# fix: motor / sensor limits, how does the model learn the limits

# priors: timing, limits
"""

import copy
from functools import partial

from smp_base.plot import histogramnd
from smp_base.measures import meas
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2
from smp_graphs.block_meas_infth import TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2, BhasimulatedBlock2
from smp_graphs.block_cls_ros import STDRCircularBlock2, LPZBarrelBlock2, SpheroBlock2

from smp_graphs.funcs import f_meshgrid, f_meshgrid_mdl, f_random_uniform, f_sin_noise
from smp_graphs.utils_conf import get_systemblock

# execution
saveplot = False
recurrent = True
debug = False
showplot = True
ros = False

# experiment
commandline_args = ['numsteps']
randseed = 12360

expr_number = 27
expr_name = 'Experiment {0}'.format(expr_number)
# desc = """The final experiment serves to illustrate that by combining
# an instantaneous error $e$ with versions of itself integrated over
# different time spans, primitive motivation $m$ is obtained. The
# motivation $m$ is hardwired to spawn a local model at the site the the
# error occurs and accumulates.""".format()

lconf = {
    # execution and global
    'numsteps': int(10000/4),
    'expr_number': expr_number,
    'expr_name': expr_name,
    # system
    'sys': {
        'name': 'pm',
        'lag': 2,
        'dim_s0': 2,
        'dim_s1': 2,
    },
    # motivation
    'motivation_i': 0,
    'motivation_rate': 40,
    # model
    'algo': 'res_eh',
    # eta = 0.99
    # eta = 0.95
    # eta = 0.7
    # eta = 0.3
    # eta = 0.25
    # eta = 0.15
    # eta = 0.1
    # eta = 0.05
    'eta': 3e-3,
    'dim_s_hidden_debug': 20,
    # model config defaults
    'mdl_cnf': {
        'mdl_modelsize': 300,
        'mdl_w_input': 1.0,
        'mdl_w_bias': 0.5,
        'mdl_theta': 1e-1,
        'mdl_eta': 1e-4,
        'mdl_spectral_radius': 0.999,
        'mdl_tau': 0.1,
        'mdl_mdltr_type': 'bin_elem',
        'mdl_mdltr_thr': 0.0,
        'mdl_wgt_thr': 1.0, # 
        'mdl_perf_measure': 'meas.square', # meas.abs, # abs, square, sum_abs, sum_square, sum_sqrt
        'mdl_perf_model_type': 'lowpass', # 'resforce'
        'mdl_coeff_a': 0.2, 
    },
    'tgt_cnf': {
        'target_f': 0.05,
    },

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
lconf['systemblock'] = get_systemblock[lconf['sys']['name']](**lconf['sys'])

numsteps = lconf['numsteps']
loopblocksize = numsteps

# learning modulation
for k in list(lconf['lm']):
    if not k.startswith('r_'): continue
    k_ = k.replace('r_', 'n_')
    r0 = lconf['lm'][k][0] * lconf['numsteps']
    r1 = lconf['lm'][k][1] * lconf['numsteps']
    # lconf['lm'][k_] = range(int(r0), int(r1))
    lconf['lm'][k_] = (int(r0), int(r1))

sysname = lconf['sys']['name']
loopblocksize = numsteps
# sysname = 'pm'
# sysname = 'sa'
# sysname = 'bha'
# sysname = 'stdr'
# sysname = 'lpzbarrel'
# sysname = 'sphero'

outputs = {
    'latex': {'type': 'latex',},
}

# FIXME: param: perf_lp/perf_lp_fancy, element-wise/np.any/np.all, input_coupling

# """system block, the robot"""
# def get_systemblock_pm(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1):
#     global np, PointmassBlock2, meas
#     return {
#         'block': PointmassBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'systype': 2,
#             'sysdim': dim_s_proprio,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero':  {'shape': (dim_s_extero, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s_proprio * 3,
#             'dt': dt,
#             'mass': 1.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.01,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s_proprio': dim_s_proprio,
#             'length_ratio': 3./2., # gain curve?
#             'm_mins': [-1.0] * dim_s_proprio,
#             'm_maxs': [ 1.0] * dim_s_proprio,
#             'dim_s_extero': dim_s_extero,
#             # pm2sys params
#             'lag': 3,
#             'order': 2,
#             'coupling_sigma': 5e-1, # 2.5e-2,
#             'transfer': 1, # 1, # 1
#             'anoise_mean': 0.0,
#             'anoise_std': 1e-3,
            
#             # model related
#             # tapping
#             'lag_past': (-4, -3),
#             'lag_future': (-1, 0),
#             # low-level params
#             'mdl_modelsize': 300,
#             'mdl_w_input': 1.0,
#             'mdl_theta': 0.5e-1,
#             'mdl_eta': 1e-3,
#             'mdl_mdltr_type': 'cont_elem', #'bin_elem',
#             'mdl_mdltr_thr': 0.005, # 0.01,
#             'mdl_wgt_thr': 1.0, # 
#             'mdl_perf_measure': meas.square, # .identity
#             'mdl_perf_model_type': 'lowpass', # 'resforce',
#             # target parameters
#             'target_f': 0.05,
#             }
#         }

# def get_systemblock_sa(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1):
#     global np, SimplearmBlock2
#     return {
#         'block': SimplearmBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'sysdim': dim_s_proprio,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
#             # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero':  {'shape': (dim_s_extero,  1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s_proprio * 3,
#             'dt': dt,
#             'lag': 1,
#             'lag_past': (-2, -1),
#             'lag_future': (-1, 0),
#             'mass': 1.0/3.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.001,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s_proprio': dim_s_proprio,
#             'length_ratio': 3./2.0,
#             'm_mins': [-1.] * dim_s_proprio,
#             'm_maxs': [ 1.] * dim_s_proprio,
#             # 's_mins': [-1.00] * 9,
#             # 's_maxs': [ 1.00] * 9,
#             # 'm_mins': -1,
#             # 'm_maxs': 1,
#             'dim_s_extero': dim_s_extero,
#             'minlag': 1,
#             'maxlag': 2, # 5
#             }
#         }

# def get_systemblock_bha(dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1):
#     global np, BhasimulatedBlock2
#     return {
#         'block': BhasimulatedBlock2,
#         'params': {
#             'id': 'robot1',
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'sysdim': dim_s_proprio,
#             # initial state
#             'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
#             # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero':  {'shape': (dim_s_extero,  1)}
#                 }, # , 's_all': [(9, 1)]},
#             'statedim': dim_s_proprio * 3,
#             'dt': dt,
#             'mass': 1.0/3.0,
#             'force_max':  1.0,
#             'force_min': -1.0,
#             'friction': 0.001,
#             'sysnoise': 1e-2,
#             'debug': False,
#             'dim_s_proprio': dim_s_proprio,
#             # 'length_ratio': 3./2.,
#             # 'm_mins': 0.05, # 0.1
#             # 'm_maxs': 0.4,  # 0.3
#             'dim_s_extero': 3,
#             'numsegs': 3,
#             'segradii': np.array([0.1,0.093,0.079]),
#             'm_mins': [ 0.10] * dim_s_proprio,
#             'm_maxs': [ 0.30] * dim_s_proprio,
#             's_mins': [ 0.10] * dim_s_proprio, # fixme all sensors
#             's_maxs': [ 0.30] * dim_s_proprio,
#             'doplot': False, # True,
#             'minlag': 1,
#             'maxlag': 3, # 5
#             'mdl_w_input': 3.0,
#             'mdl_theta': 3e-2,
#             'lag_past': (-1, 0),
#             'lag_future': (-1, 0),
#             }
#         }
    
# # ROS system STDR
# def get_systemblock_stdr(dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1):
#     global np, STDRCircularBlock2
#     return {
#         'block': STDRCircularBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero': {'shape': (dim_s_extero, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'ros': True,
#             'dt': dt,
#             'm_mins': [-0.3, 0], # [-0.1] * dim_s_proprio,
#             'm_maxs': [0.3, np.pi/4.0],    # [ 0.1] * dim_s_proprio,
#             'dim_s_proprio': dim_s_proprio, 
#             'dim_s_extero': dim_s_extero,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 1, # ha
#             'maxlag': 4, # 5
#             'lag_past': (-1, 0),
#             'lag_future': (-1, 0),
#             }
#         }

# # ROS system using lpzrobots' roscontroller to interact with the 'Barrel'
# def get_systemblock_lpzbarrel(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.01):
#     global LPZBarrelBlock2
#     systemblock_lpz = {
#         'block': LPZBarrelBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero': {'shape': (dim_s_extero, 1)}
#                 }, # , 's_all': [(9, 1)]},
#             'ros': True,
#             'dt': dt,
#             'm_mins': [-1.] * dim_s_proprio,
#             'm_maxs': [ 1.] * dim_s_proprio,
#             'dim_s_proprio': dim_s_proprio, 
#             'dim_s_extero': dim_s_extero,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 2, # 1, # 5, 4
#             'maxlag': 6, # 2,
#             # model parameters
#             'mdl_w_input': 0.5,
#             'mdl_w_bias': 0.1,
#             # 'mdl_theta': 5e-3, # 2.5e-1,
#             'mdl_theta': 2e-1, # 2.5e-1,
#             'mdl_eta': 5e-4,
            
#             # 'mdl_theta': 5e-2, # 2.5e-1,
#             # 'mdl_eta': 5e-4, # 5e-4,
            
#             # 'mdl_theta': 1e-2, # 2.5e-1,
#             # 'mdl_eta': 3e-4, # 5e-4,
            
#             # 'mdl_eta': 1e-4, # 5e-4,
#             'mdl_spectral_radius': 0.8,
#             'mdl_tau': 0.3,
#             # 'mdl_spectral_radius': 0.1, # 0.999,
#             # 'mdl_tau': 0.9,
#             'mdl_perf_measure': meas.square, # .identity
#             'mdl_perf_model_type': 'lowpass',
#             'mdl_coeff_a': 0.2,
#             # 'mdl_perf_model_type': 'resforce',
#             'mdl_mdltr_type': 'bin_joint_any',
#             # 
#             # 'lag_past':   (-1, 0), # down to -6
#             'lag_past':   (-3, -2), # down to -6
#             'lag_future': (-1, 0),
#             # target parameters
#             # 'target_f': 2.9,
#             # 'target_f': 1.45,
#             'target_f': 0.725 * 0.5,
#             }
#         }
#     return systemblock_lpz

# # systemblock_lpzbarrel = get_systemblock_lpzbarrel(dt = dt)

# # ROS system using the Sphero
# def get_systemblock_sphero(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.05):
#     global SpheroBlock2
#     systemblock_sphero = {
#         'block': SpheroBlock2,
#         'params': {
#             'id': 'robot1',
#             'debug': False,
#             'blocksize': 1, # FIXME: make pm blocksize aware!
#             'inputs': {'u': {'bus': 'pre_l0/pre'}},
#             'outputs': {
#                 's_proprio': {'shape': (dim_s_proprio, 1)},
#                 's_extero': {'shape': (dim_s_extero, 1)}
#                 }, # , 's_all': [(9, 1)]},
#                 'ros': True,
#                 'dt': dt,
#             'm_mins': [-0.75] * dim_s_proprio,
#             'm_maxs': [ 0.75] * dim_s_proprio,
#             'dim_s_proprio': dim_s_proprio, 
#             'dim_s_extero': dim_s_extero,   
#             'outdict': {},
#             'smdict': {},
#             'minlag': 2, # 2, # 4, # 2
#             'maxlag': 5,
#             'lag_past': (-2, -1),
#             'lag_future': (-1, 0),
#             }
#         }
#     return systemblock_sphero

# # systemblock_sphero = get_systemblock_sphero()

# get_systemblock = {
#     'pm': partial(get_systemblock_pm, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
#     'sa': partial(get_systemblock_sa, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
#     'bha': partial(get_systemblock_bha, dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1),
#     'lpzbarrel': partial(get_systemblock_lpzbarrel, dim_s_proprio = 2, dim_s_extero = 1, dt = 2.0/92.0), # 0.025),
#     'stdr': partial(get_systemblock_stdr, dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1),
#     'sphero': partial(get_systemblock_sphero, dim_s_proprio = 2, dim_s_extero = 1, dt = 0.0167),
#     }
    

################################################################################
# experiment variations
# - algo
# - system
# - system order
# - dimensions
# - number of modalities

# systemblock   = systemblock_lpzbarrel
# lag = 6 # 5, 4, 2 # 2 or 3 worked with lpzbarrel, dt = 0.05
# systemblock   = get_systemblock[sysname]()
# lag           = 1
# systemblock = get_systemblock[sysname](
#     lag = 2, dim_s0 = 2, dim_s1 = 2)

# get systemblock
systemblock = lconf['systemblock']

dim_s0 = systemblock['params']['dims']['s0']['dim']
dim_s1 = systemblock['params']['dims']['s1']['dim']
dim_s_hidden_debug = lconf['dim_s_hidden_debug']
m_mins = np.array([systemblock['params']['m_mins']]).T
m_maxs = np.array([systemblock['params']['m_maxs']]).T

dt = systemblock['params']['dt']

# model config defaults
mdl_cnf = lconf['mdl_cnf']
tgt_cnf = lconf['tgt_cnf']
    
# update default model config with system specific values
# for k in [
#         'mdl_w_input', 'mdl_w_bias', 'mdl_theta', 'mdl_eta',
#         'mdl_spectral_radius', ' mdl_tau', 'mdl_mdltr_type',
#         'mdl_mdltr_thr', 'mdl_perf_measure', 'mdl_perf_model_type',
#         'mdl_wgt_thr', 'mdl_coeff_a']:
for k in list(mdl_cnf.keys()):
    if k in systemblock['params']:
        mdl_cnf[k] = systemblock['params'][k]

# print "mdl_cnf = {"
# for k, v in mdl_cnf.items():
#     print "    %s = %s" % (k, v)


# update default target config with system specific values
for k in tgt_cnf:
    if k in systemblock['params']:
        tgt_cnf[k] = systemblock['params'][k]

# print "tgt_cnf = {"
# for k, v in tgt_cnf.items():
#     print "    %s = %s" % (k, v)
        
# algo = 'knn' #
# algo = 'gmm' #
# algo = 'igmm' #
# algo = 'hebbsom'
# algo = 'soesgp'
# algo = 'storkgp'
# algo = 'resrls'
# algo = 'homeokinesis'


# algo = 'res_eh'
algo = lconf['algo']
lag = systemblock['params']['lag']

# eh model in principle only needs
#  1) current goal prediction (pre_l1), current measurement (meas_l0) on the
#     block input
# lag past -lag-laglen+1 up to -lag+1, this goes directly and only into
# smpSHL.learnEH and not into smpModel

# lag_past = (-3, -2) #
# lag_past = (-3, -2) #
# lag future is a single time step now, multi-step prediction is
# difficult to formulate straightaway
lag_future = systemblock['params']['lag_future'] # (-1, 0)

# lpzbarrel
lag_past = systemblock['params']['lag_past'] # (-1, -0) # good
# lag_past = (-2, -1) # good
# lag_past = (-3, -2) # good
# lag_past = (-4, -3) # best
# lag_past = (-5, -4) # not good
# lag_past = (-4, -2)
# lag_past = (-6, -2) #
# lag_past = (-20, -1)

minlag = max(1, -max(lag_past[1], lag_future[1]))
maxlag = 1 - min(lag_past[0], lag_future[0])
maxlag_x = 120
laglen = maxlag - minlag

# print "minlag", minlag, "maxlag", maxlag

eta = lconf['eta']

desc = """An exploration and learning episode of {0} time steps of an
agent learning motor skills with the exploratory Hebbian model. The
task is a again a sequence uniformly random
goals. After washout, this learner is slower to pick up on the target
signal, as compared with the previous two models. This can be
expected from the fact that the learning rule does not convey the sign
or magnitude of the error but only wether the error magnitude has
decreased compared to the prediction. There is no noticable difference
in error levels in the testing phase at the end of the
episode.""".format(numsteps)

# motivations
from smp_graphs.utils_conf import dm_motivations
motivations = dm_motivations(m_mins, m_maxs, dim_s0, dt)
motivation_i = lconf['motivation_i']

# motivations eh local moves
lmotivation = motivations[motivation_i]
if lmotivation[1]['params']['models']['goal']['type'] == 'random_uniform':
    lmotivation[1]['params']['rate'] = lconf['motivation_rate']
    
def plot_timeseries_block(l0 = 'pre_l0', l1 = 'pre_l1', blocksize = 1):
    global partial
    global PlotBlock2, numsteps, timeseries, saveplot
    global algo, sysname, lag, lag_past, lag_future
    global dim_s1, dim_s0, dim_s_hidden_debug
    global motivations, motivation_i, lmotivation
    # global expr_number, expr_name
    goal = motivations[motivation_i][1]['params']['models']['goal']['type']
    expr_name = 'Experiment 27'
    return {
    'block': PlotBlock2,
    'params': {
        'blocksize': numsteps, # 1000, # blocksize,
        'saveplot': saveplot,
        'savetype': 'jpg',
        'title': '%s\ndev-model EH, algo %s, sys %s(dim_p=%d), goal %s, lag %s, tap- %s, tap+ %s' % (
            'Learning episode timeseries', algo, sysname, dim_s0, goal, lag, lag_past, lag_future),
        'title_pos': 'top_out',
        'hspace': 0.3,
        'desc': """An {1} agent learning to control a {0}-dimensional {2}
            system using the {3} low-level algorithm.""".format(
                dim_s0, 'eh', sysname, algo.replace('_', '\_')),
        'inputs': {
            'goals': {'bus': '%s/pre' % (l1,), 'shape': (dim_s0, blocksize)},
            'pre':   {'bus': '%s/pre' % (l0,), 'shape': (dim_s0, blocksize)},
            'err':   {'bus': '%s/err' % (l0,), 'shape': (dim_s0, blocksize)},
            'tgt':   {'bus': '%s/tgt' % (l0,), 'shape': (dim_s0, blocksize)},
            'hidden':   {'bus': '%s/hidden' % (l0,), 'shape': (dim_s_hidden_debug, blocksize)},
            'wo_norm':   {'bus': '%s/wo_norm' % (l0,), 'shape': (1, blocksize)},
            's0':    {'bus': 'robot1/s0', 'shape': (dim_s0, blocksize)},
            's1':     {'bus': 'robot1/s1',  'shape': (dim_s1, blocksize)},
            },
        'subplots': [
            
            # [
            #     {'input': ['goals', 's0'], 'plot': partial(timeseries, marker='.')},
            # ],
            # [
            #     {'input': ['goals', 'pre'], 'plot': partial(timeseries, marker='.')},
            # ],
            
            [
                {
                    'input': ['err',],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Momentary error and reward',
                    'legend': {'e(s_p)': 0},
                    # 'xticks': False,
                    'xticklabels': False,
                    'ylabel': 'Reward',
                }
            ],
            
            [
                {
                    'input': ['goals', 's0', 'pre'],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Goal, state and prediction',
                    'legend': {'Goal': 0, 'State': dim_s0, 'State_p': 2*dim_s0},
                    # 'xticks': False,
                    'xticklabels': False,
                    'ylabel': 'State',
                }
            ],
            
            [
                {
                    'input': ['hidden'],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Hidden activation of reservoir $\mathbf{r}$ (partial)',
                    'legend': {'$\mathbf{r}$': 0},
                    'xticklabels': False,
                    'ylabel': 'Activation',
                }
            ],
            
            [
                {
                    'input': ['wo_norm'],
                    'plot': partial(timeseries, marker='.'),
                    'title': 'Model %s parameter norm (accumulated adaptation)' % (algo),
                    'legend': {'$|\mathbf{W}_o|$': 0},
                    'xlabel': 'Time step [k]',
                    'ylabel': 'Weight norm',
                }
            ],
            
            # [
            #     {'input': ['pre', 's0'], 'plot': partial(timeseries, marker='.')},
            # ],
            
            # [
            #     {'input': ['pre'], 'plot': timeseries},
            # ],
            
            # [
            #     {'input': ['s0', 's1'], 'plot': timeseries},
            # ],
            
        ]}
    }

"""
sweep system subgraph
 - sweep block is an open-loop system (data source) itself
 - system block is the system we want to sweep
"""
sweepsys_steps = 6
sweepsys_input_flat = np.power(sweepsys_steps, dim_s0)
sweepsys = ('robot0', copy.deepcopy(systemblock))
sweepsys[1]['params']['blocksize'] = sweepsys_input_flat
sweepsys[1]['params']['debug'] = False
sweepsys[1]['params']['inputs'] = {'u': {'bus': 'sweepsys_grid/meshgrid'}}
sweepsys[1]['params']['outputs']['s0']['shape'] = (dim_s0, sweepsys_input_flat)
sweepsys[1]['params']['outputs']['s1']['shape']  = (dim_s1, sweepsys_input_flat)

sweepmdl_steps = 1000
sweepmdl_input_flat = sweepmdl_steps # np.power(sweepmdl_steps, dim_s0 * 2)
sweepmdl_func = f_random_uniform

# sweepmdl_steps = 3
# sweepmdl_input_flat = np.power(sweepmdl_steps, dim_s0 * 2)
# sweepmdl_func = f_meshgrid

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': sweepsys_input_flat,  # inner numsteps when used as loopblock (sideways time)
        'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
        'blockphase': [0],        # phase = 0
        'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat), 'buscopy': 'sweepsys_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s0)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
                    'func': f_meshgrid
                    },
                }),
                
                # sys to sweep
                sweepsys,

            ]),
        }
    }

loopblock_model = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': sweepmdl_input_flat,  # inner numsteps when used as loopblock (sideways time)
        'blocksize': 1,           # compute single steps, has to be 1 so inner cnt is correct etc
        'blockphase': [0],        # phase = 0
        'outputs': {
            'pre': {
                'shape': (dim_s0 * 2, sweepmdl_input_flat),
                'buscopy': 'sweepmdl_grid/meshgrid'}},
        # subgraph
        'graph': OrderedDict([
            # model sweep input
            ('sweepmdl_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'blocksize': sweepmdl_input_flat,
                    'inputs': {
                        # 'ranges': {'val': np.array([[-1, 1], [-1e-0, 1e-0]])},
                        'ranges': {
                            'val': np.array([[m_mins[0], m_maxs[0]]] * dim_s0 * 2)},
                            # 'val': np.vstack((
                            #     np.array([[m_mins[0], m_maxs[0]]] * dim_s0),
                            #     np.array([[-2.0,      1.0]]       * dim_s0),
                            #     ))},
                        'steps':  {'val': sweepmdl_steps},
                        },
                    'outputs': {
                        'meshgrid': {
                            'shape': (
                                dim_s0 * 2,
                                sweepmdl_input_flat,)}},
                    'func': sweepmdl_func,
                    },
                }),
                
            # # model sweep input
            # ('sweepmdl_grid_goal', {
            #     'block': FuncBlock2,
            #     'params': {
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             'ranges': {'val': np.array([[-1, 1]])},
            #             'steps':  {'val': sweepmdl_steps},
            #             },
            #         'outputs': {'pre': {'shape': (dim_s0, sweepmdl_input_flat)}},
            #         'func': f_meshgrid,
            #         },
            #     }),
                
            # # model sweep input
            # ('sweepmdl_grid_err', {
            #     'block': FuncBlock2,
            #     'params': {
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             'ranges': {'val': np.array([[-1e-0, 1e-0]])},
            #             'steps':  {'val': sweepmdl_steps},
            #             },
            #         'outputs': {'pre': {'shape': (dim_s0, sweepmdl_input_flat)}},
            #         'func': f_meshgrid,
            #         },
            #     }),

            ('sweep_slice', {
                'block': SliceBlock2,
                'params': {
                    'id': 'sweep_slice',
                    'blocksize': sweepmdl_input_flat,
                    # puppy sensors
                    'inputs': {
                        'x': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim_s0 * 2, sweepmdl_input_flat)}},
                    'slices': {
                        'x': {
                            'goals': slice(0, dim_s0),
                            'errs':  slice(dim_s0, dim_s0 * 2)},
                        },
                    # 'slices': ,
                    }
                }),

            # model sweep
            # learner: basic actinf predictor proprio space learn_proprio_base_0
            ('pre_l0_test', {
                'block': ModelBlock2,
                'params': {
                    'blocksize': sweepmdl_input_flat,
                    'blockphase': [0],
                    'debug': False, # True,
                    'lag': minlag,
                    'eta': eta, # 0.3,
                    'inputs': {
                        # descending prediction
                        'pre_l1': {
                            'bus': 'sweep_slice/x_goals',
                            'shape': (dim_s0,sweepmdl_input_flat)},
                        # ascending prediction error
                        'pre_l0': {
                            'bus': 'sweep_slice/x_errs',
                            'shape': (dim_s0, sweepmdl_input_flat)},
                        # ascending prediction error
                        'prerr_l0': {
                            'bus': 'pre_l0_test/err',
                            'shape': (dim_s0, minlag+1), 'lag': minlag},
                        # measurement
                        'meas_l0': {
                            'val': np.array([[-float('Inf') for i in range(sweepmdl_input_flat)]] * dim_s0),
                            # 'bus': 'robot1/s0',
                            'shape': (dim_s0, sweepmdl_input_flat)}},
                    'outputs': {
                        'pre': {'shape': (dim_s0, sweepmdl_input_flat)},
                        'err': {'shape': (dim_s0, sweepmdl_input_flat)},
                        'tgt': {'shape': (dim_s0, sweepmdl_input_flat)},
                        },
                    'models': {
                        # 'fwd': {'type': 'actinf_m1', 'algo': algo, 'idim': dim_s0 * 2, 'odim': dim},
                        'fwd': {
                            'type': 'actinf_m1', 'algo': 'copy',
                            'copyid': 'pre_l0', 'idim': dim_s0 * 2, 'odim': dim_s0},
                        },
                    'rate': 1,
                    },
                }),


            ('pre_l0_combined', {
                'block': StackBlock2,
                'params': {
                    'inputs': {
                        # 'goals': {
                        #     'bus': 'sweepmdl_grid_goal/pre',
                        #     'shape': (dim_s0, sweepmdl_input_flat),
                        #     },
                        # 'errs':  {
                        #     'bus': 'sweepmdl_grid_err/pre',
                        #     'shape': (dim_s0, sweepmdl_input_flat),
                        #     },
                        'goalerrs': {
                            'bus': 'sweepmdl_grid/meshgrid',
                            'shape': (dim_s0 * 2, sweepmdl_input_flat),
                            },
                        'pres':  {
                            'bus': 'pre_l0_test/pre',
                            'shape': (dim_s0, sweepmdl_input_flat),
                            },
                        },
                    'outputs': {
                        'y': {'shape': (dim_s0 * 3, sweepmdl_input_flat)}},
                    }
                }),
                
            # # plot timeseries
            # ('plot_ts',
            #      plot_timeseries_block(
            #          l0 = 'pre_l0_test',
            #          l1 = 'sweepmdl_grid_goal',
            #          blocksize = sweepmdl_input_flat)),
                        
            # # plot model sweep 1d
            # ('plot_model_sweep', {
            #     'block': ImgPlotBlock2,
            #     'params': {
            #         'id': 'plot_model_sweep',
            #         'logging': False,
            #         'saveplot': saveplot,
            #         'debug': False,
            #         'blocksize': sweepmdl_input_flat,
            #         'inputs': {
            #             # 'sweepin_goal': {
            #             #     'bus': 'sweepmdl_grid_goal/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             # 'sweepin_err':  {
            #             #     'bus': 'sweepmdl_grid_err/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             # 'sweepout_mdl':  {
            #             #     'bus': 'pre_l0_test/pre',
            #             #     'shape': (dim_s0, sweepmdl_input_flat)},
            #             'all': {
            #                 'bus': 'pre_l0_combined/y',
            #                 'shape': (dim_s0 * 3, sweepmdl_input_flat),
            #                 },
            #             },
            #         'outputs': {},
            #         'wspace': 0.5,
            #         'hspace': 0.5,
            #         # with one subplot and reshape
            #         'subplots': [
            #             [
            #                 {
            #                     'input': ['all'],
            #                     'shape': (dim_s0 * 3, sweepmdl_input_flat),
            #                     'ndslice': [(slice(None), slice(None))],
            #                     # 'vmin': -1.0, 'vmax': 1.0,
            #                     # 'vmin': 0.1, 'vmax': 0.3,
            #                     'cmap': 'RdGy',
            #                     'dimstack': {
            #                         'x': range(2*dim-1, dim - 1, -1),
            #                         'y': range(dim-1,   -1     , -1)},
            #                     'digitize': {'argdims': range(0, dim_s0 * 2), 'valdim': 2*dim+i, 'numbins': 2},
            #                 } for i in range(dim_s0)],

            #             ],
            #         },
            #     }),
            ]),
        }
    }
        

# main graph
graph = OrderedDict([
    # # sweep system
    # ("sweepsys", {
    #     'debug': False,
    #     'block': SeqLoopBlock2,
    #     'params': {
    #         'id': 'sweepsys',
    #         # loop specification, check hierarchical block to completely
    #         # pass on the contained in/out space?
    #         'blocksize': numsteps, # execution cycle, same as global numsteps
    #         'blockphase': [1],     # execute on first time step only
    #         'numsteps':  numsteps,          # numsteps      / loopblocksize = looplength
    #         'loopblocksize': loopblocksize, # loopblocksize * looplength    = numsteps
    #         # can't do this dynamically yet without changing init passes
    #         'outputs': {'meshgrid': {'shape': (dim_s0, sweepsys_input_flat)}},
    #         'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
    #         'loopmode': 'sequential',
    #         'loopblock': loopblock,
    #     },
    # }),

    # # plot the system sweep result
    # ('plot_sweep', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'debug': False,
    #         'blocksize': numsteps, # sweepsys_input_flat,
    #         'title': 'system sweep',
    #         'inputs': {
    #             'meshgrid':     {'bus': 'sweepsys_grid/meshgrid', 'shape': (dim_s0, sweepsys_input_flat)},
    #             's0':    {'bus': 'robot0/s0', 'shape': (dim_s0, sweepsys_input_flat)},
    #             's1':     {'bus': 'robot0/s1', 'shape': (dim_s1, sweepsys_input_flat)},
    #             },
    #             'hspace': 0.2,
    #             'subplots': [
    #                 [
    #                     {'input': ['meshgrid'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s0'], 'plot': timeseries},
    #                 ],
    #                 [
    #                     {'input': ['s1'], 'plot': timeseries},
    #                 ],
    #                 ],
    #         }
    #     }),

    # system block from definition elsewhere
    ('robot1', systemblock),
    
    # learning experiment
    ('brain_learn_proprio', {
        'block': Block2,
        'params': {
            'graph': OrderedDict([
                
                # generic count block
                ('cnt', {
                    'block': CountBlock2,
                    'params': {
                        'blocksize': 1,
                        'debug': False,
                        'inputs': {},
                        'outputs': {'x': {'shape': (dim_s0, 1)}},
                        },
                    }),

                # generic block configured with subgraph on demand
                ('pre_l1', {
                    'block': Block2,
                    'params': {
                        'numsteps': numsteps,
                        'blocksize': 1,
                        # 'subgraph': OrderedDict([motivations[0]]),
                        'subgraph': OrderedDict([lmotivation]),
                        'subgraph_rewrite_id': False,
                    }
                }),
                
                # # goal sampler (motivation) sample_discrete_uniform_goal
                # ('pre_l1', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'ros': ros,
                #         'inputs': {                        
                #             'lo': {'val': m_mins, 'shape': (dim_s0, 1)},
                #             'hi': {'val': m_maxs, 'shape': (dim_s0, 1)},
                #             },
                #         'outputs': {'pre': {'shape': (dim_s0, 1)}},
                #         'models': {
                #             'goal': {'type': 'random_uniform'}
                #             },
                #         'rate': 500, # int(numsteps/30), # 1000,
                #         },
                #     }),

                # # goal sampler (motivation) sample single angle and duplicate with offset pi/2
                # ('pre_l1', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'ros': ros,
                #         'inputs': {                        
                #             'lo': {'val': -1.0, 'shape': (dim_s0, 1)},
                #             'hi': {'val':  1.0, 'shape': (dim_s0, 1)},
                #             'meas_l0': {
                #                 'bus': 'robot1/s0',
                #                 'shape': (dim_s0, maxlag), 'lag': range(-laglen, 0)},
                #         },
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'prerr': {'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                #             'goal': {'type': 'random_uniform_pi_2'}
                #             },
                #         'rate': 10, # int(numsteps/30), # 1000,
                #         },
                #     }),
                    
                # # a random number generator, mapping const input to hi
                # ('pre_l1', {
                #     'block': FuncBlock2,
                #     'params': {
                #         'id': 'pre_l1',
                #         'outputs': {'pre': {'shape': (dim_s0, 1)}},
                #         'debug': False,
                #         'blocksize': 1,
                #         'ros': ros,
                #         # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #         # recurrent connection
                #         'inputs': {
                #             'x': {'bus': 'cnt/x'},
                #             # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
                #             # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.45]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 10.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 3.4 * dt}, # good with soesgp and eta = 0.7
                            
                #             # 'f': {'val': np.array([[0.23539]]).T * 3.0 * dt}, # good with soesgp and eta = 0.7
                            
                #             # 'f': {'val': np.array([[0.23539]]).T * 2.9 * dt}, # barrel
                            
                #             # 'f': {'val': np.array([[0.23539]]).T * 7.23 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 2.0 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 2.5 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.23539]]).T * 1.5 * dt}, # good with soesgp and eta = 0.7

                #             # # pointmass etc
                #             # 'f': {'val': np.array([[0.23539]]).T * 0.05 * dt}, # good with soesgp and eta = 0.7

                #             # configurable
                #             'f': {'val': np.array([[0.23539]]).T * tgt_cnf['target_f'] * dt}, # good with soesgp and eta = 0.7

                #             # 'f': {'val': np.array([[0.23539, 0.3148]]).T * 0.05 * dt}, # good with soesgp and eta = 0.7
                #             # 'f': {'val': np.array([[0.14, 0.14]]).T * 1.0},
                #             # 'f': {'val': np.array([[0.82, 0.82]]).T},
                #             # 'f': {'val': np.array([[0.745, 0.745]]).T},
                #             # 'f': {'val': np.array([[0.7, 0.7]]).T},
                #             # 'f': {'val': np.array([[0.65, 0.65]]).T},
                #             # 'f': {'val': np.array([[0.39, 0.39]]).T},
                #             # 'f': {'val': np.array([[0.37, 0.37]]).T},
                #             # 'f': {'val': np.array([[0.325, 0.325]]).T},
                #             # 'f': {'val': np.array([[0.31, 0.31]]).T},
                #             # 'f': {'val': np.array([[0.19, 0.19]]).T},
                #             # 'f': {'val': np.array([[0.18, 0.181]]).T},
                #             # 'f': {'val': np.array([[0.171, 0.171]]).T},
                #             # 'f': {'val': np.array([[0.161, 0.161]]).T},
                #             # 'f': {'val': np.array([[0.151, 0.151]]).T},
                #             # 'f': {'val': np.array([[0.141, 0.141]]).T},
                #             # stay in place
                #             # 'f': {'val': np.array([[0.1, 0.1]]).T},
                #             # 'f': {'val': np.array([[0.24, 0.24]]).T},
                #             # 'sigma': {'val': np.array([[0.001, 0.002]]).T}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #             'sigma': {'val': np.ones((dim_s0, 1)) * 1e-6}, # {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
                #             'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
                #             'amp': {'val': (m_maxs - m_mins)/2.0},
                #         }, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
                #         'func': f_sin_noise,
                #     },
                # }),
                
                # # dev model actinf_m1: learner is basic actinf predictor proprio space learn_proprio_base_0
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'debug': False,
                #         'lag': minlag,
                #         'eta': eta, # 3.7,
                #         'ros': ros,
                #         # FIXME: relative shift = minlag, block length the maxlag
                #         'inputs': {
                #             # descending prediction
                #             'pre_l1': {
                #                 'bus': 'pre_l1/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': range(-maxlag, -minlag)}, # lag}, # m1
                #                 # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)}, # lag}, # m2
                #             # ascending prediction error
                #             'pre_l0': {
                #                 'bus': 'pre_l0/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)}, # lag},
                #             # ascending prediction error
                #             'prerr_l0': {
                #                 'bus': 'pre_l0/err',
                #                 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)}, # lag},
                #             # measurement
                #             'meas_l0': {
                #                 'bus': 'robot1/s0',
                #                 'shape': (dim_s0, maxlag), 'lag': range(-laglen, 0)}
                #             },
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'err': {'shape': (dim_s0, 1)},
                #             'tgt': {'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                            
                #             'm1': {
                #                 'type': 'actinf_m1',
                #                 'algo': algo,
                #                 'idim': dim_s0 * laglen * 2,
                #                 'odim': dim_s0 * laglen,
                #                 'laglen': laglen,
                #                 'eta': eta
                #                 },
                                
                #             # 'm2': {
                #             #     'type': 'actinf_m2',
                #             #     'algo': algo,
                #             #     'idim': dim_s0 * laglen,
                #             #     'odim': dim_s0 * laglen,
                #             #     'laglen': laglen,
                #             #     'eta': eta
                #             #     },
                            
                #             # 'm3': {
                #             #     'type': 'actinf_m3',
                #             #     'algo': algo,
                #             #     'idim': dim_s0 * laglen * 2,
                #             #     'odim': dim_s0 * laglen,
                #             #     'laglen': laglen,
                #             #     'eta': eta
                #             #     },
                                
                #             },
                #         'rate': 1,
                #         },
                #     }),

                # dev model uniform random sampler
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'inputs': {                        
                #             'lo': {'val': np.array([m_mins]).T, 'shape': (dim_s0, 1)},
                #             'hi': {'val': np.array([m_maxs]).T, 'shape': (dim_s0, 1)},
                #             },
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'err': {'val': np.zeros((dim_s0, 1)), 'shape': (dim_s0, 1)},
                #             'tgt': {'val': np.zeros((dim_s0, 1)), 'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                #             'goal': {'type': 'random_uniform'}
                #             },
                #         'rate': 50,
                #         },
                #     }),
                    
                # # dev model homeokinesis
                # ('pre_l0', {
                #     'block': ModelBlock2,
                #     'params': {
                #         'blocksize': 1,
                #         'blockphase': [0],
                #         'debug': False,
                #         'lag': minlag,
                #         'eta': eta, # 3.7,
                #         'ros': ros,
                #         'inputs': {
                #             # descending prediction
                #             'pre_l1': {
                #                 'bus': 'pre_l1/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # ascending prediction error
                #             'pre_l0': {
                #                 'bus': 'pre_l0/pre',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # ascending prediction error
                #             'prerr_l0': {
                #                 'bus': 'pre_l0/err',
                #                 'shape': (dim_s0, maxlag), 'lag': minlag},
                #             # measurement
                #             'meas_l0': {
                #                 'bus': 'robot1/s0', 'shape': (dim_s0, maxlag)}},
                #         'outputs': {
                #             'pre': {'shape': (dim_s0, 1)},
                #             'err': {'shape': (dim_s0, 1)},
                #             'tgt': {'shape': (dim_s0, 1)},
                #             },
                #         'models': {
                #             'hk': {
                #                 'type': 'homeokinesis', 'algo': algo, 'mode': 'hk', 'idim': dim_s0, 'odim': dim_s0,
                #                 'minlag': minlag, 'maxlag': maxlag, 'laglen': laglen, 'm_mins': m_mins, 'm_maxs': m_maxs,
                #                 'epsA': 0.01, 'epsC': 0.03, 'creativity': 0.8},
                #             },
                #         'rate': 1,
                #         },
                #     }),
                    
                # dev model EH: direct inverse model learning with reward modulated hebbian updates and locally gaussian exploration
                ('pre_l0', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'debug': False,
                        'lag': minlag,
                        'eta': 1e-3, # 2e-3, # eta, # 3.7,
                        'ros': ros,
                        # FIXME: relative shift = minlag, block length the maxlag
                        'inputs': {
                            # descending prediction
                            'pre_l1': {
                                'bus': 'pre_l1/pre',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag, -minlag)},
                                'shape': (dim_s0, maxlag_x), 'lag': list(range(lag_past[0], lag_past[1]))},
                                
                            # ascending prediction error
                            'pre_l0': {
                                'bus': 'pre_l0/pre',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s0, maxlag), 'lag': list(range(lag_past[0] + 1, lag_past[1] + 1))},
                                
                            # ascending prediction error
                            'prerr_l0': {
                                'bus': 'pre_l0/err',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-maxlag + 1, -minlag + 1)},
                                'shape': (dim_s0, maxlag), 'lag': list(range(lag_past[0] + 1, lag_past[1] + 1))},
                                
                            # measurement
                            'meas_l0': {
                                'bus': 'robot1/s0',
                                # 'shape': (dim_s0, maxlag), 'lag': range(-laglen, 0)}
                                'shape': (dim_s0, maxlag_x), 'lag': list(range(lag_future[0], lag_future[1]))},
                                
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s0, 1)},
                            'err': {'shape': (dim_s0, 1)},
                            'perflp': {'shape': (dim_s0, 1)},
                            'tgt': {'shape': (dim_s0, 1)},
                            'hidden': {'shape': (dim_s_hidden_debug, 1)},
                            'wo_norm': {'shape': (1, 1)},
                        },
                        
                        'models': {
                            # eh model conf
                            'm1': {
                                'type': 'eh',
                                'lrname': 'eh',
                                'algo': algo,
                                'mdltr_type': mdl_cnf['mdl_mdltr_type'],
                                'mdltr_thr': mdl_cnf['mdl_mdltr_thr'],
                                'wgt_thr': mdl_cnf['mdl_wgt_thr'],
                                'perf_measure': mdl_cnf['mdl_perf_measure'],
                                'perf_model_type': mdl_cnf['mdl_perf_model_type'],
                                'memory': maxlag,
                                'laglen': laglen,
                                'minlag': minlag,
                                'maxlag': maxlag,
                                # model need's to know about past/future embedding
                                'lag_past': lag_past,
                                'lag_future': lag_future,
                                'idim': dim_s0 * 3, #* (lag_past[1] - lag_past[0]) * 3, # laglen
                                'odim': dim_s0 * (lag_future[1] - lag_future[0]), # laglen,
                                'eta': mdl_cnf['mdl_eta'],
                                'eta_init': 1e-3,
                                'modelsize': mdl_cnf['mdl_modelsize'],
                                'p': 0.1,
                                # 'g': 0.999,
                                'spectral_radius': mdl_cnf['mdl_spectral_radius'],
                                'tau': mdl_cnf['mdl_tau'],
                                # 'res_input_num': dim_s0 * laglen * 3,
                                # 'res_output_num': dim_s0 * laglen,
                                'res_feedback_scaling': 0.0,
                                'w_input': mdl_cnf['mdl_w_input'],
                                'w_bias': mdl_cnf['mdl_w_bias'],
                                'res_output_scaling': 1.0,
                                'nonlin_func': np.tanh,
                                'use_ip': 0,
                                # 'theta': 1e-1, # pm, sa
                                # 'theta': 3e-2, # bha
                                'theta': mdl_cnf['mdl_theta'],
                                'theta_state': 1e-2,
                                'coeff_a': mdl_cnf['mdl_coeff_a'],
                                'len_episode': numsteps,
                                'input_coupling_mtx_spec': {0: 1., 1: 1.},
                                'input_coupling': 'normal', # uniform, normal, sparse_uniform, sparse_normal, disjunct
                                'use_et': 0,
                                'et_winsize': 20,
                                'use_pre': 0,
                                'pre_inputs': [2],
                                'pre_delay': [10, 50, 100],
                                'use_wb': 0,
                                'wb_thr': 1.5,
                                'oversampling': 1,
                                # learning modulation
                                'n_washout': lconf['lm']['n_washout'],
                                'n_train': lconf['lm']['n_train'],
                                'n_test': lconf['lm']['n_test'],
                            },
                                                                
                        },
                        'rate': 1,
                    },
                }),
            # learn_proprio_e2p2e
            ]),
        }
    }),

    ################################################################################
    # use a sequential loop block to insert probes running orthogonally in time
    # blocksize:  seqloops blocksize from the outside, same as overall experiment
    # blockphase: points in the cnt % numsteps space when to execute
    # numsteps:      
    # loopblocksize: number of loop iterations = numsteps/loopblocksize 
    # ("sweepmodel", {
    #     'debug': False,
    #     'block': SeqLoopBlock2,
    #     'params': {
    #         'id': 'sweepmodel',
    #         'blocksize': numsteps, # execution cycle, same as global numsteps
    #         #                        execution phase, on first time step only
    #         # 'blockphase': [numsteps/2, numsteps-10],
    #         # 'blockphase': [int(i * numsteps)-1 for i in np.linspace(1.0/2, 1, 2)],
    #         'blockphase': [int(i * numsteps)-1 for i in np.linspace(1.0/1, 1, 1)],
    #         # 'blockphase': [0],
    #         'numsteps':  1, # numsteps,          # numsteps      / loopblocksize = looplength
    #         'loopblocksize': 1, #loopblocksize, # loopblocksize * looplength    = numsteps
    #         # can't do this dynamically yet without changing init passes
    #         'outputs': {'pre': {'shape': (dim_s0 * 2, sweepmdl_input_flat)}},
    #         'loop': [('none', {}) for i in range(2)], # lambda ref, i, obj: ('none', {}),
    #         'loopmode': 'sequential',
    #         'loopblock': loopblock_model,
    #     },
    # }),        
        
    # plot timeseries
    ('plot_ts', plot_timeseries_block(l0 = 'pre_l0', blocksize = numsteps)),
    
    ])
