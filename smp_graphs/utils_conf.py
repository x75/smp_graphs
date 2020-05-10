"""smp_graphs utils_conf

some utils for use in configuration files
"""

from functools import partial
from collections import OrderedDict

import numpy as np

from smp_base.measures import meas
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2

import logging
from smp_base.common import get_module_logger
logger = get_module_logger(modulename = 'utils_conf', loglevel = logging.INFO)

"""system block
 - a robot
"""
def get_systemblock_pm(
        dim_s0 = 2, dim_s1 = 2, dt = 0.1, lag = 1, **kwargs):
    """configuration utility function pointmass (pm)

    Generate configuration for pointmass system block using
    :mod:`smp_sys.systems.PointmassSys` and
    :mod:`smp_sys.systems.Pointmass2Sys`
    """
    global np, PointmassBlock2, meas
    logger.debug(
        'get_systemblock_pm: dim_s0 = %d, dt= %f, lag = %d, kwargs = %s' % (
            dim_s0, dt, lag, kwargs))

    # legacy argument handling
    if 'dim_s_proprio' in kwargs:
        dim_s0 = kwargs['dim_s_proprio']
    if 'dim_s_extero' in kwargs:
        dim_s1 = kwargs['dim_s_extero']

    # defaults
    dims = {
        # if we're lucky we can get away with explicit limits because they are obtained from s2s
        'm0': {'dim': dim_s0, 'dist': 0}, # , 'mins': [-1] * dim_s0, 'maxs': [1] * dim_s0
        's0': {'dim': dim_s0, 'dist': 0},
    }
    if dim_s1 is not None:
        dims['s1'] = {'dim': dim_s1, 'dist': 0}

    # tapping conf from lag
    if 'lag_past' in kwargs:
        lag_past = kwargs['lag_past']
    else:
        lag_past = (-lag -2, -lag - 1)
        
    if 'lag_future' in kwargs:
        lag_future = kwargs['lag_future']
    else:
        lag_future = (-1, 0)
        
    # print("dims = %s" % (dims, ))
        
    # FIXME: disentangle block conf from sys conf?
    sysconf = {
        'block': PointmassBlock2,
        'params': {
            # 'debug': True,
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'systype': 2,
            'sysdim': dim_s0,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s0 * 3, 1)),
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                # 's_proprio': {'shape': (dim_s0, 1), 'remap': 's0'},
                # 's_extero':  {'shape': (dim_s1, 1), 'remap': 's1'},
                's0': {'shape': (dim_s0, 1)},
                's1': {'shape': (dim_s1, 1)},
                }, # , 's_all': [(9, 1)]},
            # 'statedim': dim_s0 * 3,
            'dt': 0.1,
            # memory 
            'order': 1,
            # this is the real lag applied in the simulation
            'lag': lag,
            'mass': 1.0,
            # distortion
            'transfer': 0,
            'numelem': 1001,
            # distortion + memory
            'coupling_sigma': 1e-2,
            # external entropy
            'anoise_mean': 0.0,
            'anoise_std': 1e-3,
            'sysnoise': 1e-3,
            # other
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.01,
            'm_mins': [-1.0] * dim_s0,
            'm_maxs': [ 1.0] * dim_s0,
            'length_ratio': 3./2., # gain curve?
            
            # ground truth information for configuring the model, FIXME: make model autonomous with resp. to these params
            # memory, time: tapping
            # 'lag_past': (-4, -3),
            # 'lag_future': (-1, 0),
            'lag_past': lag_past,
            'lag_future': lag_future,
            
            # low-level params
            'mdl_modelsize': 300,
            'mdl_w_input': 1.0,
            'mdl_theta': 0.5e-1,
            'mdl_eta': 1e-3,
            'mdl_mdltr_type': 'cont_elem', #'bin_elem',
            'mdl_mdltr_thr': 0.005, # 0.01,
            'mdl_wgt_thr': 1.0, # 
            'mdl_perf_measure': meas.square, # .identity
            'mdl_perf_model_type': 'lowpass', # 'resforce',
            # target parameters
            'target_f': 0.05,
        }
    }

    # default dims
    sysconf['params']['dims'] = dims
    
    # if kwargs.has_key('numelem') or kwargs.has_key('h_numelem'):
    for k in [k for k in list(kwargs.keys()) if k in ['numelem', 'h_numelem']]:
        # h_numelem = kwargs['h_numelem']
        sysconf['params']['outputs'].update({'h':  {'shape': (dim_s0, kwargs[k]), 'trigger': 'trig/t1'}})
        sysconf['params']['numelem'] = kwargs[k]
        
    # update from kwargs
    # if kwargs.has_key('dims'):
    #     # print "updating dims", dims, kwargs['dims']
    #     dims.update(kwargs['dims'])
    #     # print "updated dims", dims
    # if kwargs.has_key('order'):
    #     order = kwargs['order']
    sysconf['params'].update(kwargs)

    return sysconf

def get_systemblock_sa(
        dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1, lag = 1, **kwargs):
    global np, SimplearmBlock2
    return {
        'block': SimplearmBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero,  1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'lag': lag,
            'lag_past': (-4, -3),
            'lag_future': (-1, 0),
            'mass': 1.0/3.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.001,
            'sysnoise': 5e-3,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            'length_ratio': 3./2.0,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            # 's_mins': [-1.00] * 9,
            # 's_maxs': [ 1.00] * 9,
            # 'm_mins': -1,
            # 'm_maxs': 1,
            'dim_s_extero': dim_s_extero,
            # 'minlag': 1,
            # 'maxlag': 2, # 5
            }
        }

def get_systemblock_bha(
        dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1, **kwargs):
    global np, BhasimulatedBlock2
    return {
        'block': BhasimulatedBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            # 'inputs': {'u': {'val': np.random.uniform(-1, 1, (3, numsteps))}},
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero,  1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'mass': 1.0/3.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.001,
            'sysnoise': 1e-2,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            # 'length_ratio': 3./2.,
            # 'm_mins': 0.05, # 0.1
            # 'm_maxs': 0.4,  # 0.3
            'dim_s_extero': 3,
            'numsegs': 3,
            'segradii': np.array([0.1,0.093,0.079]),
            'm_mins': [ 0.10] * dim_s_proprio,
            'm_maxs': [ 0.30] * dim_s_proprio,
            's_mins': [ 0.10] * dim_s_proprio, # fixme all sensors
            's_maxs': [ 0.30] * dim_s_proprio,
            'doplot': False,
            # 'minlag': 1,
            # 'maxlag': 3 # 5
            }
        }
    
# ROS system STDR
def get_systemblock_stdr(
        dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1, **kwargs):
    global np, STDRCircularBlock2
    return {
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
            'ros': True,
            'dt': dt,
            'm_mins': [-0.3, 0], # [-0.1] * dim_s_proprio,
            'm_maxs': [0.3, np.pi/4.0],    # [ 0.1] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            # 'minlag': 1, # ha
            # 'maxlag': 4, # 5
            }
        }

# ROS system using lpzrobots' roscontroller to interact with the 'Barrel'
def get_systemblock_lpzbarrel(
        dim_s_proprio = 2, dim_s_extero = 1, dt = 0.01, **kwargs):
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
            'ros': True,
            'dt': dt,
            'm_mins': [-1.] * dim_s_proprio,
            'm_maxs': [ 1.] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            # 'minlag': 2, # 1, # 5, 4
            # 'maxlag': 6, # 2,
            }
        }
    return systemblock_lpz

# ROS system using the Sphero
def get_systemblock_sphero(
        dim_s_proprio = 2, dim_s_extero = 1, dt = 0.05, **kwargs):
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
                'ros': True,
                'dt': dt,
            'm_mins': [-0.75] * dim_s_proprio,
            'm_maxs': [ 0.75] * dim_s_proprio,
            'dim_s_proprio': dim_s_proprio, 
            'dim_s_extero': dim_s_extero,   
            'outdict': {},
            'smdict': {},
            # 'minlag': 2, # 2, # 4, # 2
            # 'maxlag': 5,
            }
        }
    return systemblock_sphero

# add missing systems
get_systemblock = {
    'pm': partial(get_systemblock_pm, dim_s0 = 2, dim_s1 = 2, dt = 0.1),
    'sa': partial(get_systemblock_sa, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
    'bha': partial(get_systemblock_bha, dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1),
    'lpzbarrel': partial(get_systemblock_lpzbarrel, dim_s_proprio = 2, dim_s_extero = 1, dt = 2.0/92.0), # 0.025),
    'stdr': partial(get_systemblock_stdr, dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1),
    'sphero': partial(get_systemblock_sphero, dim_s_proprio = 2, dim_s_extero = 1, dt = 0.0167),
    }


# dm tools
from smp_graphs.block import FuncBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.funcs import f_sin_noise

def dm_motivations(m_mins, m_maxs, dim_s0, dt):
    motivations = [
        # goal sampler (motivation) sample_discrete_uniform_goal
        ('pre_l1', {
            'block': ModelBlock2,
            'params': {
                'blocksize': 1,
                'blockphase': [0],
                'inputs': {                        
                    'lo': {'val': m_mins * 1.0, 'shape': (dim_s0, 1)},
                    'hi': {'val': m_maxs * 1.0, 'shape': (dim_s0, 1)},
                },
                'outputs': {'pre': {'shape': (dim_s0, 1)}},
                'models': {
                    'goal': {'type': 'random_uniform'}
                },
                'rate': 40,
            },
        }),

    # goal sampler (motivation) sample_function_generator sinusoid
    ('pre_l1', {
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
    ('pre_l1', {
        'block': FuncBlock2,
        'params': {
            'id': 'pre_l1',
            'outputs': {'pre': {'shape': (dim_s0, 1)}},
            'debug': False,
            # 'ros': ros,
            'blocksize': 1,
            # recurrent connection
            'inputs': {
                'x': {'bus': 'cnt/x'},
                # 'f': {'val': np.array([[0.225]]).T * 5.0 * dt}, 
                # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, 
                # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
                # 'f': {'val': np.array([[0.45]]).T * 5.0 * dt}, 
                
                # barrel
                # 'f': {'val': np.array([[0.23539]]).T * 10.0 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 7.23 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 3.2 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 2.9 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 1.25 * dt}, 
                
                # pointmass
                'f': {'val': np.array([[0.23539]]).T * 0.4 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 0.1 * dt}, 
                # 'f': {'val': np.array([[0.23539]]).T * 0.05 * dt}, 
                
                # 'f': {'val': np.array([[0.23539, 0.2348, 0.14]]).T * 1.25 * dt}, 
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
            },
            'func': f_sin_noise,
        },
        })
        ]
    return motivations

################################################################################
# legacy motivations from dm_*.py

# legacy motivations from dm_imol

# # goal sampler (motivation) sample_function_generator sinusoid
# ('pre_l1', {
#     'block': ModelBlock2,
#     'params': {
#         'blocksize': 1,
#         'blockphase': [0],
#         'inputs': {
#             'x': {'bus': 'cnt/x'},
#             # pointmass
#             'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, 
#             'sigma': {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
#             'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
#             'amp': {'val': (m_maxs - m_mins)/2.0},
#         },
#         'outputs': {'pre': {'shape': (dim_s0, 1)}},
#         'models': {
#             'goal': {
#                 'type': 'function_generator',
#                 'func': f_sin_noise,
#             }
#         },
#         'rate': 1,
#     },
# }),
                
# # a random number generator, mapping const input to hi
# ('pre_l1', {
#     'block': FuncBlock2,
#     'params': {
#         'id': 'pre_l1',
#         'outputs': {'pre': {'shape': (dim_s0, 1)}},
#         'debug': False,
#         'ros': ros,
#         'blocksize': 1,
#         # recurrent connection
#         'inputs': {
#             'x': {'bus': 'cnt/x'},
#             # 'f': {'val': np.array([[0.2355, 0.2355]]).T * 1.0}, # good with knn and eta = 0.3
#             # 'f': {'val': np.array([[0.23538, 0.23538]]).T * 1.0}, 
#             # 'f': {'val': np.array([[0.45]]).T * 5.0 * dt}, 
#             # 'f': {'val': np.array([[0.225]]).T * 5.0 * dt}, 

#             # barrel
#             # 'f': {'val': np.array([[0.23539]]).T * 10.0 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 5.0 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 7.23 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 3.2 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 2.9 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 1.25 * dt}, 
                            
#             'f': {'val': np.array([[0.23539]]).T * 0.4 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 0.2 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 0.1 * dt}, 
#             # 'f': {'val': np.array([[0.23539]]).T * 0.05 * dt}, 
                            
#             # 'f': {'val': np.array([[0.23539, 0.2348, 0.14]]).T * 1.25 * dt}, 
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
#             'sigma': {'val': np.random.uniform(0, 0.01, (dim_s0, 1))},
#             'offset': {'val': m_mins + (m_maxs - m_mins)/2.0},
#             'amp': {'val': (m_maxs - m_mins)/2.0},
#         },
#         'func': f_sin_noise,
#     },
# }),
                
def get_subplots_expr0110(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    if 'mode' in kwargs:
        # mode = kwargs['mode']
        return get_subplots_expr0110_ts(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs)
    else:
        # mode = 'img'
        return get_subplots_expr0110_img(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs)

def get_subplots_expr0110_img(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    dim_s0 = rowiter
    rows = []
    for row in range(rowiter):
        cols = []
        for col in range(coliter):
            cols.append({
                'input': ['d3'], 'ndslice': (slice(scanlen), row, col),
                'shape': (dim_s0, scanlen), 'cmap': 'RdGy', 'title': 'Cross-correlation',
                'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                'yticks': False, 'yticklabels': None, 'ylabel': None,
                'xticks': (np.arange(scanlen) + 0.5).tolist(),
                'xticklabels': list(range(scanstart, scanstop)),
                'colorbar': True,
            })
        rows.append(cols)
    return rows

def get_subplots_expr0110_ts(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    from smp_base.plot import bar, timeseries
    rows = []
    for row in range(rowiter):
        cols = []
        for col in range(coliter):
            cols.append({
                'input': ['d3'],
                'ndslice': (slice(scanlen), row, col),
                'shape': (1, scanlen),
                # 'plot': partial(bar, width=0.5, orientation='vertical'),
                'plot': partial(timeseries, linestyle='', marker='o', markersize=10, alpha=0.5),
                'cmap': ['gray'],
                'title': None,
                # 'vmin': -1.0, 'vmax': 1.0,
                # 'vaxis': 'cols',
                # 'yticks': False,
                # 'yticklabels': None,
                'cross-correlation': None,
                # 'xticks': tuple((np.arange(scanlen) + 0).tolist()),
                # 'xticklabels': tuple(range(scanstart, scanstop)),
                'xlabel': 'time shift [k]',
                'ylabel': 'correlation coefficient [Pearson]',
                'legend': {'cross-correlation': 0},
                'xaxis': np.array(range(scanstart, scanstop)),
                # 'colorbar': True,
            })
        rows.append(cols)
    return rows


def get_subplots_expr0111(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    if 'mode' in kwargs and kwargs['mode'] == 'ts':
        # mode = kwargs['mode']
        return get_subplots_expr0111_ts(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs)
    elif 'mode' in kwargs and kwargs['mode'] == 'img':
        # mode = 'img'
        return get_subplots_expr0111_img(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs)
    else:
        return get_subplots_expr0111_img(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs)

def get_subplots_expr0111_img(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    dim_s0 = rowiter
    dim_m0 = coliter
    rows = []
    for row in range(rowiter):
        cols = []
        for col in range(coliter):
            cols.append({
                'input': ['d3'], 'ndslice': (slice(scanlen), row, col),
                'shape': (dim_s0, scanlen), 'cmap': 'RdGy', 'title': 'Cross-correlation',
                'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                'xticks': False, 'xlabel': None,
                'yticks': False, 'ylabel': None,
                # 'xaxis': range(scanstart, scanstop),
                'colorbar': True, 'colorbar_orientation': 'vertical',
            })
        rows.append(cols)
        
    for row in range(1):
        cols = []
        for col in range(coliter):
            cols.append({
                # mutual information scan
                'input': 'd1',
                # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                # 'xslice': (0, 1),
                'ndslice': (slice(scanlen), col, 0),
                'shape': (dim_s0, scanlen), 'cmap': 'Greys', 'title': 'Mutual information',
                'vmin': 0.0, 'vmax': 1.0, 'vaxis': 'cols',
                'xticks': (np.arange(scanlen) + 0.5).tolist(),
                'xticklabels': list(range(scanstart, scanstop)),
                'xlabel': 'time shift [steps]',
                'yticks': False, 'ylabel': None,
                'colorbar': True, 'colorbar_orientation': 'vertical',
            })
            
        rows.append(cols)

    return rows

def get_subplots_expr0111_ts(rowiter, coliter, scanlen, scanstart, scanstop, *args, **kwargs):
    from smp_base.plot import bar, timeseries
    dim_s0 = rowiter
    dim_m0 = coliter
    rows = []
    for row in range(rowiter):
        cols = []
        for col in range(coliter):
            cols.append({
                'input': ['d4', 'd1'],
                'plot': partial(timeseries, linestyle='', marker='o', markersize=10, alpha=0.5),
                'ndslice': [(slice(scanlen), row, col), (slice(scanlen), row, col)],
                'shape': (dim_s0, scanlen),
                'cmap': ['glasbey_warm'],
                'title': None,
                'title_pos': 'top_out',
                # 'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                'ylim': (-1, 1),
                'yticks': (-1, -0.5, 0, 0.5, 1),
                'xlabel': 'time shift [k]',
                'ylabel': 'person / normalized MI',
                'legend': {'cross-correlation': 0, 'mutual information': 1},
                'xaxis': np.array(range(scanstart, scanstop)),
                # 'colorbar': True, 'colorbar_orientation': 'vertical',
            })
        rows.append(cols)
        
    # for row in range(1):
    #     cols = []
    #     for col in range(coliter):
    #         cols.append({
    #             'input': 'd1',
    #             'plot': partial(timeseries, linestyle='', marker='o', markersize=10, alpha=0.5),
    #             # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
    #             # 'xslice': (0, 1),
    #             'ndslice': (slice(scanlen), col, 0),
    #             'shape': (dim_s0, scanlen),
    #             'cmap': ['gray'],
    #             'title': 'Mutual information',
    #             # 'vmin': 0.0, 'vmax': 1.0, 'vaxis': 'cols',
    #             'xticks': (np.arange(scanlen) + 0.5).tolist(),
    #             'xticklabels': list(range(scanstart, scanstop)),
    #             'xlabel': 'time shift [steps]',
    #             'yticks': False, 'ylabel': None,
    #             # 'colorbar': True, 'colorbar_orientation': 'vertical',
    #         })
    #     rows.append(cols)
        
    return rows

def get_subplots_expr0111_mi(scanlen, dim_s0, dim_m0, *args, **kwargs):
    cols = []
    for i in range(scanlen):
        cols.append({
            'input': 'd1',
            # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
            # 'xslice': (0, 1),
            'ndslice': (i, slice(None), slice(None)),
            'shape': (dim_s0, dim_m0),
            'title': 'mi-matrix',
            'cmap': 'Reds',
            'vaxis': 'rows',
            'plot': 'bla'
        })
    return cols

def get_subplots_expr0120(dim_s0, dim_m0, dim_s1, scanlen, scanstart, scanstop):
    rows = []
    # for row in range(rowiter):

    # for j in range(dim_s1):
    #     cols = []
    #     for i in range(dim_m0):
    #         cols.append({
    #             'input': ['m2s'], 'ndslice': (slice(scanlen), j, i),
    #             'shape': (dim_s0, scanlen), 'cmap': ['rainbow'],
    #             'title': 'Cross-correlation $m_0 \star s_1$',
    #             # 'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
    #             # 'xticks': False, 'xlabel': None,
    #             # 'yticks': False, 'ylabel': None,
    #             # 'xaxis': range(scanstart, scanstop),
    #             'ylim': (-1.2, 1.2),
    #             # 'colorbar': True, 'colorbar_orientation': 'vertical',
    #         })
    #     rows.append(cols)

    for j in range(dim_s1):
        cols = []
        for i in range(dim_m0):
            cols.append({
                # 'input': ['m2m', 'm2s', 's2s'],
                'input': ['m2s', 's2s'],
                'ndslice': [(slice(scanlen), j, i)] * 2,
                'shape': (dim_s0, scanlen),
                'cmap': ['glasbey_warm'],
                'title': 'Cross-correlation $m_0 \star s_1$',
                'title_pos': 'top_out',
                'ylim': (-1.2, 1.2),
                'legend': {'cross-correlation': 0, 'auto-correlation': 1},
                # 'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
                # 'xticks': False, 'xlabel': None,
                # 'yticks': False, 'ylabel': None,
                # 'xaxis': range(scanstart, scanstop),
                # 'colorbar': True, 'colorbar_orientation': 'vertical',
            })
        rows.append(cols)

    # for j in range(dim_s1):
    #     cols = []
    #     for i in range(dim_s1):
    #         cols.append({
    #             'input': ['s2s'], 'ndslice': (slice(scanlen), j, i),
    #             'shape': (dim_s1, scanlen), 'cmap': ['glasbey_warm'],
    #             'title': 'Auto-correlation $s_1 \star s_1$',
    #             'ylim': (-1.2, 1.2),
    #             # 'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',
    #             # 'xticks': False, 'xlabel': None,
    #             # 'yticks': False, 'ylabel': None,
    #             # 'xaxis': range(scanstart, scanstop),
    #             # 'colorbar': True, 'colorbar_orientation': 'vertical',
    #         })
    #     rows.append(cols)

    for j in range(dim_s1):
        cols = []
        # mutual information scan
        for i in range(dim_m0):
            cols.append({
                # 'input': ['d1', 'd2', 'd3'],
                'input': ['d2', 'd3'],
                # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
                # 'xslice': (0, 1),
                'ndslice': [(slice(scanlen), i, 0)] * 2,
                'shape': (dim_s1, scanlen), 'cmap': ['glasbey_warm'],
                'title': 'Mutual information $I_m(m_0; s_1)$',
                'title_pos': 'top_out',
                'ylim': (-1.2, 1.2),
                'xticklabels': list(range(scanstart, scanstop)),
                'legend': {'mutual information': 0, 'self information': 1}, # 'MI prop': 0, 
                # 'vmin': 0.0, 'vmax': 1.0,
                # 'vaxis': 'cols',
                # 'xticks': (np.arange(scanlen) + 0.5).tolist(),
                # 'xlabel': 'time shift [steps]',
                # 'xticks': False, 'xlabel': None,
                # 'yticks': False, 'ylabel': None,
                # 'colorbar': True, 'colorbar_orientation': 'vertical',
            })
        rows.append(cols)

    # for j in range(dim_s1):
    #     cols = []
    #     # mutual information scan
    #     for i in range(dim_s1):
    #         cols.append({
    #             'input': 'd2',
    #             # 'yslice': (i * dim_m0 * dim_s0, (i+1) * dim_m0 * dim_s0),
    #             # 'xslice': (0, 1),
    #             'ndslice': (slice(scanlen), i, 0),
    #             'shape': (dim_s1, scanlen), 'cmap': ['glasbey_warm'],
    #             'title': 'Self information $I_m(s_1; s_1)$',
    #             'ylim': (-1.2, 1.2),
    #             'xticks': (np.arange(0, scanlen, 5) + 0.5).tolist(),
    #             'xticklabels': list(range(scanstart, scanstop, 5)),
    #             'xlabel': 'time shift [steps]', 'ylog': True,
    #             # 'vmin': 0.0, 'vmax': 1.0,
    #             # 'vaxis': 'cols',
    #             # 'yticks': False, 'ylabel': None,
    #             # 'colorbar': True, 'colorbar_orientation': 'vertical',
    #         })
    #     rows.append(cols)
            
    return rows

# 'seismic'
#  +  +  +
