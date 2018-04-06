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
logger = get_module_logger(modulename = 'utils_conf', loglevel = logging.DEBUG)

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
    if kwargs.has_key('dim_s_proprio'):
        dim_s0 = kwargs['dim_s_proprio']
    if kwargs.has_key('dim_s_extero'):
        dim_s1 = kwargs['dim_s_extero']

    # defaults
    dims = {
        # if we're lucky we can get away with explicit limits because they are obtained from s2s
        'm0': {'dim': dim_s0, 'dist': 0}, # , 'mins': [-1] * dim_s0, 'maxs': [1] * dim_s0
        's0': {'dim': dim_s0, 'dist': 0},
    }
    if dim_s1 is not None:
        dims['s1'] = {'dim': dim_s1, 'dist': 0}

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
            'lag_past': (-4, -3),
            # 'lag_past': (-2, -1),
            'lag_future': (-1, 0),
            
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
    for k in [k for k in kwargs.keys() if k in ['numelem', 'h_numelem']]:
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
                
