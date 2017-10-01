"""smp_graphs utils_conf

some utils for use in configuration files
"""

from functools import partial

import numpy as np

from smp_base.measures import meas
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2

"""system block
 - a robot
"""
def get_systemblock_pm(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1, lag = 1):
    global np, PointmassBlock2, meas
    print "get_systemblock_pm dim = %d" % (dim_s_proprio, )
    return {
        'block': PointmassBlock2,
        'params': {
            'id': 'robot1',
            'blocksize': 1, # FIXME: make pm blocksize aware!
            'systype': 2,
            'sysdim': dim_s_proprio,
            # initial state
            'x0': np.random.uniform(-0.3, 0.3, (dim_s_proprio * 3, 1)),
            'inputs': {'u': {'bus': 'pre_l0/pre'}},
            'outputs': {
                's_proprio': {'shape': (dim_s_proprio, 1)},
                's_extero':  {'shape': (dim_s_extero, 1)}
                }, # , 's_all': [(9, 1)]},
            'statedim': dim_s_proprio * 3,
            'dt': dt,
            'mass': 1.0,
            'force_max':  1.0,
            'force_min': -1.0,
            'friction': 0.01,
            'sysnoise': 1e-2,
            'debug': False,
            'dim_s_proprio': dim_s_proprio,
            'length_ratio': 3./2., # gain curve?
            'm_mins': [-1.0] * dim_s_proprio,
            'm_maxs': [ 1.0] * dim_s_proprio,
            'dim_s_extero': dim_s_extero,
            'lag': lag,
            'order': 2,
            'coupling_sigma': 1e-2,
            'transfer': 0,
            'anoise_mean': 0.0,
            'anoise_std': 1e-2,
            
            # model related
            # tapping
            'lag_past': (-4, -3),
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

def get_systemblock_sa(dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1):
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
            'lag': 3,
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

def get_systemblock_bha(dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1):
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
def get_systemblock_stdr(dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1):
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
def get_systemblock_lpzbarrel(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.01):
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
def get_systemblock_sphero(dim_s_proprio = 2, dim_s_extero = 1, dt = 0.05):
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

get_systemblock = {
    'pm': partial(get_systemblock_pm, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
    'sa': partial(get_systemblock_sa, dim_s_proprio = 2, dim_s_extero = 2, dt = 0.1),
    'bha': partial(get_systemblock_bha, dim_s_proprio = 9, dim_s_extero = 3, dt = 0.1),
    'lpzbarrel': partial(get_systemblock_lpzbarrel, dim_s_proprio = 2, dim_s_extero = 1, dt = 2.0/92.0), # 0.025),
    'stdr': partial(get_systemblock_stdr, dim_s_proprio = 2, dim_s_extero = 3, dt = 0.1),
    'sphero': partial(get_systemblock_sphero, dim_s_proprio = 2, dim_s_extero = 1, dt = 0.0167),
    }
