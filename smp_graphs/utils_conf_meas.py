"""smp_graphs utils_conf_meas

Helper func for parameterized measurement stack for use in
configuration files
"""
from collections import OrderedDict

import numpy as np

# from smp_base.measures import meas_div
from smp_graphs.block import Block2, FuncBlock2
from smp_graphs.block_meas import MomentBlock2, MeasBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2
from smp_graphs.funcs import f_sum, f_envelope, f_rootmeansquare

from smp_base.common import get_module_logger
logger = get_module_logger(modulename = 'utils_conf_meas', loglevel = 'info')

kwargs = {
    'numsteps': 1000, 
    'p_vars': ['pre_l0/pre'], 
    'p_del_vars': ['pre_l0/pre'], 
    'm_vars': ['robot1/s0'], 
    'dim_s0': 1, 
    'numbins': 21, 
    # 'm_hist_bins': np.linspace(-1.1, 1.1, numbins + 1), 
    # 'm_hist_bincenters': m_hist_bins[:-1] + np.mean(np.abs(np.diff(m_hist_bins)))/2.0, 
    'div_meas': 'pyemd', 
}

def get_measures_block(*args, **kwargs):
    measblockid = kwargs['measblockid'] # 1000
    numsteps = kwargs['numsteps'] # 1000
    p_vars = kwargs['p_vars'] # ['pre_l0/pre']
    p_del_vars = kwargs['p_del_vars'] # ['pre_l0/pre']
    m_vars = kwargs['m_vars'] # ['robot1/s0']
    dim_s0 = kwargs['dim_s0'] # 1
    numbins = kwargs['numbins'] # 21
    div_meas = kwargs['div_meas'] # 'pyemd'

    m_hist_bins = np.linspace(-1.1, 1.1, numbins + 1)
    m_hist_bincenters = m_hist_bins[:-1] + np.mean(np.abs(np.diff(m_hist_bins)))/2.0
    
    # bla = OrderedDict([
    # bla = [
    # measures
    bla = ('meas%d' % (measblockid, ), {
            'block': Block2,
            'params': {
                'numsteps': 1, # numsteps,
                # 'id': 'measures%d' % (measblockid, ),
                'nocache': True,
                'graph': OrderedDict([
                    # m: mutual information I(m1;m2)
                    ('m_mi%d' % (measblockid, ), {
                        'block': MIBlock2,
                        'params': {
                            'blocksize': numsteps,
                            'shift': (0, 1),
                            'inputs': {
                                'x': {'bus': p_del_vars[0], 'shape': (dim_s0, numsteps)},
                                # 'y': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                                'y': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
                            },
                            'outputs': {
                                'mi': {'shape': (1, 1, 1)},
                            }
                        }
                    }),
                    
                    # m: information distance d(m1, m2) = 1 - (I(m1; m2)/H(m1,m2))
                    ('m_di%d' % (measblockid, ), {
                        'block': InfoDistBlock2,
                        'params': {
                            'blocksize': numsteps,
                            'shift': (0, 1),
                            'inputs': {
                                'x': {'bus': p_del_vars[0], 'shape': (dim_s0, numsteps)},
                                # 'y': {'bus': p_vars[0], 'shape': (dim_s0, numsteps)},
                                'y': {'bus': m_vars[0], 'shape': (dim_s0, numsteps)},
                            },
                            'outputs': {
                                'infodist': {'shape': (1, 1, 1)},
                            }
                        }
                    }),
    
                    # m: error
                    ('m_err%d' % (measblockid, ), {
                        'block': MeasBlock2,
                        'params': {
                            'blocksize': numsteps,
                            # 'debug': True,
                            'mode': 'basic',
                            'scope': 'local',
                            'meas': 'sub',
                            'inputs': {
                                'x1': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                                'x2': {'bus': m_vars[0], 'shape': (1, numsteps)},
                            },
                            'outputs': {
                                'y': {'shape': (1, numsteps)},
                            },
                        },
                    }),
        
                    # # m: error
                    # ('m_err_mdl1%d' % (measblockid, ), {
                    #     'block': MeasBlock2,
                    #     'params': {
                    #         'blocksize': numsteps,
                    #         # 'debug': True,
                    #         'mode': 'basic',
                    #         'scope': 'local',
                    #         'meas': 'sub',
                    #         'inputs': {
                    #             # 'x1': {'bus': p_vars[0], 'shape': (1, numsteps)},
                    #             # 'x1': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                    #             'x1': {'bus': 'mdl1/y', 'shape': (1, numsteps)},
                    #             # 'x1': {'bus': 'delay/dmdl_y',  'shape': (1, numsteps)},
                    #             'x2': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                    #         },
                    #         'outputs': {
                    #             'y': {'shape': (1, numsteps)},
                    #         },
                    #     },
                    # }),
        
                    # m: (root) mean squared error
                    ('m_err%d_a' % (measblockid, ), {
                        'block': FuncBlock2,
                        'params': {
                            # 'id': 'm_rmse',
                            'blocksize': numsteps,
                            'debug': False,
                            'func': f_envelope,
                            'inputs': {
                                'x': {'bus': 'm_err%d/y' % (measblockid, ), 'shape': (1, numsteps)},
                                'c': {'val': 0.01, 'shape': (1, 1)},
                            },
                            'outputs': {
                                'y': {'shape': (1, numsteps)},
                            },
                        },
                    }),
    
                    # m: (root) mean squared error
                    ('m_rmse%d' % (measblockid, ), {
                        'block': FuncBlock2,
                        'params': {
                            # 'id': 'm_rmse',
                            'blocksize': numsteps,
                            'debug': False,
                            'func': f_rootmeansquare,
                            'inputs': {
                                'x': {'bus': 'm_err%d/y' % (measblockid, ), 'shape': (1, numsteps)},
                            },
                            'outputs': {
                                'y': {'shape': (1, 1)},
                            },
                        },
                    }),
    
                    # m: histogram
                    ('m_hist%d' % (measblockid, ), {
                        'block': MeasBlock2,
                        'params': {
                            'id': 'm_hist',
                            'blocksize': numsteps,
                            'debug': False,
                            'mode': 'hist',
                            'scope': 'local',
                            'meas': 'hist',
                            # direct histo input?
                            # or signal input
                            'inputs': {
                                'x1': {'bus': p_del_vars[0], 'shape': (1, numsteps)},
                                'x2': {'bus': m_vars[0],     'shape': (1, numsteps)},
                            },
                            'bins': m_hist_bins,
                            'outputs': {
                                'x1_p': {'shape': (1, numbins)},
                                'x1_x': {'shape': (1, numbins + 1)},
                                'x2_p': {'shape': (1, numbins)},
                                'x2_x': {'shape': (1, numbins + 1)},
                            },
                        },
                    }),
    
                    # m: divergence histos
                    ('m_div%d' % (measblockid, ), {
                        'block': MeasBlock2,
                        'params': {
                            'id': 'm_div',
                            'blocksize': numsteps,
                            # 'debug': True,
                            'mode': 'div', # 'basic',
                            'scope': 'local',
                            'meas': div_meas, # ['chisq', 'kld'],
                            # direct histo input?
                            # or signal input
                            'inputs': {
                                'x1_p': {'bus': 'm_hist%d/x1_p' % (measblockid, ), 'shape': (1, numbins)},
                                'x1_x': {'bus': 'm_hist%d/x1_x' % (measblockid, ), 'shape': (1, numbins + 1)},
                                'x2_p': {'bus': 'm_hist%d/x2_p' % (measblockid, ), 'shape': (1, numbins)},
                                'x2_x': {'bus': 'm_hist%d/x2_x' % (measblockid, ), 'shape': (1, numbins + 1)},
                            },
                            'outputs': {
                                'y': {'shape': (1, numbins)},
                            },
                        },
                    }),
    
                    # m: sum divergence
                    ('m_div%d_sum' % (measblockid, ), {
                        'block': FuncBlock2,
                        'params': {
                            'blocksize': numsteps,
                            'debug': False,
                            'func': f_sum,
                            'inputs': {
                                'x': {'bus': 'm_div%d/y' % (measblockid, ), 'shape': (1, numbins)},
                            },
                            'outputs': {
                                'y': {'shape': (1, 1)},
                            },
                        },
                    }),
                ]),
            },
        })
    # ]
    # ])
    # logger.info('bla = %s' % (bla, ))
    return bla

if __name__ == '__main__':
    for i in range(2):
        kwargs['measblockid'] = i
        print "bla[%d]" % (i, ), get_measures_block(**kwargs)

