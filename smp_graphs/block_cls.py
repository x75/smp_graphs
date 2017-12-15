"""smp_graphs systems blocks

a system block is a wrapper for a system from smp_sys

TODO
 - replace proprio / extero by a general list of modalities
 - general formulation of both pm + sa with only individual step functions
"""

import numpy as np

from smp_graphs.block import decStep, Block2, PrimBlock2
from smp_sys.systems import PointmassSys, Pointmass2Sys
from smp_sys.systems import SimplearmSys
from smp_sys.systems_bhasim import BhaSimulatedSys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SysBlock2(PrimBlock2):
    """System block base class
    """
    defaults = {
        'block_group': 'data',
    }
    def __init__(self, conf = {}, paren = None, top = None):
        defaults = {}
        defaults.update(Block2.defaults)
        defaults.update(PrimBlock2.defaults, **self.defaults)
        self.defaults = defaults
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

class PointmassBlock2(SysBlock2):
    """Pointmass system block

    Very thin wrapper around :class:`smp_sys.systems.PointmassSys` and
    :class:`smp_sys.systems.Pointmass2Sys`.
    """
    def __init__(self, conf = {}, paren = None, top = None):
        # print "PointmassBlock2 conf = %s" %(conf,)
        # if hasattr(self, 'systype') and self.systype == 2:
        if 'systype' in conf['params'] and conf['params']['systype'] == 2:
            self.system = Pointmass2Sys(conf['params'])
        else:
            self.system = PointmassSys(conf['params'])

        conf['params']['dims'] = self.system.dims
        # conf['params']['outputs'].update(self.system.dims)
        # systems new state space
        # print "dims", self.system.x # .keys()
        for k,v in list(self.system.dims.items()):
            # print "adding output", k, self.system.x[k].shape
            # conf['params']['outputs'][k] = {'shape': self.system.x[k].shape}
            conf['params']['outputs'][k] = {'shape': (self.system.x[k].shape[0], conf['params']['blocksize'])}
            # setattr(self, k, self.x[k])
            
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # self.debug_print("init: conf = %s", (conf,))

        # dimensions
        for modality in ['proprio', 'extero']:
            if not hasattr(self, 'dim_s_%s' % modality):
                setattr(self, 'dim_s_%s' % modality, self.sysdim)
            
        # latent output variables defined by pointmass system
        self.x = {
            's_proprio': np.zeros((self.dim_s_proprio,  self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero,   self.blocksize)),
            's_all':     np.zeros((self.dim_s_proprio + self.dim_s_extero, self.blocksize)),
            # 's_proprio': np.zeros((self.sysdim,   self.blocksize)),
            # 's_extero':  np.zeros((self.sysdim,   self.blocksize)),
            # 's_all':     np.zeros((self.statedim, self.blocksize)),
        }

        # aberration / worlds hack
        self.mode = 1.0
        
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)

        # check if transfer function output 'h' is present and initialize
        if 'h' in self.outputs:
            setattr(self, 'h', np.random.uniform(0, 1, (self.dims['s0']['dim'], self.h_numelem)))
                    
    @decStep()
    def step(self, x = None):
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            
            # # aberration / worlds hack
            # if self.cnt % 10 == 0:
            #     self.mode *= -1.0

            # print "block_cls u", self.u
            
            self.x_ = self.system.step(self.u * self.mode)
            self._debug("step[%d] self.x_.keys() = %s, self.x_ = %s" % (self.cnt, list(self.x_.keys()), self.x_))

            # loop over outputs
            for k, v in list(self.outputs.items()):
                self._debug("    output %s, self.%s = %s" % (k, k, getattr(self, k)))
                # skip if not triggered
                if not self.output_is_triggered(k, v, self.bus): continue

                ######################################################################
                # system ground truth hack (gth): in general this is not known which
                # is why we are doing all of this itfp 
                if k == 'h': # if output 'h' is configured
                    # get h input samples, e.g. linspace, meshgrid
                    # FIXME: random sampling if dim > 4
                    self.h_sample = np.atleast_2d(np.hstack([np.linspace(self.m_mins[i], self.m_maxs[i], self.h_numelem) for i in range(self.dims['m0']['dim'])]))
                    # logger.debug('ref.h_sample = %s', ref.h_sample.shape)
                    # hack: loop over number of input samples
                    for h_sample_i in range(self.h_numelem): # v['shape'][1]):
                        # obtain output sample from eval of input sample on the system
                        x_ = self.system.step(self.h_sample.T[[h_sample_i],...])
                        # self._debug("    h_sample_i = %d, x_ = %s" % (h_sample_i, x_))
                        # copy to block attribute
                        self.h[...,[h_sample_i]] = x_['s0']
                ######################################################################
                    
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                k_ = getattr(self, k)
                # print "k", k, "self.k", k_.shape
                if k in self.x_:
                    #   print "i", i, "k", k, "self.x", self.x_[k].shape, "k_", k_.shape
                    k_[...,[i]] = self.x_[k]
                elif 'remap' in v:
                    k_[...,[i]] = self.x_[v['remap']]
                # else:
                #     setattr(self, k, self.x[k])
                
class SimplearmBlock2(SysBlock2):
    """Simple arm system block class

    Very thin wrapper around smp_sys.systems.SimplearmSys
    """
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # print "conf", conf['params'].keys()
        
        self.debug_print("init: conf = %s", (conf,))
        self.system = SimplearmSys(conf['params'])
        
        # dimensions
        for modality in ['proprio', 'extero']:
            if not hasattr(self, 'dim_s_%s' % modality):
                # ???
                setattr(self, 'dim_s_%s' % modality, self.sysdim)

        # latent output variables defined by simplearm system
        self.x = {
            's_proprio': np.zeros((self.sysdim,   self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero, self.blocksize)),
            's_all':     np.zeros((self.statedim, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        # print "%s.step %d" % (self.cname, self.cnt,)
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            # print "%s-%s.step[%d] self.u = %s" % (self.cname, self.id, self.cnt, self.u)
            self.x = self.system.step(self.u)
            # print "%s-%s.step[%d] self.x = %s" % (self.cname, self.id, self.cnt, self.x)
            # real output variables defined by config
            # for k in ['s_proprio', 's_extero', 's_all']:
            for k in list(self.outputs.keys()):
                k_ = getattr(self, k)
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                

class BhasimulatedBlock2(SysBlock2):
    """Bhasimulated system block, thin wrapper around smp_sys.systems_bhasim.BhasimulatedSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = BhaSimulatedSys(conf['params'])
        # latent output variables defined by simplearm system
        self.x = {
            's_proprio': np.zeros((self.sysdim,   self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero, self.blocksize)),
            's_all':     np.zeros((self.statedim, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)

        if self.doplot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection = "3d")

    @decStep()
    def step(self, x = None):
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            self.x = self.system.step(self.u)
            # print "self.u", self.u
            # real output variables defined by config
            # for k in ['s_proprio', 's_extero', 's_all']:
            for k in list(self.outputs.keys()):
                k_ = getattr(self, k)
                # print "bhasysblock k_", k_, k_[:,[i]].shape, self.x[k].shape
                # print "bhasysblock k_", k_, k_[:,[i]], self.x[k]
                k_[:,[i]] = self.x[k].T
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
            if self.doplot and self.cnt % 100 == 0:
                print("%s.step plotting arm with u = %s" % (self.cname, self.u.T))
                self.system.visualize(self.ax, self.u.T)
                plt.draw()
                plt.pause(1e-6)
