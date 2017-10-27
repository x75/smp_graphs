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
# BHA / mathias
# stdr
# sphero
# 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SysBlock2(PrimBlock2):
    """!@brief Basic system block"""
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
    """!@brief Pointmass system block, very thin wrapper around smp_sys.systems.PointmassSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        if hasattr(self, 'systype') and self.systype == 2:
            self.system = Pointmass2Sys(conf['params'])
        else:
            self.system = PointmassSys(conf['params'])

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
        

    @decStep()
    def step(self, x = None):
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            
            # # aberration / worlds hack
            # if self.cnt % 10 == 0:
            #     self.mode *= -1.0
            
            self.x = self.system.step(self.u * self.mode)
            # print "PointmassBlock2-%s.step[%d] self.u = %s" % (self.id, self.cnt, self.u)
            # real output variables defined by config
            # for k in ['s_proprio', 's_extero', 's_all']:
            for k in self.outputs.keys():
                # print "k", k, getattr(self, k)
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                k_ = getattr(self, k)
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                
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
            for k in self.outputs.keys():
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
            for k in self.outputs.keys():
                k_ = getattr(self, k)
                # print "bhasysblock k_", k_, k_[:,[i]].shape, self.x[k].shape
                # print "bhasysblock k_", k_, k_[:,[i]], self.x[k]
                k_[:,[i]] = self.x[k].T
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
            if self.doplot and self.cnt % 100 == 0:
                print "%s.step plotting arm with u = %s" % (self.cname, self.u.T)
                self.system.visualize(self.ax, self.u.T)
                plt.draw()
                plt.pause(1e-6)
