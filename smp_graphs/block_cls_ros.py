"""smp_graphs.block_cls_ros

Systems blocks using ROS
"""

import numpy as np

try:
    import rospy
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except Exception as e:
    print("Import rospy failed with %s" % (e, ))

from smp_graphs.block import decStep, PrimBlock2
from smp_graphs.block_cls import SysBlock2
from smp_sys.systems_ros import STDRCircularSys, LPZBarrelSys, SpheroSys

################################################################################
# STDR 2D robot simulator a la stage
class STDRCircularBlock2(SysBlock2):
    """STDRCircularBlock2

    System block, very thin wrapper around smp_sys.systems.STDRCircularSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = STDRCircularSys(conf['params'])
        # latent output variables defined by pointmass system
        self.x = {
            's_proprio': np.zeros((self.dim_s_proprio,  self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero,   self.blocksize)),
            's_all':     np.zeros((self.dim_s_proprio + self.dim_s_extero, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        """STDRCircular.step

        step STDR robot
        """
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            # print "u", self.u
            self.x = self.system.step(self.u)
            # print "x", self.x
            
            # real output variables defined by config
            for k in list(self.outputs.keys()):
                k_ = getattr(self, k)
                # print "k", k, "k_", k_, "k_[:,[i]]", k_[:,[i]], "self.x[k]", self.x[k]
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)

################################################################################
# connect to an lpzrobots roscontroller
class LPZBarrelBlock2(SysBlock2):
    """LPZBarrelBlock2

    System block, very thin wrapper around smp_sys.systems.LPZBarrelSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = LPZBarrelSys(conf['params'])
        # latent output variables defined by pointmass system
        self.x = {
            's_proprio': np.zeros((self.dim_s_proprio,  self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero,   self.blocksize)),
            's_all':     np.zeros((self.dim_s_proprio + self.dim_s_extero, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        """LPZBarrel.step

        step LPZBarrel robot
        """
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            # print "u", self.u
            self.x = self.system.step(self.u)
            # print "x", self.x
            
            # real output variables defined by config
            for k in list(self.outputs.keys()):
                k_ = getattr(self, k)
                # print "k", k, "k_", k_, "k_[:,[i]]", k_[:,[i]], "self.x[k]", self.x[k]
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                
################################################################################
# Sphero
class SpheroBlock2(SysBlock2):
    """SpheroBlock2

    System block, very thin wrapper around smp_sys.systems.SpheroSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = SpheroSys(conf['params'])
        # latent output variables defined by pointmass system
        self.x = {
            's_proprio': np.zeros((self.dim_s_proprio,  self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero,   self.blocksize)),
            's_all':     np.zeros((self.dim_s_proprio + self.dim_s_extero, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        """Sphero.step

        step Sphero robot
        """
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            # print "u", self.u
            self.x = self.system.step(self.u)
            # print "x", self.x
            
            # real output variables defined by config
            for k in list(self.outputs.keys()):
                k_ = getattr(self, k)
                # print "k", k, "k_", k_, "k_[:,[i]]", k_[:,[i]], "self.x[k]", self.x[k]
                k_[:,[i]] = self.x[k].T
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                
# puppy
# nao
# turtlebot
# quad
# atrv
# hucar
