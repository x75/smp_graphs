"""smp_graphs systems blocks

a system block is a wrapper for a system from smp_sys
"""

from smp_graphs.block import decStep, PrimBlock2
from smp_sys.systems import PointmassSys

class SysBlock2(PrimBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)


class PointmassBlock2(SysBlock2):
    """pointmass block, very thin wrapper around smp_sys.systems.PointmassSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        print "conf", conf
        self.system = PointmassSys(conf['params'])

    @decStep()
    def step(self, x = None):
        self.u = self.inputs['u'][0]
        self.x = self.system.step(self.u)
        for k in ['s_proprio', 's_extero']:
            setattr(self, k, self.x[k])
            print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
