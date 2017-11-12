import numpy as np

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2

# essentia
HAVE_ESSENTIA = False
try:
    import essentia as e
    import essentia.standard as estd
except ImportError, e:
    print "Failed to import essentia and essentia.standard", e

class EssentiaBlock2(PrimBlock2):
    """EssentiaBlock2 class

    Compute :mod:`essentia` features on input data
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
    

    @decStep()
    def step(self, x = None):
        # print "MomentBlock2"
        for k, v in self.inputs.items():
            print "EssentiaBlock2 step[%d] input %s = %s" % (self.cnt, k, v.keys())
