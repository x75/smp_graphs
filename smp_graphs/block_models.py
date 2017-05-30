"""smp_graphs

Models: coding, representation, prediction, inference, association
"""

import numpy as np

from smp_graphs.block import decInit, decStep, PrimBlock2

class CodingBlock2(PrimBlock2):
    """CodingBlock2

    mean/var coding block, recursive estimate of input's mu and sigma
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        for ink, inv in self.inputs.items():
            print inv
            for outk in ["mu", "sig"]:
                setattr(self, "%s_%s" % (ink, outk), np.zeros(inv['shape']))
        
    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
                         (self.__class__.__name__,self.outputs.keys(), self.bus, self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        
        if self.cnt % self.blocksize == 0:
            for ink, inv in self.inputs.items():
                for outk_ in ["mu", "sig"]:
                    outk = "%s_%s" % (ink, outk_)
                    outv_ = getattr(self, outk)

                    if outk.endswith("mu"):
                        setattr(self, outk, 0.99 * outv_ + 0.01 * inv['val'])
                    elif outk.endswith("sig"):
                        setattr(self, outk, 0.99 * outv_ + 0.01 * np.sqrt(np.square(inv['val'] - getattr(self, ink + "_mu"))))
