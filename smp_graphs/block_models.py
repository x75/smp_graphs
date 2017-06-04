"""smp_graphs

Models: coding, representation, prediction, inference, association
"""

import numpy as np

from reservoirs import Reservoir

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

# def init_identity(ref):
#     return None
                        
# def step_identity(ref, ins = {}):
#     return None

def init_musig(ref, conf, mconf):
    params = conf['params']
    ref.a1 = mconf['a1']
    for ink, inv in params['inputs'].items():
        print inv
        for outk in ["mu", "sig"]:
            outk_full = "%s_%s" % (ink, outk)
            params['outputs'][outk_full] = {'shape': inv['shape']}
            # setattr(self, "%s_%s" % (ink, outk), np.zeros(inv['shape']))
    # return None

def step_musig(ref):
    for ink, inv in ref.inputs.items():
        for outk_ in ["mu", "sig"]:
            outk = "%s_%s" % (ink, outk_)
            outv_ = getattr(ref, outk)

            if outk.endswith("mu"):
                setattr(ref, outk, ref.a1 * outv_ + (1 - ref.a1) * inv['val'])
            elif outk.endswith("sig"):
                setattr(ref, outk, ref.a1 * outv_ + (1 - ref.a1) * np.sqrt(np.square(inv['val'] - getattr(ref, ink + "_mu"))))


def init_res(ref, conf, mconf):
    params = conf['params']
    ref.res = Reservoir(
        N = 60,
        input_num = params['inputs']['x']['shape'][0],
        output_num = 1 
    )
    params['outputs']['x_res'] = {'shape': (60, 1)}

def step_res(ref):
    ref.res.execute(ref.inputs['x']['val'])
    print ref.res.r.shape
    setattr(ref, 'x_res', ref.res.r)
                        
class model(object):
    models = {
        # 'identity': {'init': init_identity, 'step': step_identity},
        'musig': {'init': init_musig, 'step': step_musig},
        'res': {'init': init_res, 'step': step_res},
    }
    # 
    def __init__(self, ref, conf, mconf = {}):
        self.modelstr = mconf['type']
        self.models[self.modelstr]['init'](ref, conf, mconf)
        
    def predict(self, ref):
        self.models[self.modelstr]['step'](ref)

class ModelBlock2(PrimBlock2):
    """Basic Model block

    Merge funcblock with memory"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        """ModelBlock2 init"""
        params = conf['params']
        
        for k, v in params['models'].items():
            v['inst_'] = model(ref = self, conf = conf, mconf = v)
            params['models'][k] = v

        # print "\n params.models = %s" % (params['models'], )
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # print "\n self.models = %s" % (self.models, )
        
    @decStep()
    def step(self, x = None):
        """ModelBlock2 step"""
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
            (self.__class__.__name__, self.outputs.keys(), self.bus,
                 self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        
        if self.cnt % self.blocksize == 0:
            for mk, mv in self.models.items():
                mv['inst_'].predict(self)
