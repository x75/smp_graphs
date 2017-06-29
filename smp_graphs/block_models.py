"""smp_graphs

Oswald Berthold 2012-2017

smp models: models are coders, representations, predictions, inferences, associations, ...



models raw (fit/predict X/Y style models)
models sm  sensorimotor  models
models dvl devleopmental models

The model design is in progress. The current approach is to have a
general Block wrapper for all models with particular models being
implemented by lightewight init() and step() function definitions
"""

import numpy as np

from mdp.nodes import PolynomialExpansionNode

# reservoir lib from smp_base
from reservoirs import Reservoir, res_input_matrix_random_sparse, res_input_matrix_disjunct_proj

from smp_graphs.graph import nxgraph_node_by_id_recursive
from smp_graphs.block import decInit, decStep, PrimBlock2
from smp_base.models_actinf import ActInfKNN, ActInfGMM, ActInfHebbianSOM

try:
    from smp_base.models_actinf import ActInfSOESGP, ActInfSTORKGP
    HAVE_SOESGP = True
except ImportError, e:
    print "couldn't import online GP models", e
    HAVE_SOESGP = False

class CodingBlock2(PrimBlock2):
    """CodingBlock2

    mean-variance-residual coding block, recursive estimate of input's mu and sigma
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        for ink, inv in self.inputs.items():
            print inv
            for outk in ["mu", "sig", "std"]:
                if outk.endswith("sig"):
                    setattr(self, "%s_%s" % (ink, outk), np.ones(inv['shape']))
                else:
                    setattr(self, "%s_%s" % (ink, outk), np.zeros(inv['shape']))
        
    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
                         (self.__class__.__name__,self.outputs.keys(), self.bus, self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        
        if self.cnt % self.blocksize == 0:
            for ink, inv in self.inputs.items():
                for outk_ in ["mu", "sig", "std"]:
                    outk = "%s_%s" % (ink, outk_)
                    outv_ = getattr(self, outk)

                    if outk.endswith("mu"):
                        setattr(self, outk, 0.99 * outv_ + 0.01 * inv['val'])
                    elif outk.endswith("sig"):
                        setattr(self, outk, 0.99 * outv_ + 0.01 * np.sqrt(np.square(inv['val'] - getattr(self, ink + "_mu"))))
                    elif outk.endswith("std"):
                        mu = getattr(self, 'x_mu')
                        sig = getattr(self, 'x_sig')
                        setattr(self, outk, (inv['val'] - mu) / sig)

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

# model func: reservoir expansion
def init_res(ref, conf, mconf):
    params = conf['params']
    ref.oversampling = mconf['oversampling']
    ref.res = Reservoir(
        N = mconf['N'],
        input_num = mconf['input_num'],
        output_num = mconf['output_num'],
        input_scale = mconf['input_scale'], # 0.03,
        bias_scale = mconf['bias_scale'], # 0.0,
        feedback_scale = 0.0,
        g = 0.99,
        tau = 0.05,
    )
    ref.res.wi = res_input_matrix_random_sparse(mconf['input_num'], mconf['N'], density = 0.2) * mconf['input_scale']
    params['outputs']['x_res'] = {'shape': (mconf['N'], 1)}

def step_res(ref):
    # print ref.inputs['x']['val'].shape
    for i in range(ref.oversampling):
        ref.res.execute(ref.inputs['x']['val'])
    # print ref.res.r.shape
    setattr(ref, 'x_res', ref.res.r)

# model func: polynomial expansion via mdp
def init_polyexp(ref, conf, mconf):
    params = conf['params']
    ref.polyexpnode = PolynomialExpansionNode(3)
    # params['outputs']['polyexp'] = {'shape': params['inputs']['x']['shape']}
    params['outputs']['polyexp'] = {'shape': (83, 1)}

def step_polyexp(ref):
    setattr(ref, 'polyexp', ref.polyexpnode.execute(ref.inputs['x']['val'].T).T)

# model func: random_uniform model
def init_random_uniform(ref, conf, mconf):
    params = conf['params']
    for outk, outv in params['outputs'].items():
        lo = -np.ones(( outv['shape'] ))
        hi = np.ones(( outv['shape'] ))
        setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))

def step_random_uniform(ref):
    if hasattr(ref, 'rate'):
        if (ref.cnt % ref.rate) not in ref.blockphase: return
            
    lo = ref.inputs['lo']['val'] # .T
    hi = ref.inputs['hi']['val'] # .T
    for outk, outv in ref.outputs.items():
        setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        # print "%s-%s[%d]model.step_random_uniform %s = %s" % (
        #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))

# active inference stuff
def init_model(ref, conf, mconf):
    """initialize sensorimotor forward model"""
    algo = mconf['algo']
    idim = mconf['idim']
    odim = mconf['odim']
    
    if not HAVE_SOESGP:
        algo = "knn"
        print "soesgp/storkgp not available, defaulting to knn"
            
    if algo == "knn":
        # mdl = KNeighborsRegressor(n_neighbors=5)
        mdl = ActInfKNN(idim, odim)
    elif algo == "soesgp":
        mdl = ActInfSOESGP(idim, odim)
    elif algo == "storkgp":
        mdl = ActInfSTORKGP(idim, odim)
    elif algo == 'copy':
        targetid = mconf['copyid']

        # # debugging
        # print "topgraph", ref.top.nxgraph.nodes()
        # print "topgraph.node[0]['block_'].nxgraph", ref.top.nxgraph.node[0]['block_'].nxgraph.nodes()
        # for n in ref.top.nxgraph.node[0]['block_'].nxgraph.nodes():
        #     print "    node.id = %s, graph = %s" % (
        #         ref.top.nxgraph.node[0]['block_'].nxgraph.node[n]['block_'].id,
        #         ref.top.nxgraph.node[0]['block_'].nxgraph.node[n]['block_'].nxgraph.nodes(), )
            
        targetnode = nxgraph_node_by_id_recursive(ref.top.nxgraph, targetid)
        print "targetid", targetid, "targetnode", targetnode
        if len(targetnode) > 0:
            # print "    targetnode id = %d, node = %s" % (
            #     targetnode[0][0],
            #     targetnode[0][1].node[targetnode[0][0]])
            # copy node
            clone = {}
            tnode = targetnode[0][1].node[targetnode[0][0]]
        mdl = tnode['block_'].mdl
    else:
        print "unknown model algorithm %s, exiting" % (algo, )
        # import sys
        # sys.exit(1)
        mdl = None

    assert mdl is not None, "Model (algo = %s) shouldn't be None, check your config" % (algo,)
        
    return mdl
        
# model func: actinf_m1
def init_actinf_m1(ref, conf, mconf):
    # params = conf['params']
    # hi = 1
    # for outk, outv in params['outputs'].items():
    #     setattr(ref, outk, np.random.uniform(-hi, hi, size = outv['shape']))
    ref.mdl = init_model(ref, conf, mconf)
    ref.X_  = np.zeros((mconf['idim'], 1))
    ref.y_  = np.zeros((mconf['odim'], 1))
    ref.pre_l1_tm1 = 0
    # # eta = 0.3
    # eta = ref.eta
    # lag = ref.lag
    # # print "Lag = %d" % (lag,)

def tapping(ref):
    # maxlag == windowsize

    # individual lags
    
    # current goal[t] prediction descending from layer above
    pre_l1   = ref.inputs['pre_l1']['val']
    # measurement[t] at current layer input
    meas_l0 = ref.inputs['meas_l0']['val']
    # prediction[t-1] at current layer input
    pre_l0   = ref.inputs['pre_l0']['val']   
    # prediction error[t-1] at current layer input
    prerr_l0 = ref.inputs['prerr_l0']['val']

    # get lag spec: None (lag = 1), int d (lag = d), array a (lag = a)
    pre_l1_tap_spec = ref.inputs['pre_l1']['lag']
    pre_l1_tap = ref.inputs['pre_l1']['val'][...,]

    # compute prediction error with respect to top level prediction (goal)
    prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-lag]]
    # compute the target for the  forward model from the prediction error
    ref.y_ = pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
    X__ = np.vstack((pre_l1[...,[-lag]], prerr_l0[...,[-(lag-1)]]))

    
    return (pre_l1, pre_l0, meas_l0, prerr_l0)
    
def step_actinf_m1(ref):
    # get lag
    # lag = ref.inputs['']['val'][...,lag]
    # lag = 0

    # deal with the lag specification for each input (lag, delay, temporal characteristic)
    (pre_l1, pre_l0, meas_l0, prerr_l0) = tapping(ref)
    
    # print "pre_l1.shape", pre_l1.shape, "pre_l0.shape", pre_l0.shape, "meas_l0.shape", meas_l0.shape

    # print "ref.pre.shape", ref.pre.shape, "ref.err.shape", ref.err.shape
    
    assert pre_l1.shape[-1] == pre_l0.shape[-1] == meas_l0.shape[-1], "step_actinf_m1: input shapes need to agree"

    # # loop over block of inputs if pre_l1.shape[-1] > 0:
    # for i in range(pre_l1.shape[-1]):
    #     (pre, prerr, y_) = step_actinf_m1_single(ref, pre_l1[...,[i]], pre_l0[...,[i]], meas_l0[...,[i]])
    #     ref.debug_print(
    #         "step_actinf_m1 id = %s, pre = %s, prerr = %s, tgt = %s",
    #         (ref.id, pre, prerr, y_))
            
    #     pre_ = getattr(ref, 'pre')
    #     pre_[...,[i]] = pre
    #     err_ = getattr(ref, 'err')
    #     err_[...,[i]] = prerr
    #     tgt_ = getattr(ref, 'tgt')
    #     tgt_[...,[i]] = y_

    
    # loop over block of inputs if pre_l1.shape[-1] > 0:
    (prerr, y_) = step_actinf_m1_fit(ref, pre_l1, pre_l0, meas_l0, prerr_l0)
    # (pre, )     = step_actinf_m1_predict(ref, pre_l1, pre_l0, meas_l0, prerr_l0)
    (pre, )     = step_actinf_m1_predict(ref, pre_l1, pre_l0, meas_l0, prerr)
     
    # ref.debug_print(
    # print "step_actinf_m1 id = %s, pre = %s, prerr = %s, tgt = %s" % (ref.id, pre, prerr, y_)
            
    pre_ = getattr(ref, 'pre')
    pre_[...,[-1]] = pre
    err_ = getattr(ref, 'err')
    err_[...,[-1]] = prerr
    tgt_ = getattr(ref, 'tgt')
    tgt_[...,[-1]] = y_
    # prerr_ = getattr(ref, 'prerr')
    # prerr_[...,[i]] = y_

    # print "pre_", pre_
    # print "err_", err_
    # print "tgt_", tgt_
            
    # publish model's internal state
    # setattr(ref, 'pre', pre_l0.T)
    # setattr(ref, 'err', prerr_l0)
    # setattr(ref, 'tgt', ref.y_)
    setattr(ref, 'pre', pre_)
    setattr(ref, 'err', err_)
    setattr(ref, 'tgt', tgt_)

def step_actinf_m1_fit(ref, pre_l1, pre_l0, meas_l0, prerr_l0):
    lag = ref.lag + 1 # because of negative indices
    # print "Lag = %d" % (lag,)
    # debug
    ref.debug_print(
        "step_actinf_m1_single ref.X_ = %s, pre_l1 = %s, meas_l0 = %s, pre_l0 = %s",
        (ref.X_.shape, pre_l1.shape, meas_l0.shape, pre_l0.shape))

    prerr_l0_ = prerr_l0[...,[-1]] # np.zeros_like(pre_l1[...,[-1]])
    
    if not np.any(np.isinf(meas_l0)):
        # pre_l1_ = pre_l1[...,[-lag]]
        # pre_l0_ = pre_l0[...,[-lag]]
        # prerr_l0_ = prerr_l0[...,[-lag]]
        # print "blub", pre_l1[...,[-1]], ref.pre_l1_tm1
        # print "goal dist", np.sum(np.abs(pre_l1[...,[-1]] - ref.pre_l1_tm1))
        # prediction error at current layer input if goal hasn't changed
        # if np.sum(np.abs(pre_l1[...,[-1]] - ref.pre_l1_tm1)) < 1e-2:
        #     # print "goal hasn't changed"
        #     # prerr_l0 = pre_l0 - pre_l1
        #     prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-lag]]
        #     # prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-1]]
        #     # prerr_l0_ = np.zeros_like(pre_l1[...,[-1]])
        # else:
        #     print "#"  * 80
        #     print "goal changed"
        #     # prerr_l0_ = np.random.uniform(-1e-3, 1e-3, pre_l1[...,[-1]].shape)
        #     prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-lag]]
        #     # prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-lag]]
            
        prerr_l0_ = meas_l0[...,[-1]] - pre_l1[...,[-lag]]

        # print "prerr_l0", prerr_l0
        # prerr statistics / expansion
        # self.prediction_errors_extended()

        # compute the target for the  forward model from the prediction error
        # if i % 10 == 0: # play with decreased update rates
        ref.y_ = pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
        # FIXME: suppress update when error is small enough (< threshold)

        # print "%s.step_actinf_m1[%d] ref.X_ = %s, ref.y_ = %s" % (ref.__class__.__name__, ref.cnt, ref.X_.shape, ref.y_.shape)
    
        # fit the model
        # prerr_l0_ = meas_l0_ - pre_l1_
        X__ = np.vstack((pre_l1[...,[-lag]], prerr_l0[...,[-(lag-1)]]))
        # print "X__.shape", X__.shape, "y_.shape", ref.y_, ref.y_.shape
        ref.mdl.fit(X__.T, ref.y_.T) # ref.X_[-lag]
    else:
        # FIXME: for model testing
        prerr_l0_ = pre_l0[...,[-1]].copy()

    # remember last descending prediction
    ref.pre_l1_tm1 = pre_l1[...,[-1]].copy()
    return (prerr_l0_.copy(), ref.y_.copy())
    
# def step_actinf_m1_single(ref, pre_l1, pre_l0, meas_l0):
def step_actinf_m1_predict(ref, pre_l1, pre_l0, meas_l0, prerr_l0):
    # FIXME: this doesn't work, need to return proper tuple
    if hasattr(ref, 'rate'):
        if (ref.cnt % ref.rate) not in ref.blockphase: return # return pre, prerr, y_

    # pre_l1 = 
    # pre_l1_ = pre_l1[...,[-lag]]
    # pre_l0_ = pre_l0[...,[-lag]]
    # prerr_l0_ = prerr_l0[...,[-lag]]
    # # prepare new model input
    # if np.sum(np.abs(pre_l1 - ref.pre_l1_tm1)) > 1e-6:
    #     # goal changed
    #     prerr_l0 = pre_l0 - pre_l1
            
    # m1: model input X is goal and prediction error
    # ref.X_ = np.hstack((pre_l1.T, prerr_l0.T))
    ref.X_ = np.vstack((pre_l1[...,[-1]], prerr_l0[...,[-1]]))
    ref.debug_print("step_actinf_m1_single ref.X_.shape = %s", (ref.X_.shape, ))

    # predict next values at current layer input
    pre_l0_ = ref.mdl.predict(ref.X_.T)

    # print "pre_l0", pre_l0
    
    ref.debug_print("step_actinf_m1_single ref.X_ = %s, pre_l0 = %s", (ref.X_.shape, pre_l0_.shape))
    # for outk, outv in ref.outputs.items():
    #     setattr(ref, outk, pre_l0)
    #     print "step_actinf_m1 %s.%s = %s" % (ref.id, outk, getattr(ref, outk))

    # return (pre_l0.T, prerr_l0, ref.y_)
    return (pre_l0_.T.copy(), )

################################################################################
# selforg / playful: hs, hk, pimax/tipi?
def init_homoekinesis(ref, conf, mconf):
    # params = conf['params']
    # hi = 1
    # for outk, outv in params['outputs'].items():
    #     setattr(ref, outk, np.random.uniform(-hi, hi, size = outv['shape']))
    ref.mdl = init_model(ref, conf, mconf)
    ref.X_  = np.zeros((mconf['idim'], 1))
    ref.y_  = np.zeros((mconf['odim'], 1))
    ref.pre_l1_tm1 = 0
    # # eta = 0.3
    # eta = ref.eta
    # lag = ref.lag
    # # print "Lag = %d" % (lag,)

def step_homeokinesis(ref):
    # get lag
    # lag = ref.inputs['']['val'][...,lag]
    # lag = 0
    # current goal[t] prediction descending from layer above
    pre_l1   = ref.inputs['pre_l1']['val']
    # measurement[t] at current layer input
    meas_l0 = ref.inputs['meas_l0']['val']
    # prediction[t-1] at current layer input
    pre_l0   = ref.inputs['pre_l0']['val']   
    # prediction error[t-1] at current layer input
    prerr_l0 = ref.inputs['prerr_l0']['val']

    return {
        's_proprio': pre_l0.copy(),
        's_extero': pre_l0.copy()}
    
class model(object):
    """model

    generic model class used by ModelBlock2

    low-level models are implemented via init_<model> and step_<model> functions
    reducing code?
    """
    models = {
        # open-loop models
        # 'identity': {'init': init_identity, 'step': step_identity},
        # expansions
        'musig': {'init': init_musig, 'step': step_musig},
        'res': {'init': init_res, 'step': step_res},
        'polyexp': {'init': init_polyexp, 'step': step_polyexp},
        # active randomness
        'random_uniform': {'init': init_random_uniform, 'step': step_random_uniform},
        # active inference
        'actinf_m1': {'init': init_actinf_m1, 'step': step_actinf_m1},
        # selforg playful
        'homeokinesis': {'init': init_homoekinesis, 'step': step_homeokinesis},
    }
    # 
    def __init__(self, ref, conf, mconf = {}):
        assert mconf['type'] in self.models.keys(), "in %s.init: unknown model type, %s not in %s" % (self.__class__.__name__, mconf['type'], self.models.keys())
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

        self.top = top
        # self.lag = 1

        # initialize model
        # FIXME: need to associate outputs with a model for arrays of models
        for k, v in params['models'].items():
            v['inst_'] = model(ref = self, conf = conf, mconf = v)
            params['models'][k] = v

        # print "\n params.models = %s" % (params['models'], )
        # print "top", top.id
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # print "\n self.models = %s" % (self.models, )
        
    @decStep()
    def step(self, x = None):
        """ModelBlock2 step"""
        # print "%s-%s.step %d" % (self.cname, self.id, self.cnt,)
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
            (self.__class__.__name__, self.outputs.keys(), self.bus,
                 self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        
        if self.cnt % self.blocksize == 0:
            for mk, mv in self.models.items():
                mv['inst_'].predict(self)
