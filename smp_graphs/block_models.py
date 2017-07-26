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

from functools import partial

import numpy as np

from mdp.nodes import PolynomialExpansionNode

# import sklearn
from sklearn import linear_model, kernel_ridge
            

# reservoir lib from smp_base
from smp_base.reservoirs import Reservoir, res_input_matrix_random_sparse, res_input_matrix_disjunct_proj

from smp_graphs.graph import nxgraph_node_by_id_recursive
from smp_graphs.block import decInit, decStep, PrimBlock2
from smp_base.models_actinf  import smpKNN, smpGMM, smpIGMM, smpHebbianSOM
from smp_base.models_selforg import HK
from smp_base.learners import smpSHL

try:
    from smp_base.models_actinf import smpOTLModel, smpSOESGP, smpSTORKGP
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
    params['outputs']['polyexp'] = {'shape': (83, 1)} # ???

def step_polyexp(ref):
    setattr(ref, 'polyexp', ref.polyexpnode.execute(ref.inputs['x']['val'].T).T)

# model func: random_uniform model
def init_random_uniform(ref, conf, mconf):
    params = conf['params']
    for outk, outv in params['outputs'].items():
        lo = -np.ones(( outv['shape'] ))
        hi = np.ones(( outv['shape'] ))
        setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        # setattr(ref, outk, np.ones(outv['shape']))
        print "block_models.py: random_uniform_init %s = %s" % (outk, getattr(ref, outk))

def step_random_uniform(ref):
    if hasattr(ref, 'rate'):
        if (ref.cnt % ref.rate) not in ref.blockphase: return
            
    lo = ref.inputs['lo']['val'] # .T
    hi = ref.inputs['hi']['val'] # .T
    for outk, outv in ref.outputs.items():
        setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        
        # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
        # np.random.uniform(lo, hi, size = outv['shape']))
        # print "%s-%s[%d]model.step_random_uniform %s = %s" % (
        #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
        # print "block_models.py: random_uniform_step %s = %s" % (outk, getattr(ref, outk))

# model func: alternating_sign model
def init_alternating_sign(ref, conf, mconf):
    params = conf['params']
    for outk, outv in params['outputs'].items():
        lo = -np.ones(( outv['shape'] ))
        hi = np.ones(( outv['shape'] ))
        # setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        setattr(ref, outk, np.ones(outv['shape']))
        print "block_models.py: alternating_sign_init %s = %s" % (outk, getattr(ref, outk))

def step_alternating_sign(ref):
    if hasattr(ref, 'rate'):
        if (ref.cnt % ref.rate) not in ref.blockphase: return
            
    lo = ref.inputs['lo']['val'] # .T
    hi = ref.inputs['hi']['val'] # .T
    for outk, outv in ref.outputs.items():
        # setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        
        # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
        if np.sum(np.abs(getattr(ref, outk))) == 0.0:
            setattr(ref, outk, np.ones(outv['shape']))
        setattr(ref, outk, getattr(ref, outk) * -1.0)
        
        # np.random.uniform(lo, hi, size = outv['shape']))
        # print "%s-%s[%d]model.step_alternating_sign %s = %s" % (
        #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
        print "block_models.py: alternating_sign_step %s = %s" % (outk, getattr(ref, outk))
        
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
        # mdl = smpKNN(idim, odim)
        mdl = smpKNN(conf = mconf)
    elif algo == "gmm":
        mdl = smpGMM(conf = mconf)
    elif algo == "igmm":
        mdl = smpIGMM(conf = mconf)
    elif algo == "hebbsom":
        mconf.update({'numepisodes': 1, 'mapsize_e': 140, 'mapsize_p': 60, 'som_lr': 1e-1, 'visualize': False})
        print "mconf", mconf
        mdl = smpHebbianSOM(conf = mconf)
        # mdl = smpHebbianSOM(idim, odim, numepisodes = 1, mapsize_e = 1000, mapsize_p = 100, som_lr = 1e-1)
    elif algo == "soesgp":
        mdl = smpSOESGP(conf = mconf)
    elif algo == "storkgp":
        mdl = smpSTORKGP(conf = mconf)
    elif algo == 'resrls':
        mconf.update(smpSHL.defaults)
        # mconf.update({'numepisodes': 1, 'mapsize_e': 140, 'mapsize_p': 60, 'som_lr': 1e-1, 'visualize': False})
        mconf.update({'idim': idim, 'odim': odim})
        mdl = smpSHL(conf = mconf)
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
    elif algo == 'homeokinesis':
        mdl = HK(conf = mconf)
    else:
        print "unknown model algorithm %s, exiting" % (algo, )
        # import sys
        # sys.exit(1)
        mdl = None

    assert mdl is not None, "Model (algo = %s) shouldn't be None, check your config" % (algo,)
        
    return mdl

# tapping, uh ah
def tapping_SM(ref, mode = 'm1'):
    # FIXME: rewrite in general form (ref, invariable) -> (tapped invariable)
    # tapping: tap data
    # tapping: build training set
    
    # maxlag == windowsize
    # individual lags as array

    ############################################################
    # instantaneous inputs
    # current goal[t] prediction descending from layer above
    if ref.inputs.has_key('blk_mode') and ref.inputs['blk_mode']['val'][0,0] == 2.0:
        # that's a wild HACK
        ref.pre_l1_inkey = 'e2p_l1'
    else:
        ref.pre_l1_inkey = 'pre_l1'
        
    pre_l1   = ref.inputs[ref.pre_l1_inkey]['val']
    # measurement[t] at current layer input
    meas_l0 = ref.inputs['meas_l0']['val']
    # prediction[t-1] at current layer input
    pre_l0   = ref.inputs['pre_l0']['val']   
    # prediction error[t-1] at current layer input
    prerr_l0 = ref.inputs['prerr_l0']['val']

    ############################################################
    # tapped inputs
    # get lag spec: None (lag = 1), int d (lag = d), array a (lag = a)
    pre_l1_tap_spec = ref.inputs[ref.pre_l1_inkey]['lag']
    # print "pre_l1_tap_spec", pre_l1_tap_spec
    pre_l1_tap_full = ref.inputs[ref.pre_l1_inkey]['val'][...,pre_l1_tap_spec]
    # print "pre_l1_tap_full", pre_l1_tap_full
    pre_l1_tap_flat = pre_l1_tap_full.reshape((ref.odim, 1))

    meas_l0_tap_spec = ref.inputs['meas_l0']['lag']
    meas_l0_tap_full = ref.inputs['meas_l0']['val'][...,meas_l0_tap_spec]
    meas_l0_tap_flat = meas_l0_tap_full.reshape((ref.odim, 1))

    pre_l0_tap_spec = ref.inputs['pre_l0']['lag']
    pre_l0_tap_full = ref.inputs['pre_l0']['val'][...,pre_l0_tap_spec]
    pre_l0_tap_flat = pre_l0_tap_full.reshape((ref.odim, 1))

    prerr_l0_tap_spec = ref.inputs['prerr_l0']['lag']
    # print "prerr_l0_tap_spec", prerr_l0_tap_spec
    prerr_l0_tap_full = ref.inputs['prerr_l0']['val'][...,prerr_l0_tap_spec]
    prerr_l0_tap_flat = prerr_l0_tap_full.reshape((ref.odim, 1))
    
    # print "meas", meas_l0[...,[-1]], "prel1", pre_l1[...,[pre_l1_tap_spec[-1]]]
    
    # compute prediction error PE with respect to top level prediction (goal)
    # momentary PE
    prerr_l0_  = meas_l0[...,[-1]] - pre_l1[...,[pre_l1_tap_spec[-1]]]
    # embedding PE
    prerr_l0__ = meas_l0_tap_flat - pre_l1_tap_flat # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
    
    # FIXME: future > 1, shift target block across the now line completely and predict entire future segment
    # X__ = np.vstack((pre_l1[...,[-lag]], prerr_l0[...,[-(lag-1)]]))
    
    # return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, X, Y, prerr_l0__)
    return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, prerr_l0__)

def tapping_XY(ref, pre_l1_tap_flat, pre_l0_tap_flat, prerr_l0_tap_flat, prerr_l0__, mode = 'm1'):
    # print "tapping pre_l1", pre_l1_tap_flat.shape, prerr_l0_tap_flat.shape, ref.idim
    # print "tapping reshape", pre_l1_tap.reshape((ref.idim/2, 1)), prerr_l0_tap.reshape((ref.idim/2, 1))
    if ref.type == 'm1' or ref.type == 'm3':
        X = np.vstack((pre_l1_tap_flat, prerr_l0_tap_flat))
        # compute the target for the  forward model from the embedding PE
        Y = (pre_l0_tap_flat - (prerr_l0__ * ref.eta)) # .reshape((ref.odim, 1)) # pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
    elif ref.type == 'm2':
        X = np.vstack((prerr_l0_tap_flat, ))
        Y = -prerr_l0__ * ref.eta # .reshape((ref.odim, 1)) # pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
    else:
        return None
    # print "X", X.shape
    
    # ref.mdl.fit(X__.T, ref.y_.T) # ref.X_[-lag]
    
    return (X, Y)

def tapping_X(ref, pre_l1_tap_flat, prerr_l0__):
    if ref.type == 'm1' or ref.type == 'm3':
        X = np.vstack((pre_l1_tap_flat, prerr_l0__))
    elif ref.type == 'm2':
        X = np.vstack((prerr_l0__, ))

    return X

# model func: actinf_m2
def init_actinf(ref, conf, mconf):
    # params = conf['params']
    # hi = 1
    # for outk, outv in params['outputs'].items():
    #     setattr(ref, outk, np.random.uniform(-hi, hi, size = outv['shape']))
    ref.pre_l1_inkey = 'pre_l1'
    ref.mdl = init_model(ref, conf, mconf)
    ref.X_  = np.zeros((mconf['idim'], 1))
    ref.y_  = np.zeros((mconf['odim'], 1))
    ref.laglen  = mconf['laglen']
    ref.pre_l1_tm1 = np.zeros((mconf['idim']/2/ref.laglen, 1))
    ref.pre_l1_tm2 = np.zeros((mconf['idim']/2/ref.laglen, 1))

    if mconf['type'] == 'actinf_m1':
        ref.type = 'm1'
    elif mconf['type'] == 'actinf_m2':
        ref.type = 'm2'
    elif mconf['type'] == 'actinf_m3':
        ref.type = 'm3'

    if mconf['type'].startswith('actinf'):
        ref.tapping_SM = partial(tapping_SM, mode = ref.type)
        ref.tapping_XY = partial(tapping_XY, mode = ref.type)
        ref.tapping_X = partial(tapping_X)
        
    # goal statistics
    ref.dgoal_fit_ = np.linalg.norm(ref.pre_l1_tm1 - ref.pre_l1_tm2)
    ref.dgoal_ = np.linalg.norm(-ref.pre_l1_tm1)
    
def step_actinf(ref):

    # deal with the lag specification for each input (lag, delay, temporal characteristic)
    (pre_l1, pre_l0, meas_l0, prerr_l0, prerr_l0_, prerr_l0__) = ref.tapping_SM(ref) # tapping_SM(ref, mode = ref.type)
    (X, Y) = ref.tapping_XY(ref, pre_l1, pre_l0, prerr_l0, prerr_l0__)
    
    # print "pre_l1.shape", pre_l1.shape, "pre_l0.shape", pre_l0.shape, "meas_l0.shape", meas_l0.shape, "prerr_l0.shape", prerr_l0.shape, "prerr_l0_", prerr_l0_.shape, "X", X.shape, "Y", Y.shape

    # print "ref.pre.shape", ref.pre.shape, "ref.err.shape", ref.err.shape
    
    assert pre_l1.shape[-1] == pre_l0.shape[-1] == meas_l0.shape[-1], "step_actinf_m2: input shapes need to agree"

    # loop over block of inputs if pre_l1.shape[-1] > 0:
    prerr = prerr_l0_

    # dgoal for fitting lag additional time steps back
    dgoal_fit = np.linalg.norm(ref.pre_l1_tm1 - ref.pre_l1_tm2)
    y_ = Y.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
    if dgoal_fit < 5e-1: #  and np.linalg.norm(prerr_l0_) > 5e-2:
    # if np.linalg.norm(dgoal_fit) <= np.linalg.norm(ref.dgoal_fit_): #  and np.linalg.norm(prerr_l0_) > 5e-2:
        # prerr = prerr_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
        # FIXME: actually, if ref.mdl.hasmemory
        if isinstance(ref.mdl, smpOTLModel):
            # print "Fitting without update"
            ref.mdl.fit(X.T, Y.T, update = False)
        else:
            ref.mdl.fit(X.T, Y.T)
    else:
        # print "not fit[%d], dgoal_fit = %s, dgoal_fit_ = %s" % (ref.cnt, dgoal_fit, ref.dgoal_fit_)
        pass

    ref.dgoal_fit_ = 0.9 * ref.dgoal_fit_ + 0.1 * dgoal_fit
            
    # ref.X_ = np.vstack((pre_l1[...,[-1]], prerr_l0[...,[-1]]))
    ref.debug_print("step_actinf_m2_single ref.X_.shape = %s", (ref.X_.shape, ))

    # predict next values at current layer input
    # pre_l1_tap_spec = ref.inputs[ref.pre_l1_inkey]['lag']
    pre_l1_tap_full = ref.inputs[ref.pre_l1_inkey]['val'][...,-ref.laglen:]
    pre_l1_tap_flat = pre_l1_tap_full.reshape((ref.odim, 1))
    
    dgoal = np.linalg.norm(ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]] - ref.pre_l1_tm1)
    if dgoal > 5e-1: # fixed threshold
    # if np.linalg.norm(dgoal) > np.linalg.norm(ref.dgoal_): #  and np.linalg.norm(prerr_l0_) > 5e-2:
        # goal changed
        m = ref.inputs['meas_l0']['val'][...,[-1]].reshape((ref.odim / ref.laglen, 1))
        p = ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]].reshape((ref.odim / ref.laglen, 1))
        # prerr_l0_ = (m - p) # * 0.1
        # prerr_l0_ = -p.copy()
        prerr_l0_ = np.random.uniform(-1e-3, 1e-3, prerr_l0_.shape)
        # print "goal changed predict[%d], |dgoal| = %f, |PE| = %f" % (ref.cnt, dgoal, np.linalg.norm(prerr_l0_))

        # prerr_l0__ = meas_l0 - pre_l1_tap_flat
        tmp = prerr_l0__.reshape((ref.odim / ref.laglen, -1))
        tmp[...,[-1]] = prerr_l0_.copy()
        prerr_l0__ = tmp.reshape((ref.odim, 1)) # meas_l0_tap_flat - pre_l1_tap_flat # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
        # pre_l1[...,[-1]]).reshape((ref.odim, 1))
        
        prerr = prerr_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
        
    ref.dgoal_ = 0.9 * ref.dgoal_ + 0.1 * dgoal

    # print "prerr_l0__", prerr_l0__.shape

    ref.X_ = tapping_X(ref, pre_l1_tap_flat, prerr_l0__)

    pre_l0_ = ref.mdl.predict(ref.X_.T)
    # print "cnt = %s, pre_l0_" % (ref.cnt,), pre_l0_, "prerr_l0_", prerr_l0_.shape
    pre = pre_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
    # prerr = prerr_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
                
    pre_ = getattr(ref, 'pre')
    if ref.type == 'm1' or ref.type == 'm3':
        pre_[...,[-1]] = pre
    elif ref.type == 'm2':
        pre_[...,[-1]] = np.clip(pre_[...,[-1]] + pre, -1, 1)
    err_ = getattr(ref, 'err')
    err_[...,[-1]] = prerr
    tgt_ = getattr(ref, 'tgt')
    tgt_[...,[-1]] = y_

    # publish model's internal state
    setattr(ref, 'pre', pre_)
    setattr(ref, 'err', err_)
    setattr(ref, 'tgt', tgt_)

    # remember stuff
    ref.pre_l1_tm2 = ref.pre_l1_tm1.copy()
    ref.pre_l1_tm1 = ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]].copy() # pre_l1[...,[-1]].copy()

# def step_actinf_prediction_errors_extended(ref):
#     # if np.sum(np.abs(ref.goal_prop - ref.goal_prop_tm1)) > 1e-2:
#     #     ref.E_prop_pred_fast = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred_fast.shape)
#     #     ref.E_prop_pred_slow = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred_slow.shape)
#     #     # recompute error
#     #     # ref.E_prop_pred = ref.M_prop_pred - ref.goal_prop
#     #     # ref.E_prop_pred[:] = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred.shape)
#     #     #else:            
                
#     E_prop_pred_tm1 = ref.E_prop_pred.copy()

#     # prediction error's
#     ref.E_prop_pred_state = ref.S_prop_pred - ref.M_prop_pred
#     ref.E_prop_pred_goal  = ref.M_prop_pred - ref.goal_prop
#     ref.E_prop_pred = ref.E_prop_pred_goal
        
#     ref.E_prop_pred__fast = ref.E_prop_pred_fast.copy()
#     ref.E_prop_pred_fast  = ref.coef_smooth_fast * ref.E_prop_pred_fast + (1 - ref.coef_smooth_fast) * ref.E_prop_pred

#     ref.E_prop_pred__slow = ref.E_prop_pred_slow.copy()
#     ref.E_prop_pred_slow  = ref.coef_smooth_slow * ref.E_prop_pred_slow + (1 - ref.coef_smooth_slow) * ref.E_prop_pred
                
#     ref.dE_prop_pred_fast = ref.E_prop_pred_fast - ref.E_prop_pred__fast
#     ref.d_E_prop_pred_ = ref.coef_smooth_slow * ref.d_E_prop_pred_ + (1 - ref.coef_smooth_slow) * ref.dE_prop_pred_fast

# def step_actinf_sample_error_gradient(ref):
#     # sample error gradient
#     numsamples = 20
#     # was @ 50
#     lm = linear_model.Ridge(alpha = 0.0)
            
#     S_ = []
#     M_ = []
#     for i in range(numsamples):
#         # S_.append(np.random.normal(self.S_prop_pred, 0.01 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
#         # larger sampling range
#         S_.append(np.random.normal(self.S_prop_pred, 0.3 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
#         # print "S_[-1]", S_[-1]
#         M_.append(self.environment.compute_motor_command(S_[-1]))
#         S_ext_ = self.environment.compute_sensori_effect(M_[-1]).reshape((1, self.dim_ext))
#     S_ = np.array(S_).reshape((numsamples, self.S_prop_pred.shape[1]))
#     M_ = np.array(M_).reshape((numsamples, self.S_prop_pred.shape[1]))
#     print "S_", S_.shape, "M_", M_.shape
#     # print "S_", S_, "M_", M_

#     lm.fit(S_, M_)
#     self.grad = np.diag(lm.coef_)
#     print "grad", np.sign(self.grad), self.grad
            
#     # pl.plot(S_, M_, "ko", alpha=0.4)
#     # pl.show()
    
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

    m_mins   = ref.mdl.m_mins # 0.1
    m_mins_2 = m_mins * 2 # 0.2
    one_over_m_mins = 1.0/m_mins
    m_ranges = ref.mdl.m_maxs - ref.mdl.m_mins
    
    # predict next values at current layer input
    # pre_l0_ = ref.mdl.predict(ref.X_.T)
    # pre_l0_ = ref.mdl.step((meas_l0 - m_mins_2) * one_over_m_mins) # bha m_mins/m_maxs
    pre_l0_ = ref.mdl.step(meas_l0)
    err_ = ref.mdl.xsi 
    tgt_ = ref.mdl.v
   
    # print "meas_l0", meas_l0.shape
    # print "pre_l0_", pre_l0_.shape

    setattr(ref, 'pre', ((pre_l0_[:,[-1]] + 1) * 0.5) * m_ranges + m_mins) # m_mins + m_mins_2) # bha m_mins/m_maxs
    # setattr(ref, 'pre', pre_l0_[:,[-1]])#  - pre_l1[:,[-1]])
    setattr(ref, 'err', err_)
    setattr(ref, 'tgt', tgt_)
    # return (pre_l0_.T.copy(), )
    # return {
    #     's_proprio': pre_l0.copy(),
    #     's_extero': pre_l0.copy()}

def init_e2p(ref, conf, mconf):
    ref.mdl = init_model(ref, conf, mconf)
    ref.X_  = np.zeros((mconf['idim'], 1))
    ref.y_  = np.zeros((mconf['odim'], 1))
    ref.pre = np.zeros_like(ref.y_)
    ref.pre_l1_tm1 = 0

def step_e2p(ref):
    # current goal[t] prediction descending from layer above
    proprio   = ref.inputs['meas_l0_proprio']['val'][...,[-1]]
    # measurement[t] at current layer input
    extero    = ref.inputs['meas_l0_extero']['val'][...,[-1]]

    # print "proprio", proprio.shape
    # print "extero", extero.shape
    
    ref.mdl.fit(extero.T, proprio.T)

    # if ref.inputs['blk_mode']['val'] == 2.0:
    # if True:
    if ref.inputs.has_key('blk_mode') and ref.inputs['blk_mode']['val'][0,0] == 2.0:
        if ref.cnt % 100 == 0:
            # uniform prior
            extero_ = np.random.uniform(-1e-1, 1e-1, extero.shape)
            # model prior?
            extero_ = ref.mdl.sample_prior()
            # print "extero_", extero_.shape
            sample = np.clip(ref.mdl.predict(extero_.T), -1, 1)
            # print "sample", sample.shape
            setattr(ref, 'pre', sample.T)
            setattr(ref, 'pre_ext', extero_)
    
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
        # constants
        'alternating_sign': {'init': init_alternating_sign, 'step': step_alternating_sign},        
        # active randomness
        'random_uniform': {'init': init_random_uniform, 'step': step_random_uniform},
        # closed-loop models
        # active inference
        'actinf_m1': {'init': init_actinf, 'step': step_actinf},
        'actinf_m2': {'init': init_actinf, 'step': step_actinf},
        'actinf_m3': {'init': init_actinf, 'step': step_actinf},
        'e2p':       {'init': init_e2p,    'step': step_e2p},
        # selforg playful
        'homeokinesis': {'init': init_homoekinesis, 'step': step_homeokinesis},
    }
    # 
    def __init__(self, ref, conf, mconf = {}):
        assert mconf['type'] in self.models.keys(), "in %s.init: unknown model type, %s not in %s" % (self.__class__.__name__, mconf['type'], self.models.keys())
        # FIXME: ignoring multiple entries taking 'last' one, in dictionary order
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

        for k in ['idim', 'odim']:
            if v.has_key(k):
                setattr(self, k, v[k])
            
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

        # if rospy.is_shutdown():
        #     sys.exit()
