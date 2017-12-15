"""smp_graphs information theoretic (infth) measures

2017 Oswald Berthold

blocks for computing various uni-, bi-, and multivariate information theoretic
measures on timeseries like joint entropy, mutual information, transfer entropy,
and conditional transfer entropy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from smp_base.measures_infth import measMI, measH, compute_mutual_information, infth_mi_multivariate, compute_information_distance, compute_transfer_entropy, compute_conditional_transfer_entropy, compute_mi_multivariate, compute_transfer_entropy_multivariate, compute_entropy_multivariate
from smp_base.common import get_module_logger

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2
from smp_graphs.utils import myt

# block wrappers for smp_base/measures_infth.py, similar to the general smp_base/measures.py pattern

from logging import DEBUG as logging_DEBUG
# import logging
logger = get_module_logger(modulename = 'block_meas_infth', loglevel = logging_DEBUG)

################################################################################
# Block decorator init
class decInitInfthPrim():
    """!@brief PrimBlock2.init wrapper for inth blocks"""
    def __call__(self, f):
        @decInit()
        def wrap(xself, *args, **kwargs):
            # set defaults
            xself.norm_out = True
            xself.embeddingscan = None
            
            # call init
            f(xself, *args, **kwargs)
        return wrap

class decStepInfthPrim():
    """!@brief PrimBlock2.init wrapper for inth blocks"""
    def __call__(self, f):
        @decStep()
        def wrap(xself, *args, **kwargs):
            # call step
            meas = f(xself, *args, **kwargs)
            meas = np.array(meas)
            print("%s meas.shape = %s" % (xself.cname, meas.shape))
            for outk, outv in list(xself.outputs.items()):
                # self.jh = meas.reshape()
                setattr(xself, outk, meas.reshape(outv['shape']))
        return wrap
    
class InfthPrimBlock2(PrimBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def normalize(self, src, dst, cond = None):
        """InfthPrimBlock2.normalize

        compute joint entropy of two (n) groups of variables as the self-information
        of the concatenation of both (all n) groups (FIXME: n)
        """
        # reasonable default
        jhinv = 1.0
        if self.norm_out:
            # normalize from external input, overrides stepwise norm_out
            if 'norm' in self.inputs:
                jhinv = 1.0 / self.get_input('norm').T
                
            # normalize over input block
            else:
                # stack src and destination
                randvars = tuple(rv for rv in [src, dst, cond] if rv is not None) # src, dst)
                st = np.hstack(randvars)
                # compute full joint entropy as normalization constant
                jhinv = 1.0 / compute_mi_multivariate(data = {'X': st, 'Y': st})
                
        return jhinv

class JHBlock2(InfthPrimBlock2):
    """Compute the global scalar joint entropy of multivariate data
    """
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
    
    @decStepInfthPrim()
    def step(self, x = None):
        meas = []
        src = self.inputs['x']['val'].T
        dst = self.inputs['y']['val'].T
        
        print("%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape), end=' ')
        
        for i in range(self.shift[0], self.shift[1]):
            print("%d" % (i, ), end=' ')
            sys.stdout.flush()
            
            # src_ = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # src_ = np.roll(src, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            # st = np.hstack((src_, dst))
            # jh = compute_mi_multivariate(data = {'X': st, 'Y': st})

            # use jidt's delay param
            jh = compute_entropy_multivariate((src, dst), delay = -i)
            # jh = compute_mi_multivariate(data = {'X': st, 'Y': st}, delay = -i)
            # print "%s.step[%d] data = %s, jh = %f" % (self.cname, self.cnt, st.shape, jh)
            meas.append(jh)
        print("")
        return meas
        
class MIBlock2(InfthPrimBlock2):
    """Compute the elementwise elementwise mutual information among all variables in the data
    """
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        meas = []
        
        dst = self.get_input('x').T
        src = self.get_input('y').T

        # compute norm factor
        jh = self.normalize(src, dst)
        
        self._debug("%s.step[%d]-%s src.shape = %s, dst.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape,))
        
        for i in range(self.shift[0], self.shift[1]):
            self._debug("%d" % (i, ),)
            sys.stdout.flush()

            # # self-rolled time shift with np.roll
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            # mi = self.compute_mutual_information(src, dst)
            
            mi = compute_mutual_information(src, dst, delay = -i, norm_in = True)
            # print "mi", mi
            
            meas.append(mi.copy())

        print("")
        meas = np.array(meas)
        self.mi = meas.T.copy() * jh
        self._debug("%s-%s.mi = %s\n    mi/jh = %s/%s" % (self.cname, self.id, self.mi.shape, self.mi, jh)) # , mi.shape

class InfoDistBlock2(InfthPrimBlock2):
    """Compute the information distance between to variables

    Compute elementwise information distance among all variables in
    dataset. This is obtained via the MI by interpreting the MI as
    proximity and inverting it. It's normalized by the joint entropy.

    FIXME: default conf
    """
    defaults = {
        'shift': [0, 1],
    }
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        mis = []
        # src = self.inputs['x']['val'].T
        # dst = self.inputs['y']['val'].T
        src = self.get_input('x').T
        dst = self.get_input('y').T
        
        self._debug("%s%s-%s.step[%d] src.shape = %s, dst.shape = %s" % (
            self.nesting_indent, self.cname, self.id, self.cnt, src.shape, dst.shape,),)

        # compute norm factor
        jh = self.normalize(src, dst)
        
        for i in range(self.shift[0], self.shift[1]):
            self._debug("%d" % (i, ),)
            sys.stdout.flush()

            # # self-rolled time shift
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            # mi = compute_information_distance(src, dst)

            mi = compute_information_distance(src, dst, delay = -i, normalize = jh)
            mis.append(mi)
            
            # if src eq dst 
            # blank out the diagonal since it's always one
            # np.fill_diagonal(mi, np.max(mi))
            
        # print ""            
        mis = np.array(mis)
        # print "mis", mis
        # print "%s-%s.step infodist.shape = %s / %s" % (self.cname, self.id, mis.shape, mis.T.shape)
        # print "%s.%s infodist = %s" % (self.cname, self.id, mi)

        # why transpose?
        self.infodist = myt(mis, direction = -1).copy()
        # self.infodist = np.clip(myt(mis, direction = -1), -1, 2) # implicit copy?
        self.infodist_pos = np.clip(myt(mis, direction = -1), 1, np.inf)
        self.infodist_neg = np.clip(myt(mis, direction = -1), -np.inf, 0)
        # self.infodist = mis.copy()
        self._debug("infodist block = %s/\n    id/jh = %s/%s" % (self.infodist.shape, self.infodist, jh)) # , mi.shape
        
class TEBlock2(InfthPrimBlock2):
    """!@brief Compute elementwise transfer entropy from src to dst variables in dataset"""
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        tes = []
        src = self.inputs['x']['val'].T
        dst = self.inputs['y']['val'].T
        
        print("%s.step[%d]-%s src.shape = %s, dst.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape,), end=' ')

        # norm
        jh = self.normalize(src, dst)
        
        for i in range(self.shift[0], self.shift[1]):
            self._debug("%d" % (i, ),)
            sys.stdout.flush()
            
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            
            # st = np.hstack((src, dst))
            
            # te = compute_transfer_entropy(dst, src)
            te = compute_transfer_entropy(dst, src, delay = -i)
            tes.append(te.copy())
        # print ""

        tes = np.array(tes)
        # print "%s-%s.step tes.shape = %s / %s" % (self.cname, self.id, tes.shape, tes.T.shape)
        # self.te[:,0] = tes.flatten()
        self.te = tes.T.copy() * jh

class CTEBlock2(InfthPrimBlock2):
    """!@brief Compute elementwise conditional transfer entropy from src to dst variables conditioned
    on cond variables in dataset"""
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        # set fedaults
        self.xcond = False
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        ctes = []
        dst  = self.inputs['x']['val'].T
        src  = self.inputs['y']['val'].T
        cond = self.inputs['cond']['val'].T
        
        print("%s.step[%d]-%s src.shape = %s, dst.shape = %s, cond.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape, cond.shape), end=' ')

        # norm
        jh = self.normalize(src, dst, cond)

        for i in range(self.shift[0], self.shift[1]):
            print("%d" % (i, ), end=' ')
            sys.stdout.flush()
            
            # src  = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst  = self.inputs['y']['val'].T
            
            # st = np.hstack((src, dst))
            
            # cte = compute_conditional_transfer_entropy(dst, src, cond)
            cte = compute_conditional_transfer_entropy(src, dst, cond, delay = -i, xcond = self.xcond)
            ctes.append(cte.copy())
        print("")

        ctes = np.array(ctes)
        # print "%s-%s.step ctes.shape = %s / %s" % (self.cname, self.id, ctes.shape, ctes.T.shape)
        # print "ctes", ctes
        # self.cte[:,0] = ctes.flatten()
        self.cte = ctes.T.copy() * jh

################################################################################
# multivariate versions
class MIMVBlock2(InfthPrimBlock2):
    """Compute the multivariate mutual information between X and Y, aka the total MI"""
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        self._debug("step[%d] self.inputs['x']['val'].T.shape = %s, shifting from %s to %s" % (
            self.cnt, self.inputs['x']['val'].T.shape, self.shift[0], self.shift[1]),)
        # for k, v in self.inputs.items():
        #     # print "k", k, "v", v
        #     if k == "norm":
        #         print "bus value", v['val'], self.bus[v['bus']]
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        mimvs = []

        # get inputs
        src = self.get_input('y').T
        dst = self.get_input('x').T

        # normalize
        jh = self.normalize(src, dst)
        
        for i in range(self.shift[0], self.shift[1]):
            # print "%d" % (i, ),
            # sys.stdout.flush()
            
            # print "self.inputs['x']['val'].T.shape", self.inputs['x']['val'].T.shape
            # dst_ = np.roll(dst, shift = i, axis = 0)
            
            # st = np.hstack((src, dst))
            # jh = self.measH.step(st)
            # mi = self.meas.step(st, st)

            if self.embeddingscan == "src":
                self.inputs['y']['embedding'] = (i - self.shift[0]) + 1
                src_ = self.get_input('y').T
                mi = compute_mi_multivariate(data = {'X': src_, 'Y': dst}, estimator = "kraskov2", normalize = True)
            else: # delayscan
                mi = compute_mi_multivariate(data = {'X': src, 'Y': dst}, estimator = "kraskov2", normalize = True, delay = -i)
            # print "mimv = %s" % mi
            mimvs.append(mi)
        # print ""
        mimvs = np.array(mimvs)
        # print "@%d mimvs.shape = %s" % (self.cnt, mimvs.shape, )
        # print "@%d mimvs       = %s" % (self.cnt, mimvs, )
                    
        # mi = self.meas.step(self.inputs['x']['val'].T, self.inputs['y']['val'].T)
        # np.fill_diagonal(mi, np.min(mi))
        # print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        # self.mi[:,0] = (mi/jh).flatten()
        # self.mimv[0,shiftsl] = mimvs.flatten() # /maxjh
        self.mimv[0,shiftsl] = mimvs.flatten() * jh # /maxjh
        self._debug("step[%d] self.mimv.shape = %s, mimv = %s, jh = %f" % (self.cnt, self.mimv.shape, self.mimv, 1.0/jh))

class TEMVBlock2(InfthPrimBlock2):
    """!@brief Compute the multivariate transfer entropy from X to Y, aka the total TE"""
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        temvs = []
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        # jh = self.measH.step(st)
        src = self.inputs['y']['val'].T
        dst = self.inputs['x']['val'].T
        
        print("%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by" % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape), end=' ')

        # norm
        jh = self.normalize(src, dst)
        
        for i in range(self.shift[0], self.shift[1]):
            print("%d" % (i, ), end=' ')
            sys.stdout.flush()
            
            # dst_ = np.roll(dst, shift = i, axis = 0)
            
            # st = np.hstack((src, dst))
            
            # mi = compute_transfer_entropy_multivariate(src, dst, delay = -self.shift[0]+i)
            mi = compute_transfer_entropy_multivariate(src, dst, delay = -i)
            temvs.append(mi)
        print("")
        temvs = np.array(temvs)
        # self.temv[0,shiftsl] = temvs.flatten()
        self.temv[0,shiftsl] = temvs.flatten() * jh

class CTEMVBlock2(InfthPrimBlock2):
    """!@brief Compute the multivariate conditional transfer entropy from X to Y, conditioned on C, aka the total CTE (doesn't exist yet)"""
    @decInitInfthPrim()
    def __init__(self, conf = {}, paren = None, top = None):
        InfthPrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
