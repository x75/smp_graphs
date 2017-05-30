"""smp_graphs/measures/information theoretic (infth)
"""

import sys
import numpy as np

from smp_base.measures_infth import measMI, measH, compute_mutual_information, infth_mi_multivariate, compute_information_distance, compute_transfer_entropy, compute_conditional_transfer_entropy, compute_mi_multivariate, compute_transfer_entropy_multivariate

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2
from smp_graphs.utils import myt

# block wrappers for smp_base/measures_infth.py, similar to the general smp_base/measures.py pattern

class JHBlock2(PrimBlock2):
    """!@brief Compute scalar joint entropy of multivariate data"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
    
    @decStep()
    def step(self, x = None):
        jhs = []
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        
        src = self.inputs['x']['val'].T
        dst = self.inputs['y']['val'].T
        st = np.hstack((src, dst))
        jh0 = compute_mi_multivariate(data = {'X': st, 'Y': st}, delay = 0)
        print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src_ = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # src_ = np.roll(src, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            # st = np.hstack((src_, dst))
            # jh = compute_mi_multivariate(data = {'X': st, 'Y': st})

            # use jidt's delay param
            jh = compute_mi_multivariate(data = {'X': st, 'Y': st}, delay = -i)
            # print "%s.step[%d] data = %s, jh = %f" % (self.cname, self.cnt, st.shape, jh)
            jhs.append(jh)
        print ""
        jhs = np.array(jhs) # /jh0
        print "jhs.shape", jhs.shape
        # self.jh[0,shiftsl] = jhs
        self.jh[0,shiftsl] = jhs
        
class MIBlock2(PrimBlock2):
    """!@brief Compute elementwise mutual information among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        # jhs = []
        
        dst = self.inputs['x']['val'].T
        src = self.inputs['y']['val'].T
        st = np.hstack((src, dst))

        # shouldn't change too much under time shift
        jh = compute_mi_multivariate(data = {'X': st, 'Y': st})
        
        # print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape),
        # print "%s.step[%d]-%s src.sh = %s, src = %s" % (self.cname, self.cnt, self.id, src.shape, src)
        # print "%s.step[%d]-%s dst.sh = %s, dst = %s" % (self.cname, self.cnt, self.id, dst.shape, dst)
        print "%s.step[%d]-%s src.shape = %s, dst.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape,),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            # st = np.hstack((src, dst))
            # jh = self.compute_mutual_information(st, st)
            # mi = self.compute_mutual_information(src, dst)
            
            # jh = self.measH.step(st)
            # mi = self.meas.step(src, dst)

            mi = compute_mutual_information(src, dst, delay = -i)
            # print "mi", mi
            
            mis.append(mi.copy())
            # jhs.append(jh)

        print ""
        mis = np.array(mis)
        # print "%s-%s.step mis.shape = %s / %s" % (self.cname, self.id, mis.shape, mis.T.shape)
        # print "mis = ", mis.flatten()
        # jhs = np.array(jhs)
        # print "mis.shape = %s, jhs.shape = %s" % (mis.shape, jhs.shape)
                    
        # mi = self.meas.step(self.inputs['x']['val'].T, self.inputs['y']['val'].T)
        # np.fill_diagonal(mi, np.min(mi))
        # print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        # maxjh = np.max(jhs)
        # print "mutual info self.mi.shape = %s, mi.shape = %s, maxjh = %s" % (self.mi.shape, mi.shape, maxjh)
        
        # self.mi[:,0] = (mi/jh).flatten()
        # self.mi[:,0] = mis.flatten()/maxjh
        # normalized by joint entropy
        # self.mi[:,0] = mis.flatten()/jh
        # self.mi[:,0] = mis.flatten()
        self.mi = mis.T.copy()

class InfoDistBlock2(PrimBlock2):
    """!@brief Compute elementwise information distance among all variables in dataset. This is
    obtained via the MI by interpreting the MI as proximity and inverting it. It's normalized by
    the joint entropy."""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # self.meas = measMI()
        # self.measH = measH()

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        mis = []
        src = self.inputs['x']['val'].T
        dst = self.inputs['y']['val'].T
        # print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s" % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape)
        print "%s.step[%d]-%s src.shape = %s, dst.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape,),
        st = np.hstack((src, dst))
        jh = compute_mi_multivariate(data = {'X': st, 'Y': st})
        
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            
            # st = np.hstack((src, dst))
            # jh = self.measH.step(st)
            # mi = self.meas.step(st, st)
            # mi = compute_information_distance(st, st)

            mi = compute_information_distance(src, dst, delay = -i, normalize = jh)
            mis.append(mi)
            # mi = self.meas.step(self.inputs['x']['val'].T, self.inputs['y']['val'].T)
        
            # if src eq dst 
            # blank out the diagonal since it's always one
            # np.fill_diagonal(mi, np.max(mi))
        print ""            
        mis = np.array(mis)
        print "%s-%s.step infodist.shape = %s / %s" % (self.cname, self.id, mis.shape, mis.T.shape)
        # print "%s.%s infodist = %s" % (self.cname, self.id, mi)
        
        # self.infodist[:,0] = mi.flatten()
        # self.infodist[:,0] = mis.flatten()
        self.infodist = myt(mis/jh, direction = -1).copy()
        print "infodist block", self.infodist.shape, mi.shape
        
class TEBlock2(PrimBlock2):
    """!@brief Compute elementwise transfer entropy from src to dst variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        tes = []
        src = self.inputs['x']['val'].T
        dst = self.inputs['y']['val'].T
        # jh = self.measH.step(st)
        # print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape),
        # print "%s.step[%d]-%s src.sh = %s, src = %s" % (self.cname, self.cnt, self.id, src.shape, src)
        # print "%s.step[%d]-%s dst.sh = %s, dst = %s" % (self.cname, self.cnt, self.id, dst.shape, dst)
        print "%s.step[%d]-%s src.shape = %s, dst.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape,),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst = self.inputs['y']['val'].T
            
            # st = np.hstack((src, dst))
            
            # te = compute_transfer_entropy(dst, src)
            te = compute_transfer_entropy(dst, src, delay = -i)
            tes.append(te.copy())
        print ""

        tes = np.array(tes)
        # print "%s-%s.step tes.shape = %s / %s" % (self.cname, self.id, tes.shape, tes.T.shape)
        # self.te[:,0] = tes.flatten()
        self.te = tes.T.copy()

class CTEBlock2(PrimBlock2):
    """!@brief Compute elementwise conditional transfer entropy from src to dst variables conditioned
    on cond variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        # set fedaults
        self.xcond = False
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        ctes = []
        # jh = self.measH.step(st)
        dst  = self.inputs['x']['val'].T
        src  = self.inputs['y']['val'].T
        cond = self.inputs['cond']['val'].T
        # print "%s.step[%d]-%s src.sh = %s, src = %s" % (self.cname, self.cnt, self.id, src.shape, src)
        # print "%s.step[%d]-%s dst.sh = %s, dst = %s" % (self.cname, self.cnt, self.id, dst.shape, dst)
        # print "%s.step[%d]-%s cond.sh = %s, cond = %s" % (self.cname, self.cnt, self.id, cond.shape, cond)
        print "%s.step[%d]-%s src.shape = %s, dst.shape = %s, cond.shape = %s" % (self.cname, self.cnt, self.id, src.shape, dst.shape, cond.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src  = np.roll(self.inputs['x']['val'].T, shift = i, axis = 0)
            # dst  = self.inputs['y']['val'].T
            
            # st = np.hstack((src, dst))
            
            # cte = compute_conditional_transfer_entropy(dst, src, cond)
            cte = compute_conditional_transfer_entropy(src, dst, cond, delay = -i, xcond = self.xcond)
            ctes.append(cte.copy())
        print ""

        ctes = np.array(ctes)
        # print "%s-%s.step ctes.shape = %s / %s" % (self.cname, self.id, ctes.shape, ctes.T.shape)
        # print "ctes", ctes
        # self.cte[:,0] = ctes.flatten()
        self.cte = ctes.T.copy()

################################################################################
# multivariate versions
class MIMVBlock2(PrimBlock2):
    """!@brief Compute the multivariate mutual information between X and Y, aka the total MI"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # self.meas = measMI()
        # self.measH = measH()

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by" % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape),
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        mimvs = []
        jhs = []
        src = self.inputs['y']['val'].T
        dst = self.inputs['x']['val'].T
        
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            # print "self.inputs['x']['val'].T.shape", self.inputs['x']['val'].T.shape
            # dst_ = np.roll(dst, shift = i, axis = 0)
            
            # st = np.hstack((src, dst))
            # jh = self.measH.step(st)
            # mi = self.meas.step(st, st)

            # mi = compute_mi_multivariate(data = {'X': src, 'Y': dst_}, estimator = "kraskov2", normalize = True)
            mi = compute_mi_multivariate(data = {'X': src, 'Y': dst}, estimator = "kraskov2", normalize = True, delay = -i)
            # print "mimv = %s" % mi
            mimvs.append(mi)
            # jhs.append(jh)
        print ""
        mimvs = np.array(mimvs)
        # jhs = np.array(jhs)
        # print "@%d mimvs.shape = %s" % (self.cnt, mimvs.shape, )
        # print "@%d mimvs       = %s" % (self.cnt, mimvs, )
                    
        # mi = self.meas.step(self.inputs['x']['val'].T, self.inputs['y']['val'].T)
        # np.fill_diagonal(mi, np.min(mi))
        # print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        # maxjh = np.max(jhs)
        # print "mutual info self.mi.shape = %s, mi.shape = %s, maxjh = %s" % (self.mi.shape, mi.shape, maxjh)
        
        # self.mi[:,0] = (mi/jh).flatten()
        # self.mimv[0,shiftsl] = mimvs.flatten() # /maxjh
        self.mimv[0,shiftsl] = mimvs.flatten() # /maxjh
        print "@%d self.mimv.shape = %s, mimv = %s" % (self.cnt, self.mimv.shape, self.mimv)

class TEMVBlock2(PrimBlock2):
    """!@brief Compute the multivariate transfer entropy from X to Y, aka the total TE"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        temvs = []
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        # jh = self.measH.step(st)
        src = self.inputs['y']['val'].T
        dst = self.inputs['x']['val'].T
        print "%s.step[%d]-%s self.inputs['x']['val'].T.shape = %s, shifting by" % (self.cname, self.cnt, self.id, self.inputs['x']['val'].T.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # dst_ = np.roll(dst, shift = i, axis = 0)
            
            # st = np.hstack((src, dst))
            
            # mi = compute_transfer_entropy_multivariate(src, dst, delay = -self.shift[0]+i)
            mi = compute_transfer_entropy_multivariate(src, dst, delay = -i)
            temvs.append(mi)
        print ""
        temvs = np.array(temvs)
        # self.temv[0,shiftsl] = temvs.flatten()
        self.temv[0,shiftsl] = temvs.flatten()

class CTEMVBlock2(PrimBlock2):
    """!@brief Compute the multivariate conditional transfer entropy from X to Y, conditioned on C, aka the total CTE (doesn't exist yet)"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
