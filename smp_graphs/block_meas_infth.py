"""smp_graphs/measures/information theoretic (infth)
"""

import sys
import numpy as np

from smp_base.measures_infth import measMI, measH, compute_mutual_information, infth_mi_multivariate, compute_information_distance, compute_transfer_entropy, compute_conditional_transfer_entropy, compute_mi_multivariate, compute_transfer_entropy_multivariate

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2


# block wrappers for smp_base/measures_infth.py, similar to the general smp_base/measures.py pattern

class JHBlock2(PrimBlock2):
    """!@brief Compute scalar joint entropy of multivariate data"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
    
    @decStep()
    def step(self, x = None):
        jhs = []
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        
        src = self.inputs['x'][0].T
        dst = self.inputs['y'][0].T
        st = np.hstack((src, dst))
        jh0 = compute_mi_multivariate(data = {'X': st, 'Y': st}, delay = 0)
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src_ = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            # src_ = np.roll(src, shift = i, axis = 0)
            # dst = self.inputs['y'][0].T
            # st = np.hstack((src_, dst))
            # jh = compute_mi_multivariate(data = {'X': st, 'Y': st})

            # use jidt's delay param
            jh = compute_mi_multivariate(data = {'X': st, 'Y': st}, delay = -i)
            # print "%s.step[%d] data = %s, jh = %f" % (self.cname, self.cnt, st.shape, jh)
            jhs.append(jh)
        print ""
        jhs = np.array(jhs) # /jh0
        self.jh[0,shiftsl] = jhs
        
class MIBlock2(PrimBlock2):
    """!@brief Compute elementwise mutual information among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        # jhs = []
        
        dst = self.inputs['x'][0].T
        src = self.inputs['y'][0].T
        st = np.hstack((src, dst))

        # shouldn't change too much under time shift
        jh = compute_mi_multivariate(data = {'X': st, 'Y': st})
        
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            # dst = self.inputs['y'][0].T
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
        # print "mis = ", mis.flatten()
        # jhs = np.array(jhs)
        # print "mis.shape = %s, jhs.shape = %s" % (mis.shape, jhs.shape)
                    
        # mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        # np.fill_diagonal(mi, np.min(mi))
        # print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        # maxjh = np.max(jhs)
        # print "mutual info self.mi.shape = %s, mi.shape = %s, maxjh = %s" % (self.mi.shape, mi.shape, maxjh)
        
        # self.mi[:,0] = (mi/jh).flatten()
        # self.mi[:,0] = mis.flatten()/maxjh
        # normalized by joint entropy
        # self.mi[:,0] = mis.flatten()/jh
        self.mi[:,0] = mis.flatten()

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
        src = self.inputs['x'][0].T
        dst = self.inputs['y'][0].T
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s" % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape)
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            # src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            # dst = self.inputs['y'][0].T
            
            # st = np.hstack((src, dst))
            # jh = self.measH.step(st)
            # mi = self.meas.step(st, st)
            # mi = compute_information_distance(st, st)

            mi = compute_information_distance(src, dst, delay = -i)
            mis.append(mi.copy())
            # mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        
            # if src eq dst 
            # blank out the diagonal since it's always one
            # np.fill_diagonal(mi, np.max(mi))
        mis = np.array(mis)
        print ""            
        # print "%s.%s infodist = %s" % (self.cname, self.id, mi)
        # print "infodist block", self.infodist.shape, mi.shape
        
        # self.infodist[:,0] = mi.flatten()
        self.infodist[:,0] = mis.flatten()
        
class TEBlock2(PrimBlock2):
    """!@brief Compute elementwise transfer entropy from src to dst variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        src = self.inputs['x'][0].T
        dst = self.inputs['y'][0].T
        # jh = self.measH.step(st)
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s, shifting by " % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape),
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            # dst = self.inputs['y'][0].T
            
            # st = np.hstack((src, dst))
            
            # mi = compute_transfer_entropy(dst, src)
            mi = compute_transfer_entropy(dst, src, delay = -i)
            mis.append(mi)
        print ""

        mis = np.array(mis)
        self.te[:,0] = mis.flatten()

class CTEBlock2(PrimBlock2):
    """!@brief Compute elementwise conditional transfer entropy from src to dst variables conditioned
    on cond variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        # jh = self.measH.step(st)
        dst  = self.inputs['x'][0].T
        src  = self.inputs['y'][0].T
        cond = self.inputs['cond'][0].T
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s" % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape)
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            
            # src  = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            # dst  = self.inputs['y'][0].T
            
            # st = np.hstack((src, dst))
            
            # mi = compute_conditional_transfer_entropy(dst, src, cond)
            mi = compute_conditional_transfer_entropy(src, dst, cond, delay = -i, xcond = self.xcond)
            mis.append(mi)
        print ""

        mis = np.array(mis)
        self.cte[:,0] = mis.flatten()

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
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s, shifting by" % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape),
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        mimvs = []
        jhs = []
        src = self.inputs['y'][0].T
        dst = self.inputs['x'][0].T
        
        for i in range(self.shift[0], self.shift[1]):
            print "%d" % (i, ),
            sys.stdout.flush()
            # print "self.inputs['x'][0].T.shape", self.inputs['x'][0].T.shape
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
        # print "mimvs.shape = %s" % (mimvs.shape, )
                    
        # mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        # np.fill_diagonal(mi, np.min(mi))
        # print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        # maxjh = np.max(jhs)
        # print "mutual info self.mi.shape = %s, mi.shape = %s, maxjh = %s" % (self.mi.shape, mi.shape, maxjh)
        
        # self.mi[:,0] = (mi/jh).flatten()
        self.mimv[0,shiftsl] = mimvs.flatten() # /maxjh

class TEMVBlock2(PrimBlock2):
    """!@brief Compute the multivariate transfer entropy from X to Y, aka the total TE"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        temvs = []
        shiftsl = slice(None, (self.shift[1] - self.shift[0]))
        # jh = self.measH.step(st)
        src = self.inputs['y'][0].T
        dst = self.inputs['x'][0].T
        print "%s.step[%d]-%s self.inputs['x'][0].T.shape = %s, shifting by" % (self.cname, self.cnt, self.id, self.inputs['x'][0].T.shape),
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
        self.temv[0,shiftsl] = temvs.flatten()

class CTEMVBlock2(PrimBlock2):
    """!@brief Compute the multivariate conditional transfer entropy from X to Y, conditioned on C, aka the total CTE (doesn't exist yet)"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
