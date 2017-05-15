"""smp_graphs/measures/information theoretic (infth)
"""

import numpy as np

from smp_base.measures_infth import measMI, measH, compute_mutual_information, infth_mi_multivariate, compute_information_distance, compute_transfer_entropy, compute_conditional_transfer_entropy

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2


# block wrappers for smp_base/meas_infth.py with general smp_base/meas.py pattern
# sift input from
#  - smp/smp/infth.py
#  - smp/playground/infth_feature_relevance.py
#  - smp_sphero (was smp_infth)
#  - evoplast/ep3.py
#  - smp/infth
#  - smp/infth/infth_homeokinesis_analysis_cont.py
#  - smp/infth/infth_playground
#  - smp/infth/infth_explore.py
#  - smp/infth/infth_pointwise_plot.py
#  - smp/infth/infth_measures.py: unfinished
#  - smp/infth/infth_playground.py
#  - smp/infth/infth_EH-2D.py
#  - smp/infth/infth_EH-2D_clean.py

class MIBlock2(PrimBlock2):
    """Compute cross-correlation functions among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.meas = measMI()
        self.measH = measH()

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        mis = []
        jhs = []
        for i in range(self.shift[0], self.shift[1]):
            print "self.inputs['x'][0].T.shape", self.inputs['x'][0].T.shape
            src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            dst = self.inputs['y'][0].T
            
            st = np.hstack((src, dst))
            jh = self.measH.step(st)
            # mi = self.meas.step(st, st)

            mi = self.meas.step(src, dst)
            mis.append(mi.copy())
            jhs.append(jh)

        mis = np.array(mis)
        jhs = np.array(jhs)
        print "mis.shape = %s, jhs.shape = %s" % (mis.shape, jhs.shape)
                    
        # mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        # np.fill_diagonal(mi, np.min(mi))
        print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)

        maxjh = np.max(jhs)
        print "mutual info self.mi.shape = %s, mi.shape = %s, maxjh = %s" % (self.mi.shape, mi.shape, maxjh)
        
        # self.mi[:,0] = (mi/jh).flatten()
        self.mi[:,0] = mis.flatten()/maxjh

class InfoDistBlock2(PrimBlock2):
    """Compute cross-correlation functions among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # self.meas = measMI()
        # self.measH = measH()

    @decStep()
    def step(self, x = None):
        # print "%s meas = %s" % (self.cname, self.meas)
        mis = []
        for i in range(self.shift[0], self.shift[1]):
            print "self.inputs['x'][0].T.shape", self.inputs['x'][0].T.shape
            src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            dst = self.inputs['y'][0].T
            
            st = np.hstack((src, dst))
            # jh = self.measH.step(st)
            # mi = self.meas.step(st, st)
            # mi = compute_information_distance(st, st)

            mi = compute_information_distance(src, dst)
            mis.append(mi.copy())
            # mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        
            # if src eq dst 
            # blank out the diagonal since it's always one
            # np.fill_diagonal(mi, np.max(mi))
        mis = np.array(mis)
            
        print "%s.%s infodist = %s" % (self.cname, self.id, mi)
        print "infodist block", self.infodist.shape, mi.shape
        
        # self.infodist[:,0] = mi.flatten()
        self.infodist[:,0] = mis.flatten()
        
class TEBlock2(PrimBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        # jh = self.measH.step(st)
        for i in range(self.shift[0], self.shift[1]):
            src = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            dst = self.inputs['y'][0].T
            
            st = np.hstack((src, dst))
            
            mi = compute_transfer_entropy(dst, src)
            mis.append(mi)

        mis = np.array(mis)
        self.te[:,0] = mis.flatten()

class CTEBlock2(PrimBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        mis = []
        # jh = self.measH.step(st)
        for i in range(self.shift[0], self.shift[1]):
            src  = np.roll(self.inputs['x'][0].T, shift = i, axis = 0)
            dst  = self.inputs['y'][0].T
            cond = self.inputs['cond'][0].T
            
            # st = np.hstack((src, dst))
            
            mi = compute_conditional_transfer_entropy(dst, src, cond)
            mis.append(mi)

        mis = np.array(mis)
        self.cte[:,0] = mis.flatten()
