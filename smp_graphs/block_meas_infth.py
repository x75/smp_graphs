"""smp_graphs/measures/information theoretic (infth)
"""

import numpy as np

from smp_base.measures_infth import measMI, measH

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
        st = np.hstack((self.inputs['x'][0].T, self.inputs['y'][0].T))
        jh = self.measH.step(st)
        mi = self.meas.step(self.inputs['x'][0].T, self.inputs['y'][0].T)
        print "%s.%s mi = %s, jh = %s, normalized mi = mi/jh = %s" % (self.cname, self.id, mi, jh, mi/jh)
        self.mi[:,0] = mi.flatten()
