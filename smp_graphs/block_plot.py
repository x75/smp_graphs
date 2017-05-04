
import matplotlib.pyplot as plt

from smp_graphs.block import decStep, decInit
from smp_graphs.block import PrimBlock2

from smp_base.plot import makefig, timeseries, histogram

################################################################################
# Plotting blocks

class TimeseriesPlotBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        # print "plotblock conf", self.ibuf
        # print "plotblock real", self.inputs['d1'][0].shape
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            self.debug_print("%s.step ibuf = %s", (self.cname, self.inputs['d1'][0]))
            # plt.plot(self.bufs['ibuf'].T)
            # plt.show()
            rows = len(self.subplots)
            cols = len(self.subplots[0])
            fig = makefig(rows = rows, cols = cols)
            # self.debug_print("fig.axes = %s", (fig.axes, ))
            for i, plot in enumerate(self.subplots):
                for j, plotconf in enumerate(plot):
                    idx = (i*cols)+j
                    self.debug_print("%s.step idx = %d, conf = %s, data = %s", (
                        self.__class__.__name__, idx, plotconf, self.inputs[plotconf['input']][0]))
                    plotconf['plot'](
                        fig.axes[idx],
                        self.inputs[plotconf['input']][0].T) # [plotconf['slice'][0]:plotconf['slice'][1]].T)
                    # timeseries(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
                    # histogram(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
            fig.show()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))
            
