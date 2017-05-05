
import matplotlib.pyplot as plt
import numpy as np

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
        if len(self.inputs) < 1: return
        # print "plotblock conf", self.ibuf
        # print "plotblock real", self.inputs['d1'][0].shape
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            # self.debug_print("step ibuf = %s, in(%s).shape = %s", (self.ibuf, ink, inv[0]))
            # plt.plot(self.bufs['ibuf'].T)
            # plt.show()
            
            rows = len(self.subplots)
            cols = len(self.subplots[0])
            # create figure
            fig = makefig(rows = rows, cols = cols)
            # self.debug_print("fig.axes = %s", (fig.axes, ))

            # loop over configured plots
            for i, plot in enumerate(self.subplots):
                for j, plotconf in enumerate(plot):
                    idx = (i*cols)+j
                    # self.debug_print("%s.step idx = %d, conf = %s, data = %s", (
                    #     self.__class__.__name__, idx,
                    #     plotconf, self.inputs[plotconf['input']][0]))
                    if type(plotconf['input']) is str:
                        t = np.linspace(0, self.blocksize-1, self.blocksize)
                        plotdata = self.inputs[plotconf['input']][0].T
                    elif type(plotconf['input']) is list:
                        t = self.inputs[plotconf['input'][0]][0].T
                        plotdata = self.inputs[plotconf['input'][1]][0].T
                    # fix nans
                    plotdata[np.isnan(plotdata)] = -1.0
                    
                    # print plotdata
                    plotconf['plot'](
                        fig.axes[idx],
                        data = plotdata, ordinate = t) # [plotconf['slice'][0]:plotconf['slice'][1]].T)
                    # timeseries(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
                    # histogram(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
            fig.show()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))
            
