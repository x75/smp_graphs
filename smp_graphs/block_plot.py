
import matplotlib.pyplot as plt

from smp_graphs.block import decStep, decInit
from smp_graphs.block import Block

from smp_base.plot import makefig, timeseries, histogram

################################################################################
# Plotting blocks

class TimeseriesPlotBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)

    @decStep()
    def step(self, x = None):
        if (self.cnt % self.blocksize) == (self.blocksize - 1):
            self.debug_print("%s.step ibuf = %s", (self.__class__.__name__, self.bufs['ibuf']))
            # plt.plot(self.bufs['ibuf'].T)
            # plt.show()
            rows = len(self.subplots)
            cols = len(self.subplots[0])
            fig = makefig(rows = rows, cols = cols)
            self.debug_print("fig.axes = %s", (fig.axes))
            for i, plot in enumerate(self.subplots):
                for j, plotconf in enumerate(plot):
                    idx = (i*cols)+j
                    self.debug_print("%s.step idx = %d, conf = %s", (
                        self.__class__.__name__, idx, plotconf))
                    plotconf['plot'](
                        fig.axes[idx],
                        self.bufs['ibuf'][plotconf['input']][plotconf['slice'][0]:plotconf['slice'][1]].T)
                    # timeseries(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
                    # histogram(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
            fig.show()
        else:
            self.debug_print("%s.step", (self.__class__.__name__))
