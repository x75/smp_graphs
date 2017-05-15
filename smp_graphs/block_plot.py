
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

from smp_graphs.block import decStep, decInit
from smp_graphs.block import PrimBlock2

from smp_base.plot import makefig, timeseries, histogram

################################################################################
# Plotting blocks

class FigPlotBlock2(PrimBlock2):
    """!@brief Basic plotting block

params
 - blocksize: usually numsteps (meaning plot all data created by that episode/experiment)
 - subplots: array of arrays, each cell of that matrix hold on subplot configuration dict
  - subplotconf: dict with inputs: list of input keys, plot: plotting function pointer
"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # configure figure and plot axes
        self.fig_rows = len(self.subplots)
        self.fig_cols = len(self.subplots[0])
        # create figure
        self.fig = makefig(rows = self.fig_rows, cols = self.fig_cols)
        # self.debug_print("fig.axes = %s", (self.fig.axes, ))
        
    @decStep()
    def step(self, x = None):
        # have inputs at all?
        if len(self.inputs) < 1: return

        # make sure that data can have been generated
        if self.cnt > 0 and (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for ink, inv in self.inputs.items():
                self.debug_print("[%s]step in[%s].shape = %s", (self.id, ink, inv[0]))

            plots = self.plot_subplots()
            
            # set figure title and show the fig
            self.fig.suptitle("%s" % (self.top.id,))
            self.fig.show()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))

    def plot_subplots(self):
        print "%s.plot_subplots(): implement me" % (self.cname,)

class PlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        """loop over configured subplot and plot the data according to config"""
        if True:
            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    idx = (i*self.fig_cols)+j
                    self.debug_print("[%s]step idx = %d, conf = %s, data = %s/%s", (
                        self.id, idx,
                        subplotconf, subplotconf['input'], self.inputs[subplotconf['input']]))
                    # self.inputs[subplotconf['input']][0]))

                    # x axis slice spec
                    if subplotconf.has_key('xslice'):
                        xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                    else:
                        xslice = slice(None)
                        
                    # configure x axis
                    if subplotconf.has_key('xaxis'):
                        t = self.inputs[subplotconf['xaxis']][0].T[xslice]
                    else:
                        t = np.linspace(0, self.blocksize-1, self.blocksize)[xslice]

                    # assert input an array 
                    if type(subplotconf['input']) is str:
                        subplotconf['input'] = [subplotconf['input']]

                    # plotdata = self.inputs[subplotconf['input']][0].T
                    # elif type(subplotconf['input']) is list:
                    # plotdata = self.inputs[subplotconf['input'][1]][0].T
                    plotdata = {}
                    plotvar = ""
                    for k, ink in enumerate(subplotconf['input']):
                        plotdata[ink] = self.inputs[ink][0].T[xslice]
                        # fix nans
                        plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                        plotvar += "%s, " % (self.inputs[ink][2],)
                        
                    # different 
                    if subplotconf.has_key('mode'):
                        ivecs = tuple(self.inputs[ink][0].T[xslice] for k, ink in enumerate(subplotconf['input']))
                        # for item in ivecs:
                        #     print "ivec.shape", item.shape
                        plotdata = {}
                        if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                            plotdata['all'] = np.hstack(ivecs)
                        
                    if hasattr(subplotconf['plot'], 'func_name'):
                        # plain function
                        plottype = subplotconf['plot'].func_name
                    elif hasattr(subplotconf['plot'], 'func'):
                        # partial'ized func
                        plottype = subplotconf['plot'].func.func_name
                    else:
                        # unknown func type
                        plottype = "unk type"

                    # if type(subplotconf['input']) is list:
                    if subplotconf.has_key('xaxis'):
                        plotvar += " over %s" % (self.inputs[subplotconf['xaxis']][2], )
                        # plotvar = ""
                        # # FIXME: if len == 2 it is x over y, if len > 2 concatenation
                        # for k, inputvar in enumerate(subplotconf['input']):
                        #     tmpinput = self.inputs[inputvar][2]
                        #     plotvar += str(tmpinput)
                        #     if k != (len(subplotconf['input']) - 1):
                        #         plotvar += " revo "
                    # else:
                    # plotvar = self.inputs[subplotconf['input'][0]][2]
                        
                        
                    # plot the plotdata
                    for ink, inv in plotdata.items():
                        # print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s, t.sh = %s" % (self.cname, ink, plotvar, inv.shape, t.shape)
                        subplotconf['plot'](
                            self.fig.axes[idx],
                            data = inv, ordinate = t)
                        # metadata
                    self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                    # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)


class SnsMatrixPlotBlock2(PrimBlock2):
    """!@brief Plotting block doing seaborn pairwaise matrix plots: e.g. scatter, hexbin, ...

params
 - blocksize: usually numsteps (meaning plot all data created by that episode/experiment)
 - f_plot_diag: diagonal cells
 - f_plot_matrix: off diagonal cells
 - numpy matrix of data, plot iterates over all pairs with given function
"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        print "%s.step inputs: %s"  % (self.cname, self.inputs.keys())

        subplotconf = self.subplots[0][0]
        
        # different 
        if subplotconf.has_key('mode'):
            ivecs = tuple(self.inputs[ink][0].T for k, ink in enumerate(subplotconf['input']))
            for ivec in ivecs:
                print "ivec.shape", ivec.shape
            plotdata = {}
            if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                plotdata['all'] = np.hstack(ivecs)

        data = plotdata['all']

        print "SnsPlotBlock2:", data.shape
        scatter_data_raw  = data
        scatter_data_cols = ["x_%d" % (i,) for i in range(data.shape[1])]

        # prepare dataframe
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
        
        g = sns.PairGrid(df)
        g.map_diag(plt.hist)
        # g.map_diag(sns.kdeplot)
        g.map_offdiag(plt.hexbin, cmap="gray", gridsize=20, bins="log");
        # g.map_offdiag(plt.plot, linestyle = "None", marker = "o", alpha = 0.5) # , bins="log");

        # print "dir(g)", dir(g)
        # print g.diag_axes
        # print g.axes
    
        # for i in range(data.shape[1]):
        #     for j in range(data.shape[1]): # 1, 2; 0, 2; 0, 1
        #         if i == j:
        #             continue
        #         # column gives x axis, row gives y axis, thus need to reverse the selection for plotting goal
        #         # g.axes[i,j].plot(df["%s%d" % (self.cols_goal_base, j)], df["%s%d" % (self.cols_goal_base, i)], "ro", alpha=0.5)
        #         g.axes[i,j].plot(df["x_%d" % (j,)], df["x_%d" % (i,)], "ro", alpha=0.5)

        # plt.show()
        

class ImgPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):

            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    # ink = subplot
                    idx = (i*self.fig_cols)+j

                    plotdata = {}
                    plotdata['i_%d_%d' % (i, j)] = self.inputs[subplotconf['input']][0][:,0].reshape(subplotconf['shape'])
                    plotvar = self.inputs[subplotconf['input']][2]
                                            
                    # plot the plotdata
                    for ink, inv in plotdata.items():
                        print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s" % (self.cname, ink, plotvar, inv.shape)
                        # subplotconf['plot'](
                        #     self.fig.axes[idx],
                        #     data = inv, ordinate = t)
                        # metadata
                        ax = self.fig.axes[idx]
                        # mormalize to [0, 1]
                        # mpl = ax.imshow(inv, interpolation = "none")
                        Linv = np.log(inv + 1)
                        mpl = ax.pcolormesh(Linv)
                        ax.grid()
                        # Linv = inv
                        # mpl = ax.pcolormesh(
                        #     Linv,
                        #     norm = colors.LogNorm(vmin=Linv.min(), vmax=Linv.max()))
                        # ax.grid()
                        # plt.colorbar(mappable = mpl, ax = ax)
                        plt.colorbar(mappable = mpl, ax = ax)
                    ax.set_title("%s of %s" % ('matrix', plotvar, ), fontsize=8)
