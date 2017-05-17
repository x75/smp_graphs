
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
# pull stuff from
# smp/im/im_quadrotor_plot
# 

def subplot_input_fix(input_spec):
    # assert input an array 
    if type(input_spec) is str:
        return [input_spec]
    else:
        return input_spec


class FigPlotBlock2(PrimBlock2):
    """!@brief Basic plotting block

params
 - blocksize: usually numsteps (meaning plot all data created by that episode/experiment)
 - subplots: array of arrays, each cell of that matrix hold on subplot configuration dict
  - subplotconf: dict with inputs: list of input keys, plot: plotting function pointer
"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        self.wspace = 0.0
        self.hspace = 0.0
        self.saveplot = False
        self.savetype = "jpg"
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # configure figure and plot axes
        self.fig_rows = len(self.subplots)
        self.fig_cols = len(self.subplots[0])
        # create figure
        self.fig = makefig(rows = self.fig_rows, cols = self.fig_cols, wspace = self.wspace, hspace = self.hspace)
        # self.fig.tight_layout(pad = 1.0)
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
            self.fig.suptitle("%s" % (self.top.id, ))
            self.fig.show()

            if self.saveplot:
                print "%s.step saving plot" % (self.cname,)
                FigPlotBlock2.save(self)
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))

    def plot_subplots(self):
        print "%s.plot_subplots(): implement me" % (self.cname,)

    @staticmethod
    def save(plotinst):
        """save the figure using configuration options"""
        subplotstr = "_".join(np.array([["r%d_c%d_%s" % (r, c, "_".join(subplot_input_fix(sbc['input'])),) for c,sbc in enumerate(sbr)] for r, sbr in enumerate(plotinst.subplots)]).flatten())
        filename = "data/%s_%s_%s_%s.%s" % (plotinst.top.id, plotinst.id, "_".join(plotinst.inputs.keys()), subplotstr, plotinst.savetype)
        print "%s.save filename = %s, subplotstr = %s" % (plotinst.cname, filename, subplotstr)
        plotinst.fig.set_size_inches((plotinst.fig_cols * 2 * 2.5, plotinst.fig_rows * 1.2 * 2.5))
        plotinst.fig.savefig(filename, dpi=300, bbox_inches="tight")

class PlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        """loop over configured subplot and plot the data according to config"""
        if True:
            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    idx = (i*self.fig_cols)+j
                    # self.debug_print("[%s]step idx = %d, conf = %s, data = %s/%s", (
                    #     self.id, idx,
                    #     subplotconf, subplotconf['input'], self.inputs[subplotconf['input']]))
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

                    subplotconf['input'] = subplot_input_fix(subplotconf['input'])
                        
                    # plotdata = self.inputs[subplotconf['input']][0].T
                    # elif type(subplotconf['input']) is list:
                    # plotdata = self.inputs[subplotconf['input'][1]][0].T
                    plotdata = {}
                    plotvar = " "
                    title = ""
                    if subplotconf.has_key('title'): title += subplotconf['title']
                    for k, ink in enumerate(subplotconf['input']):
                        plotdata[ink] = self.inputs[ink][0].T[xslice]
                        # fix nans
                        plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                        plotvar += "%s, " % (self.inputs[ink][2],)
                    title += plotvar
                        
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
                        plottype = "unk plottype"

                    # append to title
                    title += " " + plottype

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
                    labels = []
                    for ink, inv in plotdata.items():
                        # print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s, t.sh = %s" % (self.cname, ink, plotvar, inv.shape, t.shape)
                        # this is the plotfunction from the config
                        subplotconf['plot'](
                            self.fig.axes[idx],
                            data = inv, ordinate = t, label = "%s" % ink, title = title)
                        # labels.append("%s" % ink)
                        # metadata
                    # self.fig.axes[idx].legend()
                    # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
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
        self.saveplot = False
        self.savetype = 'jpg'
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
        g.map_offdiag(plt.hexbin, cmap="gray", gridsize=40, bins="log");
        # g.map_offdiag(plt.histogram2d, cmap="gray", bins=30)
        # g.map_offdiag(plt.plot, linestyle = "None", marker = "o", alpha = 0.5) # , bins="log");
        self.fig = g.fig
        self.fig_rows, self.fig_cols = g.axes.shape
        # print "dir(g)", dir(g)
        # print g.diag_axes
        # print g.axes
        if self.saveplot:
            FigPlotBlock2.save(self)
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
        numrows = len(self.subplots)
        numcols = len(self.subplots[0])
        
        vmins_sb = [[] for i in range(numcols)]
        vmaxs_sb = [[] for i in range(numcols)]

        vmins = [None for i in range(numcols)]
        vmaxs = [None for i in range(numcols)]
        
        for i, subplot in enumerate(self.subplots): # rows
            for j, subplotconf in enumerate(subplot): # cols
                vmins_sb[j].append(np.min(self.inputs[subplotconf['input']][0]))
                vmaxs_sb[j].append(np.max(self.inputs[subplotconf['input']][0]))
                # print "i", i, "j", j, vmins_sb, vmaxs_sb
        vmins_sb = np.array(vmins_sb)
        vmaxs_sb = np.array(vmaxs_sb)
        print "vmins_sb, vmaxs_sb", i, j, vmins_sb.shape, vmaxs_sb.shape

        for i in range(numcols):
            vmins[i] = np.min(vmins_sb[i])
            # vmins[1] = np.min(vmins_sb[1])
            vmaxs[i] = np.max(vmaxs_sb[i])
            # vmaxs[1] = np.max(vmaxs_sb[1])

        print "vmins, vmaxs", i, vmins, vmaxs

        if True:
            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    assert subplotconf.has_key('shape'), "image plot needs shape spec"
                    assert subplotconf.has_key('xslice'), "image plot needs shape spec"
                    # ink = subplot
                    idx = (i*self.fig_cols)+j
                    print "self.inputs[subplotconf['input']][0].shape", self.inputs[subplotconf['input']][0].shape

                    xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                    print "xslice", xslice, self.inputs[subplotconf['input']][0].shape

                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][xslice,0]
                    plotdata_cand = self.inputs[subplotconf['input']][0][:,xslice]
                    print "plotdata_cand.shape", plotdata_cand.shape
                    
                    plotdata = {}
                    plotdata['i_%d_%d' % (i, j)] = plotdata_cand.reshape(subplotconf['shape'])
                    plotvar = self.inputs[subplotconf['input']][2]

                    if not subplotconf.has_key('cmap'):
                        subplotconf['cmap'] = 'gray'
                    cmap = plt.get_cmap(subplotconf['cmap'])
                                                                
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
                        # Linv = np.log(inv + 1)
                        Linv = inv
                        mpl = ax.pcolormesh(Linv, vmin = vmins[j], vmax = vmaxs[j], cmap = cmap)
                        ax.grid()
                        # Linv = inv
                        # mpl = ax.pcolormesh(
                        #     Linv,
                        #     norm = colors.LogNorm(vmin=Linv.min(), vmax=Linv.max()))
                        # ax.grid()
                        # plt.colorbar(mappable = mpl, ax = ax)
                    # ax.set_aspect(1)
                    # plt.colorbar(mappable = mpl, ax = ax, orientation = "horizontal")
                    # ax.set_title("%s of %s" % ('matrix', plotvar, ), fontsize=8)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])
