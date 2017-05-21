
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

from smp_graphs.block import decStep, decInit
from smp_graphs.block import PrimBlock2
from smp_graphs.utils import myt

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
                self.debug_print("[%s]step in[%s].shape = %s", (self.id, ink, inv['shape']))

            plots = self.plot_subplots()
            
            # set figure title and show the fig
            self.fig.suptitle("%s: %s-%s" % (self.top.id, self.cname, self.id))
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
        # filename = "data/%s_%s_%s_%s.%s" % (plotinst.top.id, plotinst.id, "_".join(plotinst.inputs.keys()), subplotstr, plotinst.savetype)
        # filename = "data/%s_%s_%s.%s" % (plotinst.top.id, plotinst.id, "_".join(plotinst.inputs.keys()), plotinst.savetype)
        filename = "data/%s_%s.%s" % (plotinst.top.id, plotinst.id, plotinst.savetype)
        print "%s.save filename = %s, subplotstr = %s" % (plotinst.cname, filename, subplotstr)
        plotinst.fig.set_size_inches((min(plotinst.fig_cols * 2 * 2.5, 20), min(plotinst.fig_rows * 1.2 * 2.5, 12)))
        try:
            plotinst.fig.savefig(filename, dpi=300, bbox_inches="tight")
        except Exception, e:
            print "%s error %s" % ('FigPlotBlock2', e)

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
                        xslice = slice(0, self.blocksize)
                        
                    # configure x axis
                    if subplotconf.has_key('xaxis'):
                        t = self.inputs[subplotconf['xaxis']]['val'].T[xslice]
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
                        # print "%s.plot_subplots k = %s, ink = %s" % (self.cname, k, ink)
                        # plotdata[ink] = self.inputs[ink]['val'].T[xslice]
                        plotdata[ink] = myt(self.inputs[ink]['val'])[xslice].reshape((self.blocksize, -1))
                        # print "plotdata", plotdata[ink]
                        # fix nans
                        plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                        plotvar += "%s, " % (self.inputs[ink]['bus'],)
                    title += plotvar
                        
                    # different 
                    if subplotconf.has_key('mode'):
                        # ivecs = tuple(self.inputs[ink][0].T[xslice] for k, ink in enumerate(subplotconf['input']))
                        ivecs = tuple(self.inputs[ink][0].myt()[xslice] for k, ink in enumerate(subplotconf['input']))
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
                        inv = self.inputs[subplotconf['xaxis']]
                        if inv.has_key('bus'):
                            plotvar += " over %s" % (inv['bus'], )
                        else:
                            plotvar += " over %s" % (inv['val'], )
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
                        # print "v", inv
                        # this is the plotfunction from the config
                        subplotconf['plot'](
                            self.fig.axes[idx],
                            data = inv, ordinate = t, label = "%s" % ink,
                            title = title)
                        # labels.append("%s" % ink)
                        # metadata
                    # self.fig.axes[idx].legend()
                    # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                    # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)

# plot a matrix via imshow/pcolor
class ImgPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        numrows = len(self.subplots)
        numcols = len(self.subplots[0])

        extrema = np.zeros((2, numrows, numcols))
        
        vmins_sb = [[] for i in range(numcols)]
        vmaxs_sb = [[] for i in range(numcols)]

        vmins = [None for i in range(numcols)]
        vmaxs = [None for i in range(numcols)]
        vmins_r = [None for i in range(numrows)]
        vmaxs_r = [None for i in range(numrows)]
        
        for i, subplot in enumerate(self.subplots): # rows
            for j, subplotconf in enumerate(subplot): # cols
                vmins_sb[j].append(np.min(self.inputs[subplotconf['input']][0]))
                vmaxs_sb[j].append(np.max(self.inputs[subplotconf['input']][0]))
                extrema[0,i,j] = np.min(self.inputs[subplotconf['input']][0])
                extrema[1,i,j] = np.max(self.inputs[subplotconf['input']][0])
                # print "i", i, "j", j, vmins_sb, vmaxs_sb
        print "mins", self.id, extrema[0]
        print "maxs", extrema[1]
        vmins_sb = np.array(vmins_sb)
        vmaxs_sb = np.array(vmaxs_sb)
        # print "vmins_sb, vmaxs_sb", i, j, vmins_sb.shape, vmaxs_sb.shape

        for i in range(numcols):
            vmins[i] = np.min(vmins_sb[i])
            # vmins[1] = np.min(vmins_sb[1])
            vmaxs[i] = np.max(vmaxs_sb[i])
            # vmaxs[1] = np.max(vmaxs_sb[1])

        # for i in range(numrows):
        #     vmins_r[i] = np.min(vmins_sb[i])
        #     # vmins[1] = np.min(vmins_sb[1])
        #     vmaxs_r[i] = np.max(vmaxs_sb[i])
        #     # vmaxs[1] = np.max(vmaxs_sb[1])
            
        rowmins = np.min(extrema[0], axis = 0) 
        rowmaxs = np.max(extrema[1], axis = 0) 
        colmins = np.min(extrema[0], axis = 1) 
        colmaxs = np.max(extrema[1], axis = 1)
        
        if True:
            for i, subplot in enumerate(self.subplots): # rows
                for j, subplotconf in enumerate(subplot): # cols
                    assert subplotconf.has_key('shape'), "image plot needs shape spec"

                    # map loop indices to gridspec linear index
                    idx = (i*self.fig_cols)+j
                    # print "self.inputs[subplotconf['input']][0].shape", self.inputs[subplotconf['input']][0].shape

                    xslice = slice(None)
                    yslice = slice(None)
                    
                    # check for slice specs
                    if subplotconf.has_key('xslice'):
                        xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                        # print "xslice", xslice, self.inputs[subplotconf['input']][0].shape

                    if subplotconf.has_key('yslice'):
                        yslice = slice(subplotconf['yslice'][0], subplotconf['yslice'][1])
                        # print "yslice", yslice, self.inputs[subplotconf['input']][0].shape

                    # min, max values for colormap
                    axis = 0
                    aidx = j
                    if subplotconf.has_key('vaxis'):
                        if subplotconf['vaxis'] == 'rows':
                            axis = 1
                            aidx = i
                            
                    vmin = np.min(extrema[0], axis = axis)[aidx]
                    vmax = np.max(extrema[1], axis = axis)[aidx]
                    # print "vmins, vmaxs", i, vmins, vmaxs
                    # vmin = vmins[sbidx]
                    # vmax = vmaxs[sbidx]
                    # vmin = extrema[0]

                    # print "vmin", vmin, "vmax", vmax
                    if subplotconf.has_key('vmin'):
                        vmin = subplotconf['vmin']
                    if subplotconf.has_key('vmax'):
                        vmax = subplotconf['vmax']
                        
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][xslice,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,xslice]
                    
                    plotdata_cand = self.inputs[subplotconf['input']][0][yslice,xslice]
                    # print "%s[%d]-%s.step, inputs = %s, %s " % (self.cname, self.cnt, self.id, self.inputs[subplotconf['input']][0].shape,
                    #                                         self.inputs[subplotconf['input']][0])
                    # print "%s[%d]-%s.step plotdata_cand.shape" % (self.cname, self.cnt, self.id), plotdata_cand.shape, subplotconf['shape'], xslice, yslice
                    # print "plotdata_cand", plotdata_cand
                    
                    plotdata = {}
                    plotdata['i_%d_%d' % (i, j)] = plotdata_cand.reshape(subplotconf['shape'])
                    plotvar = self.inputs[subplotconf['input']][2]

                    title = "img plot"
                    if subplotconf.has_key('title'): title = subplotconf['title']
                    # for k, ink in enumerate(subplotconf['input']):
                    #     plotdata[ink] = self.inputs[ink][0].T[xslice]
                    #     # fix nans
                    #     plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                    #     plotvar += "%s, " % (self.inputs[ink][2],)
                    # title += plotvar

                    # colormap
                    if not subplotconf.has_key('cmap'):
                        subplotconf['cmap'] = 'gray'
                    cmap = plt.get_cmap(subplotconf['cmap'])
                                                                
                    # plot the plotdata
                    for ink, inv in plotdata.items():
                        # print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s" % (self.cname, ink, plotvar, inv.shape)
                        # subplotconf['plot'](
                        #     self.fig.axes[idx],
                        #     data = inv, ordinate = t)
                        # metadata
                        ax = self.fig.axes[idx]
                        # mormalize to [0, 1]
                        # mpl = ax.imshow(inv, interpolation = "none")
                        # Linv = np.log(inv + 1)
                        Linv = inv
                        # print "Linv", Linv
                        mpl = ax.pcolorfast(Linv, vmin = vmin, vmax = vmax, cmap = cmap)
                        # mpl = ax.pcolorfast(Linv, vmin = vmins[j], vmax = vmaxs[j], cmap = cmap)
                        # mpl = ax.pcolorfast(Linv, vmin = -2, vmax = 2, cmap = cmap)
                        # mpl = ax.pcolormesh(Linv, cmap = cmap)
                        # mpl = ax.pcolor(Linv)
                        # mpl = ax.pcolorfast(Linv)
                        # mpl = ax.imshow(Linv, interpolation = "none")
                        ax.grid()
                        # Linv = inv
                        # mpl = ax.pcolormesh(
                        #     Linv,
                        #     norm = colors.LogNorm(vmin=Linv.min(), vmax=Linv.max()))
                        # ax.grid()
                        # plt.colorbar(mappable = mpl, ax = ax)
                    # ax.set_aspect(10)
                    # plt.colorbar(mappable = mpl, ax = ax, orientation = "horizontal")
                    # ax.set_title("%s of %s" % ('matrix', plotvar, ), fontsize=8)
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])

################################################################################
# non FigPlot plot blocks
class SnsMatrixPlotBlock2(PrimBlock2):
    """!@brief Plotting block doing seaborn pairwaise matrix plots: e.g. scatter, hexbin, ...

Seaborne manages figures itself, so it can't be a FigPlotBlock2
    
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
            ivecs = tuple(self.inputs[ink]['val'].T for k, ink in enumerate(subplotconf['input']))
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
        
                    
