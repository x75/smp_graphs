
# import itertools

from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from  matplotlib import rcParams
import numpy as np
import pandas as pd
# FIXME: soft import
import seaborn as sns

from smp_graphs.block import decStep, decInit
from smp_graphs.block import PrimBlock2
from smp_graphs.utils import myt, mytupleroll
import smp_graphs.logging as log

from smp_base.dimstack import dimensional_stacking, digitize_pointcloud
from smp_base.plot     import makefig, timeseries, histogram, plot_img, plotfuncs

################################################################################
# Plotting blocks
# pull stuff from
# smp/im/im_quadrotor_plot
# ...

# FIXME: do some clean up here

rcParams['figure.titlesize'] = 8

def subplot_input_fix(input_spec):
    # assert input an array 
    if type(input_spec) is str or type(input_spec) is tuple:
        return [input_spec]
    else:
        return input_spec


class FigPlotBlock2(PrimBlock2):
    """!@brief Basic plotting block

    matplotlib figure based plot, creates the figure and a gridspec on init, uses fig.axes in the step function
    
    Arguments:
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

        if not hasattr(self, 'title'):
            # self.title = "%s - %s-%s" % (self.top.id, self.cname, self.id)
            self.title = "%s" % (self.top.id, )
        
        # create figure
        self.fig = makefig(
            rows = self.fig_rows, cols = self.fig_cols,
            wspace = self.wspace, hspace = self.hspace,
            title = self.title)
        # self.fig.tight_layout(pad = 1.0)
        # self.debug_print("fig.axes = %s", (self.fig.axes, ))

        # FIXME: too special
        self.isprimitive = False
        
    @decStep()
    def step(self, x = None):
        # have inputs at all?
        if len(self.inputs) < 1: return

        # make sure that data has been generated
        if (self.cnt % self.blocksize) in self.blockphase: # or (xself.cnt % xself.rate) == 0:

            # # HACK: override block inputs with log.log_store
            print "log.log_store", log.log_store.keys()
            # log.log_pd_store()
            # for ink, inv in self.inputs.items():
            #     bus = '/%s' % (inv['bus'], )
            #     print "ink", ink, "inv", inv['bus'], inv['shape'], inv['val'].shape
            #     if bus in log.log_store.keys():
            #         print "overriding bus", bus, "with log", log.log_store[bus].shape
            #         inv['val'] = log.log_store[bus].values.copy().T # reshape(inv['shape'])
            
            # for ink, inv in self.inputs.items():
            #     self.debug_print("[%s]step in[%s].shape = %s", (self.id, ink, inv['shape']))

            plots = self.plot_subplots()
            
            # set figure title and show the fig
            # self.fig.suptitle("%s: %s-%s" % (self.top.id, self.cname, self.id))
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
        # print "%s.save filename = %s, subplotstr = %s" % (plotinst.cname, filename, subplotstr)
        print "%s.save filename = %s" % (plotinst.cname, filename)
        plotinst.fig.set_size_inches((min(plotinst.fig_cols * 2 * 2.5, 20), min(plotinst.fig_rows * 1.2 * 2.5, 12)))
        try:
            plotinst.fig.savefig(filename, dpi=300, bbox_inches="tight")
        except Exception, e:
            print "%s error %s" % ('FigPlotBlock2', e)

class PlotBlock2(FigPlotBlock2):
    """PlotBlock2 class
    
    Block for plotting timeseries and histograms
    """
    # PlotBlock2.defaults
    defaults = {
        # 'inputs': {
        #     'x': {'bus': 'x'}, # FIXME: how can this be known? id-1?
        # },
        'blocksize': 1,
        'subplots': [[{'input': ['x'], 'plot': timeseries}]],
     }
    
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        """loop over configured subplot and plot the data according to config"""
        self.debug_print("%s plot_subplots self.inputs = %s",
                             (self.cname, self.inputs))
        if True:
            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    assert subplotconf.has_key('input'), "PlotBlock2 needs 'input' key in the plot spec = %s" % (subplotconf,)
                    # assert subplotconf.has_key('plot'), "PlotBlock2 needs 'plot' key in the plot spec = %s" % (subplotconf,)
                    # make it a list if it isn't
                    for input_spec_key in ['input', 'ndslice', 'shape']:
                        if subplotconf.has_key(input_spec_key):
                            subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                            # print "    id: %s, subplotconf[%s] = %s" % (self.id, input_spec_key, subplotconf[input_spec_key])

                    # subplot index from rows*cols
                    idx = (i*self.fig_cols)+j
                        
                    # self.debug_print("[%s]step idx = %d, conf = %s, data = %s/%s", (
                    #     self.id, idx,
                    #     subplotconf, subplotconf['input'], self.inputs[subplotconf['input']]))
                    # self.inputs[subplotconf['input']][0]))

                    # hier
                    
                    # plotdata = self.inputs[subplotconf['input']][0].T
                    # elif type(subplotconf['input']) is list:
                    # plotdata = self.inputs[subplotconf['input'][1]][0].T
                    # plotdata = {}
                    plotdata = OrderedDict()
                    plotvar = " "
                    title = ""
                    if subplotconf.has_key('title'): title += subplotconf['title']
                        
                    # default plot type
                    if not subplotconf.has_key('plot'): subplotconf['plot'] = timeseries

                    if type(subplotconf['plot']) is list:
                        subplotconf_plot = subplotconf['plot'][j]
                        assert subplotconf_plot is not type(str), "FIXME: plot callbacks is array of strings, eval strings"
                    elif type(subplotconf['plot']) is str:
                        gv = plotfuncs # {'timeseries': timeseries, 'histogram': histogram}
                        gv['partial'] = partial
                        lv = {}
                        code = compile("f_ = %s" % (subplotconf['plot'], ), "<string>", "exec")
                        # print "code", code
                        # subplotconf_plot = eval(code)
                        exec(code, gv, lv)
                        subplotconf['plot'] = lv['f_']
                        subplotconf_plot = lv['f_']
                        # subplotconf_plot = eval(subplotconf['plot'])
                        print "subplotconf_plot", subplotconf_plot
                        # subplotconf_plot = eval(subplotconf['plot'])
                    else:
                        subplotconf_plot = subplotconf['plot']
                        
                    if hasattr(subplotconf_plot, 'func_name'):
                        # plain function
                        plottype = subplotconf_plot.func_name
                    elif hasattr(subplotconf_plot, 'func'):
                        # partial'ized func
                        plottype = subplotconf_plot.func.func_name
                    else:
                        # unknown func type
                        plottype = timeseries # "unk plottype"

                    # append to title
                    title += " " + plottype
                        
                    for k, ink in enumerate(subplotconf['input']):
                        plotlen = self.inputs[subplotconf['input'][0]]['shape'][-1]
                        xslice = slice(0, plotlen)
                        plotshape = mytupleroll(self.inputs[subplotconf['input'][k]]['shape'])
                        # print "%s.subplots defaults: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                    
                        # x axis slice spec
                        if subplotconf.has_key('xslice'):
                            xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                            plotlen = xslice.stop - xslice.start
                            plotshape = (plotlen, ) + tuple((b for b in plotshape[1:]))
                        
                        # print "%s.subplots post-xslice: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                        
                        if subplotconf.has_key('shape'):
                            plotshape = mytupleroll(subplotconf['shape'][0])
                            plotlen = plotshape[0]
                            xslice = slice(0, plotlen)

                        # print "%s.subplots post-shape: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                        
                        # configure x axis
                        if subplotconf.has_key('xaxis'):
                            t = self.inputs[subplotconf['xaxis']]['val'].T[xslice]
                        else:
                            t = np.linspace(xslice.start, xslice.start+plotlen-1, plotlen)[xslice]
                    
                        
                        # print "%s.plot_subplots k = %s, ink = %s" % (self.cname, k, ink)
                        # plotdata[ink] = self.inputs[ink]['val'].T[xslice]
                        # if ink == 'd0':
                        #     print "plotblock2", self.inputs[ink]['val'].shape
                        #     print "plotblock2", self.inputs[ink]['val'][0,...,:]
                        ink_ = "%s_%d" % (ink, k)
                        # print "      input shape %s: %s" % (ink, self.inputs[ink]['val'].shape)
                        if subplotconf.has_key('ndslice'):
                            # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                            plotdata[ink_] = myt(self.inputs[ink]['val'])[subplotconf['ndslice'][k]]
                            # print "      ndslice %s: %s, numslice = %d" % (ink, subplotconf['ndslice'][k], len(subplotconf['ndslice']))
                        else:
                            plotdata[ink_] = myt(self.inputs[ink]['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))

                        assert plotdata[ink_].shape != (0,), "no data to plot"
                        # print "      id %s, ink = %s, plotdata = %s, plotshape = %s" % (self.id, ink_, plotdata[ink_].shape, plotshape)
                        # plotdata[ink_] = plotdata[ink_].reshape((plotshape[1], plotshape[0])).T
                        plotdata[ink_] = plotdata[ink_].reshape(plotshape)
                        
                        # fix nans
                        plotdata[ink_][np.isnan(plotdata[ink_])] = -1.0
                        plotvar += "%s, " % (self.inputs[ink]['bus'],)
                    # assign and trim title
                    title += plotvar[:-2]
                        
                    # different
                    if subplotconf.has_key('mode'):
                        # ivecs = tuple(self.inputs[ink][0].T[xslice] for k, ink in enumerate(subplotconf['input']))
                        ivecs = tuple(self.inputs[ink][0].myt()[xslice] for k, ink in enumerate(subplotconf['input']))
                        # for item in ivecs:
                        #     print "ivec.shape", item.shape
                        plotdata = {}
                        if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                            plotdata['all'] = np.hstack(ivecs)

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
                    self.fig.axes[idx].clear()
                    inkc = 0
                    for ink, inv in plotdata.items():
                        # print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s, t.sh = %s" % (self.cname, ink, plotvar, inv.shape, t.shape)
                        if type(subplotconf['plot']) is list:
                            plotfunc_conf = subplotconf['plot'][inkc]
                        else:
                            plotfunc_conf = subplotconf['plot']
                            
                        # this is the plotfunction from the config
                        plotfunc_conf(self.fig.axes[idx], data = inv, ordinate = t, label = "%s" % ink, title = title)
                        # labels.append("%s" % ink)
                        # metadata
                        inkc += 1
                    # self.fig.axes[idx].legend()
                    # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                    # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)
        plt.draw()
        plt.pause(1e-9)

# plot a matrix via imshow/pcolor
class ImgPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        self.debug_print("%s plot_subplots self.inputs = %s", (self.cname, self.inputs))
        # print "%s.plot_subplots(): all = %s" % (self.cname, self.inputs['all']['val'].shape)
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
                # check conditions
                assert subplotconf.has_key('shape'), "image plot needs shape spec"
                
                # make it a list if it isn't
                for input_spec_key in ['input', 'ndslice', 'shape']:
                    if subplotconf.has_key(input_spec_key):
                        subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                        
                # for img plot use only first input item
                subplotin = self.inputs[subplotconf['input'][0]]
                # print "subplotin[%d,%d].shape = %s / %s" % (i, j, subplotin['val'].shape, subplotin['shape'])
                vmins_sb[j].append(np.min(subplotin['val']))
                vmaxs_sb[j].append(np.max(subplotin['val']))
                extrema[0,i,j] = np.min(subplotin['val'])
                extrema[1,i,j] = np.max(subplotin['val'])
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

                    # map loop indices to gridspec linear index
                    idx = (i*self.fig_cols)+j
                    # print "self.inputs[subplotconf['input']][0].shape", self.inputs[subplotconf['input'][0]]['val'].shape, self.inputs[subplotconf['input'][0]]['shape']

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
                    
                    # print "%s plot_subplots self.inputs[subplotconf['input'][0]]['val'].shape = %s" % (self.cname, self.inputs[subplotconf['input'][0]]['val'].shape)
                    # old version
                    # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][yslice,xslice]

                    # FIXME completeness if input is ndim, currently only first dim is handled
                    if subplotconf.has_key('ndslice'):
                        # di = subplotconf['ndslice'][0]
                        # dj = subplotconf['ndslice'][1]
                        # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][di, dj, :, -1]
                        ink = subplotconf['input'][0]
                        plotdata_cand = myt(self.inputs[ink]['val'])[subplotconf['ndslice'][0]]
                        # print "%s[%d]-%s.step plotdata_cand.shape = %s, ndslice = %s, shape = %s, xslice = %s, yslice = %s" % (self.cname, self.cnt, self.id, plotdata_cand.shape, subplotconf['ndslice'], subplotconf['shape'], xslice, yslice)
                        # print "plotdata_cand", plotdata_cand
                    else:
                        try:
                            plotdata_cand = myt(self.inputs[subplotconf['input'][0]]['val'])[xslice,yslice]
                        except Exception, e:
                            print self.cname, self.id, self.cnt, self.inputs, subplotconf['input']
                            # print "%s[%d]-%s.step, inputs = %s, %s " % (self.cname, self.cnt, self.id, self.inputs[subplotconf['input']][0].shape)
                            print e
                    #                                         self.inputs[subplotconf['input']][0])
                    # print "plotdata_cand", plotdata_cand.shape

                    ################################################################################
                    # digitize a random sample (continuous arguments, continuous values)
                    # to an argument grid and average the values
                    # FIXME: to separate function
                    if subplotconf.has_key('digitize'):
                        argdims = subplotconf['digitize']['argdims']
                        numbins = subplotconf['digitize']['numbins']
                        valdims = subplotconf['digitize']['valdim']

                        # print "%s.plot_subplots(): digitize argdims = %s, numbins = %s, valdims = %s" % (self.cname, argdims, numbins, valdims)
                        
                        # plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims)
                        plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims, f_fval = np.mean)
                        
                    plotdata = {}

                    # if we're dimstacking, now is the time
                    if subplotconf.has_key('dimstack'):
                        plotdata['i_%d_%d' % (i, j)] = dimensional_stacking(plotdata_cand, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                        # print "plotdata[" + 'i_%d_%d' % (i, j) + "].shape", plotdata['i_%d_%d' % (i, j)].shape
                        # print "%s.plot_subplots(): dimstack x = %s, y = %s" % (self.cname, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                    else:
                        plotdata['i_%d_%d' % (i, j)] = plotdata_cand.reshape(subplotconf['shape'][0])
                    if subplotconf.has_key('ylog'):
                        # plotdata['i_%d_%d' % (i, j)] = np.log(plotdata['i_%d_%d' % (i, j)] + 1.0)
                        # print plotdata['i_%d_%d' % (i, j)]
                        yscale = 'log'
                    else:
                        yscale = 'linear'
                    plotvar = self.inputs[subplotconf['input'][0]]['bus']

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
                        # FIXME: put the image plotting code into function
                        ax = self.fig.axes[idx]
                        
                        inv[np.isnan(inv)] = -1.0

                        # Linv = np.log(inv + 1)
                        Linv = inv
                        # print "Linv.shape", Linv.shape
                        # print "Linv", np.sum(np.abs(Linv))
                        plotfunc = "pcolorfast"
                        plot_img(ax = ax, data = Linv, plotfunc = plotfunc,
                                     vmin = vmin, vmax = vmax, cmap = cmap,
                                     title = title)
        # update
        plt.draw()
        plt.pause(1e-9)
                        

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
        # print "%s.step inputs: %s"  % (self.cname, self.inputs.keys())

        subplotconf = self.subplots[0][0]
        
        # different 
        if subplotconf.has_key('mode'):
            ivecs = tuple(self.inputs[ink]['val'].T for k, ink in enumerate(subplotconf['input']))
            # for ivec in ivecs:
            #     print "ivec.shape", ivec.shape
            plotdata = {}
            if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                plotdata['all'] = np.hstack(ivecs)

        data = plotdata['all']

        # print "SnsPlotBlock2:", data.shape
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
        
                    
