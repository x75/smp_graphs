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

# perceptually uniform colormaps
import colorcet as cc

from smp_graphs.block import decStep, decInit, block_cmaps
from smp_graphs.block import PrimBlock2
from smp_graphs.utils import myt, mytupleroll
import smp_graphs.logging as log

from smp_base.plot_utils import put_legend_out_right, put_legend_out_top
from smp_base.dimstack   import dimensional_stacking, digitize_pointcloud
from smp_base.plot       import makefig, timeseries, histogram, plot_img, plotfuncs, uniform_divergence
from smp_base.plot       import get_colorcycler, kwargs_plot_clean

################################################################################
# Plotting blocks
# pull stuff from
# smp/im/im_quadrotor_plot
# ...

# FIXME: do some clean up here
#  - unify subplot spec and options handling
#  - clarify preprocessing inside / outside plotblock
#  - general matrix / systematic combinations plotting for n-dimensional data
#   - from scatter_matrix to modality-timedelay matrix
#   - modality-timedelay matrix is: modalities on x, timedelay on y
#   - modality-timedelay matrix is: different dependency measures xcorr, expansion-xcorr, mi, rp, kldiv, ...
#   - information decomposition matrix (ica?)
# 
rcParams['figure.titlesize'] = 11

# smp_graphs style
rcParams['axes.grid'] = False
rcParams['axes.spines.bottom'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.spines.left'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.facecolor'] = 'none'
# rcParams['axes.labelcolor'] = .15
# rcParams['axes.labelpad'] = 4.0
rcParams['axes.titlesize'] = 10.0
rcParams['axes.labelsize'] = 8.0
rcParams['axes.labelweight'] = 'normal'
rcParams['legend.framealpha'] = 0.3
rcParams['legend.fontsize'] = 9.0
rcParams['legend.labelspacing'] = 0.5
rcParams['xtick.labelsize'] = 8.0
rcParams['ytick.labelsize'] = 8.0

# f = open("rcparams.txt", "w")
# f.write("rcParams = %s" % (rcParams, ))
# f.close()

def subplot_input_fix(input_spec):
    # assert input an array 
    if type(input_spec) is str or type(input_spec) is tuple:
        return [input_spec]
    else:
        return input_spec


class AnalysisBlock2(PrimBlock2):
    defaults = {
        'nocache': True,
        'saveplot': False,
        'savetype': 'jpg',
        'block_group': 'measure',
        'desc': 'Some kind of analysis'
        }
    def __init__(self, conf = {}, paren = None, top = None):
        # use inputs from logfile even in no-cached epxeriment
        self.inputs_log = None
        # saving plots
        self.saveplot = False
        self.savetype = "jpg"

        defaults = {}
        # defaults.update(Block2.defaults)
        defaults.update(PrimBlock2.defaults, **self.defaults)
        self.defaults = defaults
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        # print "AnalysisBlock2.init", conf['params']['saveplot'], self.conf['params']['saveplot']
        # print "AnalysisBlock2.init saveplot =", self.saveplot

        # default title?
        if not hasattr(self, 'title'):
            # self.title = "%s - %s-%s" % (self.top.id, self.cname, self.id)
            # self.title = "%s of %s" % (self.cname, self.top.id[:20], )
            self.title = "%s of %s, numsteps = %d" % (self.id, self.top.id, self.top.numsteps,)
        
    def save(self):
        """Save the analysis, redirect to corresponding class method, passing the instance
        """
        if isinstance(self, FigPlotBlock2) or isinstance(self, SnsMatrixPlotBlock2):
            FigPlotBlock2.savefig(self)

    def check_plot_type(self, conf, defaults = {}):
        """Get 'plot' configuration item and make sure it is a list of function pointers

        Returns:
         - list of plotfunction pointers
        """
        defaults.update(conf)
        conf = defaults
        # print "conf", conf        
        if type(conf['plot']) is list:
            # check str or func for each element
            conf_plot = [self.check_plot_type_single(f) for f in conf['plot']]
            # conf_plot = conf['plot'] # [j]
            # assert conf_plot is not type(str), "FIXME: plot callbacks is array of strings, eval strings"
        elif type(conf['plot']) is str:
            conf_plot = self.eval_conf_str(conf['plot'])
            if type(conf_plot) is list:
                conf_plot = self.check_plot_type(conf, defaults)
            else:
                conf_plot = [conf_plot]
        else:
            conf_plot = [conf['plot']]
        return conf_plot
        
    def eval_conf_str(self, confstr):
        gv = plotfuncs # {'timeseries': timeseries, 'histogram': histogram}
        gv['partial'] = partial
        lv = {}
        code = compile("f_ = %s" % (confstr, ), "<string>", "exec")
        # print "code", code
        # conf_plot = eval(code)
        exec(code, gv, lv)
        # conf['plot'] = lv['f_']
        conf_plot = lv['f_']
        # conf_plot = eval(conf['plot'])
        # print "conf_plot", conf_plot
        # conf_plot = eval(conf['plot'])
        return conf_plot
        
    def check_plot_type_single(self, f):
        """Get and if necessary type-fix the 'plot' subplot configuration param, translating from string to func.

        Returns:
         - single function pointer
        """
        # default plot type
        # if not conf.has_key('plot'): conf['plot'] = timeseries
        # print "defaults", defaults
        # defaults.update(conf)
        # conf = defaults
        # print "conf", conf

        if type(f) is str:
            return self.eval_conf_str(f)
        else:
            return f

    def get_title_from_plot_type(self, plotfunc_conf):
        title = ""
        for plotfunc in plotfunc_conf:
            # get the plot type from the plotfunc type
            if hasattr(plotfunc, 'func_name'):
                # plain function
                plottype = plotfunc.func_name
            elif hasattr(plotfunc, 'func'):
                # partial'ized func
                plottype = plotfunc.func.func_name
            else:
                # unknown func type
                plottype = timeseries # "unk plottype"

            # append plot type to title
            title += " " + plottype
        return title
        

class BaseplotBlock2(AnalysisBlock2):
    """Plotting base class
    
    Common features for all plots.

    Variants:
     - :class:`FigPlotBlock2' is :mod:`matplotlib` figure based plot block
     - :class:`SnsMatrixPlotBlock2` is a :mod:`seaborn` based plot which do not cooperate with external figure handles

    Plot block_group is both measure *and* output [wins]
    """
    defaults = {
        'block_group': ['output', 'measure'],
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # update child class 'self' defaults
        # self.defaults.update(BaseplotBlock2.defaults)
        defaults = {}
        # defaults.update(Block2.defaults)
        defaults.update(AnalysisBlock2.defaults, **self.defaults)
        self.defaults = defaults
        # super init
        AnalysisBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def prepare_saveplot(self):
        """if saveplot set, compute filename and register top.outputs of type fig
        """
        if self.saveplot:
            self.filename = '%s_%s.%s' % (self.top.datafile_expr, self.id, self.savetype)
            self.top.outputs['%s' % (self.id, )] = {
                'type': 'fig',
                'filename': self.filename,
                'label': self.top.id,
                'id': self.id,
                'desc': self.desc,
                'width': 1.0,
            }
    
class FigPlotBlock2(BaseplotBlock2):
    """FigPlotBlock2 class

    PlotBlock base class for matplotlib figure-based plots. Creates
    the figure and a gridspec on init, uses fig.axes in the step
    function
    
    Args:
     - blocksize(int): the blocksize
     - subplots(list): an array of arrays, each cell of that matrix contains one subplot configuration dict
     - subplotconf(dict): dict with entries *inputs*, a list of input keys, *plot*, a plot function pointer
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        self.wspace = 0.0
        self.hspace = 0.0
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # configure figure and plot axes
        self.fig_rows = len(self.subplots)
        self.fig_cols = len(self.subplots[0])

        # create figure
        self.fig = makefig(
            rows = self.fig_rows, cols = self.fig_cols,
            wspace = self.wspace, hspace = self.hspace,
            title = self.title)
        # self.fig.tight_layout(pad = 1.0)
        # self.debug_print("fig.axes = %s", (self.fig.axes, ))

        self.prepare_saveplot()
        
        # FIXME: too special
        self.isprimitive = False
        
    @staticmethod
    def savefig(plotinst):
        """Save the figure 'fig' using configurable options

        Args:
         - plotinst(BaseplotBlock2): a plot block instance

        Returns:
         - None
        """
        subplotstr = "_".join(np.array([["r%d_c%d_%s" % (r, c, "_".join(subplot_input_fix(sbc['input'])),) for c,sbc in enumerate(sbr)] for r, sbr in enumerate(plotinst.subplots)]).flatten())
        # filename = "data/%s_%s_%s_%s.%s" % (plotinst.top.id, plotinst.id, "_".join(plotinst.inputs.keys()), subplotstr, plotinst.savetype)
        # filename = "data/%s_%s_%s.%s" % (plotinst.top.id, plotinst.id, "_".join(plotinst.inputs.keys()), plotinst.savetype)
        # filename = "data/%s_%s.%s" % (plotinst.top.id, plotinst.id, plotinst.savetype)
        # filename = '%s_%s.%s' % (plotinst.top.datafile_expr, plotinst.id, plotinst.savetype)
        filename = plotinst.filename
        # print "%s.save filename = %s, subplotstr = %s" % (plotinst.cname, filename, subplotstr)
        # plotinst.fig.set_size_inches((min(plotinst.fig_cols * 2 * 2.5, 20), min(plotinst.fig_rows * 1.2 * 2.5, 12)))
        if not hasattr(plotinst, 'savesize'):
            savescale = 3
            plotinst.savesize = (
                min(plotinst.fig_cols * 2.5 * savescale, 24),
                min(plotinst.fig_rows * 1.0 * savescale, 12))
            
        print "savesize w/h = %f/%f, fig_cols/fig_rows = %s/%s" % (plotinst.savesize[0], plotinst.savesize[1], plotinst.fig_cols, plotinst.fig_rows)
        plotinst.fig.set_size_inches(plotinst.savesize)

        # write the figure to file
        try:
            print "%s-%s.save saving plot %s to filename = %s" % (plotinst.cname, plotinst.id, plotinst.title, filename)
            plotinst.fig.savefig(filename, dpi=300, bbox_inches="tight")
            # if plotinst.top.
            plotinst.top.outputs['latex']['figures'][plotinst.id] = {
                'filename': filename,
                'label': plotinst.top.id,
                'id': plotinst.id,
                'desc': plotinst.desc}
            # plotinst.fig.savefig(filename, dpi=300)
        except Exception, e:
            print "%s.save saving failed with %s" % ('FigPlotBlock2', e)
            
    @decStep()
    def step(self, x = None):
        """Call the :func:`plot_subplots` function

        Makes sure
         - that there is some data to plot
         - that the data is loaded from the :data:`log_store` instead
           of the :class:`Bus` inputs if the :data:`inputs_log` is
           set.
        """
        
        # have inputs at all?
        if len(self.inputs) < 1: return

        # make sure that data has been generated
        if (self.cnt % self.blocksize) in self.blockphase: # or (xself.cnt % xself.rate) == 0:

            # HACK: override block inputs with log.log_store
            if self.inputs_log is not None:
                print "Using inputs from log.log_store = %s with keys = %s instead of bus" % (log.log_store.filename, log.log_store.keys(), )
                # commit data
                log.log_pd_store()
                # iterate input items
                for ink, inv in self.inputs.items():
                    bus = '/%s' % (inv['bus'], )
                    # print "ink", ink, "inv", inv['bus'], inv['shape'], inv['val'].shape
                    # check if a log exists
                    if bus in log.log_store.keys():
                        # print "overriding bus", bus, "with log", log.log_store[bus].shape
                        # copy log data to input value
                        inv['val'] = log.log_store[bus].values.copy().T # reshape(inv['shape'])
            
            # for ink, inv in self.inputs.items():
            #     self.debug_print("[%s]step in[%s].shape = %s", (self.id, ink, inv['shape']))

            # run the plots
            plots = self.plot_subplots()
            
            # set figure title and show the fig
            # self.fig.suptitle("%s: %s-%s" % (self.top.id, self.cname, self.id))
            self.fig.show()

            # if self.saveplot:
            #     self.save_fig()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))

    def plot_subplots(self):
        """FigPlotBlock2.plot_subplots

        This is a stub and has to be implement by children classes.
        """
        print "%s-%s.plot_subplots(): implement me" % (self.cname, self.id,)

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
        'xlim_share': True,
        'ylim_share': True,
        'subplots': [[{'input': ['x'], 'plot': timeseries}]],
     }
    
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)
         
    def plot_subplots(self):
        """loop over configured subplots and plot the data according to the configuration

        The function does not take any arguments. Instead, the args
        are taken from the :data:`subplots` member.

        subplots is a list of lists, specifying a the subplot
        grid. `subplots[:]` are rows and `subplots[:][:]` are the
        columns.

        Each subplot entry is a dictionary with the following keys:
         - input: list of block.inputs label keys
         - plot: function pointer for a plotting function like
           :func:`timeseries`, :func:`histogram`, ...
         - ndslice: a multidimensional slice selecting data from tensor input
         - shape: the shape of the data after nd-slicing
         - xslice: just the x-axis slice, usually time

        Arguments:
         - None

        Returns:
         - None
        """
        self.debug_print("%s plot_subplots self.inputs = %s",
                             (self.cname, self.inputs))

        # subplots pass 0: remember ax limits
        sb_rows = len(self.subplots)
        sb_cols = len(self.subplots[0])
        
        rows_ylim_max = [(1e9, -1e9) for _ in range(sb_rows)]
        cols_xlim_max = [(1e9, -1e9) for _ in range(sb_cols)]

        # default plot size
        self.fig.set_size_inches((sb_cols * 6, sb_rows * 3))
        
        # subplots pass 1: the hard work, iterate over subplot config and build the plot
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

                # get this subplot's plotfunc configuration and make sure its a list
                plotfunc_conf = self.check_plot_type(subplotconf)
                # print "%s-%s plotfunc_conf = %s" % (self.cname, self.id, plotfunc_conf)
                assert type(plotfunc_conf) is list, "plotfunc_conf must be a list, not %s" % (type(plotfunc_conf), )

                if title == '':
                    title += self.get_title_from_plot_type(plotfunc_conf)

                # loop over inputs
                for k, ink in enumerate(subplotconf['input']):
                    # FIXME: array'ize this loop
                    # vars: input, ndslice, shape, xslice, ...

                    # get numsteps of data for the input
                    plotlen = self.inputs[ink]['shape'][-1] # numsteps at shape[-1]

                    # set default slice
                    xslice = slice(0, plotlen)
                    # compute final shape of plot data, custom transpose from horiz time to row time
                    plotshape = mytupleroll(self.inputs[ink]['shape'])
                    
                    # print "%s.subplots defaults: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                
                    # x axis slice spec
                    if subplotconf.has_key('xslice'):
                        # get slice conf
                        if type(subplotconf['xslice']) is list:
                            subplotconf_xslice = subplotconf['xslice'][k]
                        else:
                            subplotconf_xslice = subplotconf['xslice']
                        # set slice
                        xslice = slice(subplotconf_xslice[0], subplotconf_xslice[1])
                        # update plot length
                        plotlen = xslice.stop - xslice.start
                        # and plot shape
                        plotshape = (plotlen, ) + tuple((b for b in plotshape[1:]))
                    
                    # print "%s.subplots post-xslice: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)

                    # explicit shape key
                    if subplotconf.has_key('shape'):
                        if len(subplotconf['shape']) > 1:
                            subplotconf_shape = subplotconf['shape'][k]
                        else:
                            subplotconf_shape = subplotconf['shape'][0]
                        # get the shape spec, custom transpose from horiz t to row t
                        plotshape = mytupleroll(subplotconf_shape)
                        # update plot length
                        plotlen = plotshape[0]
                        # and xsclice
                        xslice = slice(0, plotlen)

                    # print "%s.subplots post-shape: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                    
                    # configure x axis, default implicit number of steps
                    if subplotconf.has_key('xaxis'):
                        t = self.inputs[subplotconf['xaxis']]['val'].T[xslice]
                    else:
                        t = np.linspace(xslice.start, xslice.start+plotlen-1, plotlen)[xslice]
                    
                    # print "%s.plot_subplots k = %s, ink = %s" % (self.cname, k, ink)
                    # plotdata[ink] = self.inputs[ink]['val'].T[xslice]
                    # if ink == 'd0':
                    #     print "plotblock2", self.inputs[ink]['val'].shape
                    #     print "plotblock2", self.inputs[ink]['val'][0,...,:]
                    # ink_ = "%s_%d" % (ink, k)
                    ink_ = "%d_%s" % (k + 1, ink)
                    # print "      input shape %s: %s" % (ink, self.inputs[ink]['val'].shape)

                    # if explicit n-dimensional slice is given
                    if subplotconf.has_key('ndslice'):
                        # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                        # slice the data to spec, custom transpose from h to v time
                        ndslice = subplotconf['ndslice'][k]
                        # print "k", k, "ink", ink, "ndslice", ndslice
                        plotdata[ink_] = myt(self.inputs[ink]['val'])[ndslice]
                        # print "      ndslice %s: %s, numslice = %d" % (ink, subplotconf['ndslice'][k], len(subplotconf['ndslice']))
                    else:
                        plotdata[ink_] = myt(self.inputs[ink]['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))

                    assert plotdata[ink_].shape != (0,), "no data to plot"
                    # print "      input = %s" % self.inputs[ink]['val']
                    # print "      id %s, ink = %s, plotdata = %s, plotshape = %s" % (self.id, ink_, plotdata[ink_], plotshape)
                    # plotdata[ink_] = plotdata[ink_].reshape((plotshape[1], plotshape[0])).T
                    plotdata[ink_] = plotdata[ink_].reshape(plotshape)
                    
                    # fix nans
                    plotdata[ink_][np.isnan(plotdata[ink_])] = -1.0
                    plotvar += "%s, " % (self.inputs[ink]['bus'],)
                    
                # # assign and trim title
                # title += plotvar[:-2]
                    
                # combine inputs into one backend plot call to automate color cycling etc
                if subplotconf.has_key('mode'):
                    """FIXME: fix dangling effects of stacking"""
                    # ivecs = tuple(myt(self.inputs[ink]['val'])[xslice] for k, ink in enumerate(subplotconf['input']))
                    ivecs = [plotdatav for plotdatak, plotdatav in plotdata.items()]
                    # plotdata = {}
                    if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                        plotdata['_stacked'] = np.hstack(ivecs)

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
                # kwargs_ = {} # kwargs_plot_clean(**kwargs)
                kwargs = {}
                for k in ['xlabel', 'ylabel']:
                    if subplotconf.has_key(k):
                        kwargs[k] = subplotconf[k]
                        
                labels = []
                self.fig.axes[idx].clear()
                inkc = 0

                num_cgroups = 5
                num_cgroup_color = 5
                num_cgroup_dist = 255/num_cgroups
                # cmap_str = 'cyclic_mrybm_35_75_c68'
                # cmap_str = 'colorwheel'
                cmap_str = 'rainbow'
                
                # axis handle shortcut
                ax = self.fig.axes[idx]
                ax.set_prop_cycle(
                    get_colorcycler(
                        cmap_str = cmap_str, cmap_idx = None,
                        c_s = inkc * num_cgroup_dist, c_e = (inkc + 1) * num_cgroup_dist, c_n = num_cgroup_color
                    )
                )

                # stacked data?
                if plotdata.has_key('_stacked'):
                    print "block_plot.py plotting stacked"
                    plotfunc_conf[0](ax, data = plotdata['_stacked'], ordinate = t, title = title, **kwargs)
                
                # iterate over plotdata items
                for ink, inv in plotdata.items():
                    # print "%s.plot_subplots: ink = %s, plotvar = %s, inv.sh = %s, t.sh = %s" % (self.cname, ink, plotvar, inv.shape, t.shape)

                    # if multiple input groups, increment color group
                    if inkc > 0:
                        ax.set_prop_cycle(
                            get_colorcycler(
                                cmap_str = cmap_str, cmap_idx = None,
                                c_s = (inkc + 1) * num_cgroup_dist, c_e = (inkc + 2) * num_cgroup_dist, c_n = num_cgroup_color
                            ),
                        )

                    # select single element at first slot or increment index with plotdata items
                    plotfunc_idx = inkc % len(plotfunc_conf)
                    
                    # this is the plot function array from the config
                    if not plotdata.has_key('_stacked'):
                        # print "    plot_subplots plotfunc", plotfunc_conf[plotfunc_idx]
                        # print "                      args", ax, inv, t, title, kwargs
                        plotfunc_conf[plotfunc_idx](ax = ax, data = inv, ordinate = t, title = title, **kwargs)

                    # label = "%s" % ink, title = title
                    # tmp_cmaps_ = [k for k in cc.cm.keys() if 'cyclic' in k and not 'grey' in k]
                        
                    # ax.set_prop_cycle(get_colorcycler(cmap_str = tmp_cmaps_[inkc]))
                    for invd in range(inv.shape[1]):
                        label_ = "%s-%d" % (ink, invd + 1)
                        if len(label_) > 16:
                            label_ = label_[:16]
                        labels.append(label_)
                    # metadata
                    inkc += 1
                    
                # store the final plot data
                # print "sbdict", self.subplots[i][j]
                self.subplots[i][j]['p1_plottitle'] = title
                self.subplots[i][j]['p1_plotdata'] = plotdata
                self.subplots[i][j]['p1_plotvar'] = plotvar
                self.subplots[i][j]['p1_plotlabels'] = labels
                self.subplots[i][j]['p1_plotxlim'] = ax.get_xlim()
                self.subplots[i][j]['p1_plotylim'] = ax.get_ylim()
                    
                # save axis limits
                # print "xlim", ax.get_xlim()
                sb = self.subplots[i][j]
                if sb['p1_plotxlim'][0] < cols_xlim_max[j][0]: cols_xlim_max[j] = (sb['p1_plotxlim'][0], cols_xlim_max[j][1])
                if sb['p1_plotxlim'][1] > cols_xlim_max[j][1]: cols_xlim_max[j] = (cols_xlim_max[j][0], sb['p1_plotxlim'][1])
                if sb['p1_plotylim'][0] < rows_ylim_max[i][0]: rows_ylim_max[i] = (sb['p1_plotylim'][0], rows_ylim_max[i][1])
                if sb['p1_plotylim'][1] > rows_ylim_max[i][1]: rows_ylim_max[i] = (rows_ylim_max[i][0], sb['p1_plotylim'][1])
 
                # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)
        # subplots pass 1: done

        ################################################################################
        # subplots pass 2: clean up and compute globally shared dynamic vars
        # adjust xaxis
        for i, subplot in enumerate(self.subplots):
            for j, subplotconf in enumerate(subplot):
                # subplot handle shortcut
                sb = self.subplots[i][j]
                # subplot index from rows*cols
                idx = (i*self.fig_cols)+j
                # axis handle shortcut
                ax = self.fig.axes[idx]

                # consolidate axis limits
                if self.xlim_share:
                    ax.set_xlim(cols_xlim_max[j])
                if self.ylim_share:
                    ax.set_ylim(rows_ylim_max[i])
                
                # fix legends
                # ax.legend(labels)
                put_legend_out_right(
                    labels = sb['p1_plotlabels'],
                    ax = ax, resize_by = 0.9)
                # put_legend_out_top(labels = labels, ax = ax, resize_by = 0.8)

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
# from input vector of size n: set of vector elements, set of all undirected pairs
# from input time offsets: set of all time offset for a pair
# matrix of binary measure for all pairs (rows) for all time offsets (cols)

# compute delay bank (block)
# compute full variable stack vector (block)

# compute measure
# compute image0d
#  - avg MI
# compute image2d
#  - hexbin(x,y) or hexbin(x,y,C)
#  - hist2d
#  - scatter2d
#  - recurrence_plot_2d
#  - dimstack(imagend)
# compute imagend
#  - histnd
#  - scatternd

# plot stack of images: single array, list of arrays, list of lists of array
#  - subplotgrid
#  - imagegrid


class MatrixPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        pass

################################################################################
# non FigPlot plot blocks
class SnsMatrixPlotBlock2(BaseplotBlock2):
    """SnsMatrixPlotBlock2 class

    Plot block for seaborn pairwaise matrix plots: e.g. scatter, hexbin, ...

    Seaborne (stubornly) manages figures itself, so it can't be a FigPlotBlock2
    
    Arguments:
    - blocksize: usually numsteps (meaning plot all data created by that episode/experiment)
    - f_plot_diag: diagonal cells
    - f_plot_matrix: off diagonal cells
    - numpy matrix of data, plot iterates over all pairs with given function
    """
    
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # self.saveplot = False
        # self.savetype = 'jpg'
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.prepare_saveplot()
        
    @decStep()
    def step(self, x = None):
        # print "%s.step inputs: %s"  % (self.cname, self.inputs.keys())

        subplotconf = self.subplots[0][0]
        
        # vector combination
        if not subplotconf.has_key('mode'):
            subplotconf['mode'] = 'stack'

        # plotting func
        subplotconf_plot = self.check_plot_type(subplotconf, defaults = {'plot': plt.hexbin})
        
        # if not subplotconf.has_key('plot'):
        #     subplotconf['plot'] = plt.hexbin
            
        # ilbls = [[['%s%d' % (self.inputs[ink]['bus'], j)] for j in range(self.inputs[ink]['shape'][0])] for i, ink in enumerate(subplotconf['input'])]
        # print "ilbls", ilbls
        # ivecs = tuple(self.inputs[ink]['val'].T for k, ink in enumerate(subplotconf['input']))
        
        def unwrap_ndslice(self, subplotconf, k, ink):
            # default x-axis slice
            xslice = slice(None)
            # apply ndslice
            if subplotconf.has_key('ndslice'):
                # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                print "      ndslice %s: %s, numslice = %d" % (ink, subplotconf['ndslice'][k], len(subplotconf['ndslice']))
                plotdata = myt(self.inputs[ink]['val'])
                print "      ndslice plotdata", plotdata.shape
                plotdata = plotdata[subplotconf['ndslice'][k]]
                print "      ndslice plotdata", plotdata.shape
            else:
                plotdata = myt(self.inputs[ink]['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))
            print "       ndslice plotdata", plotdata.shape
            
            # apply shape
            if subplotconf.has_key('shape'):
                if type(subplotconf['shape']) is list:
                    plotdata_shape = subplotconf['shape'][k]
                else:
                    plotdata_shape = subplotconf['shape']
                print "       ndslice plotshape", plotdata_shape
            else:
                plotdata_shape = plotdata.T.shape

            plotdata = myt(plotdata).reshape(plotdata_shape)
            print "        shape plotdata", plotdata.shape
    
            return plotdata    
            
        ivecs = []
        ilbls = []
        for k, ink in enumerate(subplotconf['input']):
            # ivec = myt(self.inputs[ink]['val'])
            ivec = unwrap_ndslice(self, subplotconf, k, ink)
            ivecs.append(ivec)
            ilbls += ['%s%d' % (self.inputs[ink]['bus'], j) for j in range(ivec.shape[0])] # range(self.inputs[ink]['shape'][0])]
            # ilbls.append(ilbl)
        print "ilbls", ilbls
        
        # ivecs = tuple(myt(self.inputs[ink]['val']) for k, ink in enumerate(subplotconf['input']))
        # for ivec in ivecs:
        #     print "ivec.shape", ivec.shape
        plotdata = {}
        if subplotconf['mode'] in ['stack', 'combine', 'concat']:
            # plotdata['all'] = np.hstack(ivecs)
            plotdata['all'] = np.vstack(ivecs).T

        data = plotdata['all']
        print "data", data
        
        print "SnsPlotBlock2:", data.shape
        scatter_data_raw  = data
        # scatter_data_cols = ["x_%d" % (i,) for i in range(data.shape[1])]
        
        scatter_data_cols = np.array(ilbls).flatten().tolist()

        # prepare dataframe
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
        
        g = sns.PairGrid(df)
        # ud_cmap = cc.cm['diverging_cwm_80_100_c22'] # rainbow
        histcolorcycler = get_colorcycler('isoluminant_cgo_70_c39')
        # print histcolorcycler
        histcolor = histcolorcycler.by_key()['color'][:df.shape[1]]
        # print histcolor
        
        # # rcParams['axes.prop_cycle'] = histcolorcycler
        # g.map_diag(plt.hist, histtype = 'step')
        # for i in range(df.shape[1]):
        #     ax_diag = g.axes[i,i]
        #     # print type(ax_diag), dir(ax_diag)
        #     ax_diag.grid()
        #     ax_diag.set_prop_cycle(histcolorcycler)
            
        # g.map_diag(sns.kdeplot)
        # g.map_offdiag(plt.hexbin, cmap="gray", gridsize=40, bins="log");
        # g.map_offdiag(plt.histogram2d, cmap="gray", bins=30)
        # g.map_offdiag(plt.plot, linestyle = "None", marker = "o", alpha = 0.5) # , bins="log");
        plotf = partial(uniform_divergence, f = subplotconf_plot)
        g.map_diag(plotf)
        # g = g.map_diag(sns.kdeplot, lw=3, legend=False)
        g.map_offdiag(plotf) #, cmap="gray", gridsize=40, bins='log')

        # clean up figure
        self.fig = g.fig
        self.fig_rows, self.fig_cols = g.axes.shape
        self.fig.suptitle(self.title)
        # print "dir(g)", dir(g)
        # print g.diag_axes
        # print g.axes
        # if self.saveplot:
        #     FigPlotBlock2.save(self)
        # for i in range(data.shape[1]):
        #     for j in range(data.shape[1]): # 1, 2; 0, 2; 0, 1
        #         if i == j:
        #             continue
        #         # column gives x axis, row gives y axis, thus need to reverse the selection for plotting goal
        #         # g.axes[i,j].plot(df["%s%d" % (self.cols_goal_base, j)], df["%s%d" % (self.cols_goal_base, i)], "ro", alpha=0.5)
        #         g.axes[i,j].plot(df["x_%d" % (j,)], df["x_%d" % (i,)], "ro", alpha=0.5)

        # plt.show()
        
                    
