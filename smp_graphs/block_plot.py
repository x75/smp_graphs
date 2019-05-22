"""smp_graphs.block_plot

Plotting blocks

.. moduleauthor:: Oswald Berthold, 2017

Available blocks: :class:`AnalysisBlock2`, :class:`BaseplotBlock2`,
:class:`FigPlotBlock2`, :class:`PlotBlock2`, :class:`ImgPlotBlock2`,
:class:`MatrixPlotBlock2`, :class:`SnsMatrixPlotBlock2`

The two main plot blocks are PlotBlock2 and ImgPlotBlock2. Each is
configured with a set of figure parameters and with a 2D array of
subplot configurations.

Figure params:
 - figtitle (title)
 - figsize (plotsize)
 - figbbox_inches ('tight')
 - fontsizes: figure.titlesize
 - hor. and vert. spacing
 - padding?

Subplot params:
 - subplot title loc and fontsize
 - tick loc and fontsize
 - label loc and fontsize
 - legend loc, fontsize
 - plotfunc
 - slicing / selecting
"""
import re, time, inspect
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

from smp_base.common     import get_module_logger
from smp_base.plot_utils import custom_legend, put_legend_out_right, put_legend_out_top, set_latex_header
from smp_base.dimstack   import dimensional_stacking, digitize_pointcloud
from smp_base.plot       import makefig, timeseries, histogram, plot_img, plotfuncs, uniform_divergence
from smp_base.plot       import get_colorcycler, fig_interaction
from smp_base.plot       import ax_invert, ax_set_aspect

from smp_base.codeops import code_compile_and_run
from smp_graphs.common import listify, tuple2inttuple
from smp_graphs.block import decStep, decInit, block_cmaps, get_input
from smp_graphs.block import PrimBlock2
from smp_graphs.utils import myt, mytupleroll
import smp_graphs.utils_logging as log
from functools import reduce

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

rcParams['figure.titlesize'] = 11 # suptitle

axes_spines = False
# smp_graphs style
rcParams['axes.grid'] = False
rcParams['axes.spines.bottom'] = axes_spines
rcParams['axes.spines.top'] = axes_spines
rcParams['axes.spines.left'] = axes_spines
rcParams['axes.spines.right'] = axes_spines
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
rcParams['xtick.direction'] = 'in'
rcParams['ytick.labelsize'] = 8.0
rcParams['ytick.direction'] = 'out'
# subplots
rcParams['figure.subplot.bottom'] = 0.12 # 0.11
rcParams['figure.subplot.left'] = 0.1 # 0.125
rcParams['figure.subplot.right'] = 0.9
rcParams['figure.subplot.top'] = 0.88 # 0.88

# rcParams['text.usetex'] = True
# rcParams["text.latex.preamble"] = [ r'\usepackage{amsmath}',
#     r'\usepackage{amsfonts}', r'\usepackage{amssymb}',
#     r'\usepackage{latexsym}', r'\usepackage{bm}']

# set_latex_header()

# f = open("rcparams.txt", "w")
# f.write("rcParams = %s" % (rcParams, ))
# f.close()

from logging import DEBUG as logging_DEBUG
import logging
logger = get_module_logger(modulename = 'block_plot', loglevel = logging_DEBUG)

def subplot_input_fix(input_spec):
    """subplot_input_fix

    Subplot configuration convenience function.
    
    Convert subplot configuration items into a list if they are
    singular types like numbers, strs and tuples.

    See also: smp_graphs.common.listify (FIXME: merge)
    """
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
        'desc': 'Some kind of analysis',
        'inputs_log': None, # True,
        }
    def __init__(self, conf = {}, paren = None, top = None):
        # use inputs from logfile even in no-cached epxeriment
        # self.inputs_log = None
        # saving plots
        # self.saveplot = False
        # self.savetype = "jpg"

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
            # self.title = "%s %s\nnumsteps = %d, caching = %s" % (self.id, self.top.id, self.top.numsteps, self.top.docache)
            self.title = "%s %s numsteps = %d" % (self.top.id, self.id, self.top.numsteps)
        
    def save(self):
        """Save the analysis, redirect to corresponding class method, passing the instance
        """
        if isinstance(self, FigPlotBlock2) or isinstance(self, SnsMatrixPlotBlock2):
            FigPlotBlock2.savefig(self)

    def check_plot_input(self, ink, args):
        """check_plot_input

        Input sanity check that requested item 'ink' exists.
        """
        i, j, k = args[:3]
        if ink not in self.inputs:
            # self._debug("    triggered: bus[%s] = %s, buskeys = %s" % (buskey, xself.bus[v['buskey']], bus.keys()))
            self._warning('plot_subplot pass 1 subplotconf[%d,%d] input[%d] = %s doesn\'t exist in self.inputs %s' % (
                i, j, k, ink, list(self.inputs.keys())))
            return False
        return self.inputs[ink]

    def check_plot_type(self, conf, defaults = {}):
        """check_plot_type

        Subplot plotfunc configuration type fix function part 1: raw conf

        Get subplot configuration item 'plot' and make sure it is a
        list of function pointers

        Returns:
         - list of plotfunc pointers
        """
        # merge defaults with conf
        defaults.update(conf)
        conf = defaults

        # check 'plot' type
        if 'plot' not in conf:
            conf_plot = [self.check_plot_type_single('timeseries')]
        elif type(conf['plot']) is list:
            # check if str or func for each single element
            conf_plot = [self.check_plot_type_single(f) for f in conf['plot']]
            # conf_plot = conf['plot'] # [j]
            # assert conf_plot is not type(str), "FIXME: plot callbacks is array of strings, eval strings"
        elif type(conf['plot']) is str:
            # conf_plot = self.eval_conf_str(conf['plot'])
            rkey = 'fp'
            conf_plot = code_compile_and_run(code = '%s = %s' % (rkey, conf['plot']), gv = plotfuncs, lv = {}, return_keys = [rkey])
            if type(conf_plot) is list:
                conf_plot = self.check_plot_type(conf, defaults)
            else:
                conf_plot = [conf_plot]
        else:
            conf_plot = [conf['plot']]
        return conf_plot

    def check_plot_type_single(self, f):
        """check_plot_type_single

        Subplot plotfunc configuration type fix function part 2: single list item

        Get subplot configuration item 'plot' and, if necessary,
        type-fix the value by translating strings to functions.

        Returns:
         - single function pointer
        """
        # convert a str to a func by compiling it
        if type(f) is str:
            # return self.eval_conf_str(f)
            rkey = 'fp'
            return code_compile_and_run(
                code = '%s = %s' % (rkey, f),
                gv = plotfuncs,
                lv = {},
                return_keys = [rkey]) # [rkey]
        else:
            return f

    def get_title_from_plot_type(self, plotfunc_conf):
        """get_title_from_plot_type

        Get title component from the plot type for automatic titling.
        """
        title = ""
        for plotfunc in plotfunc_conf:
            # get the plot type from the plotfunc type
            if hasattr(plotfunc, 'func_name'):
                # plain function
                plotfunc_ = plotfunc
            elif hasattr(plotfunc, 'func'):
                # partial'ized func
                # plottype = plotfunc.func.func_name
                plotfunc_ = plotfunc.func
            else:
                # unknown func type
                plotfunc_ = timeseries # "unk plottype"

            plottype = plotfunc_.__name__
            # self._debug('get_title_from_plot_type func_name / plottype = %s/%s' % (type(plottype), plottype))
            
            # # append plot type to title, unwrap if necessary
            # if plottype is 'wrap':
            #     self._debug('get_title_from_plot_type wrapped = %s' % (dir(plotfunc_), ))
            #     self._debug('get_title_from_plot_type wrapped = %s' % (plotfunc_.func_code, ))
            #     # title += self.get_title_from_plot_type([plotfunc_])
            # else:
            #     title += " " + plottype
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
        """if saveplot is set, compute filename and register top.outputs of type fig
        """
        if self.saveplot:
            # generate filename
            self.filename = '%s_%s.%s' % (self.top.datafile_expr, self.id, self.savetype)
            # register as top block output
            self.top.outputs['%s' % (self.id, )] = {
                'type': 'fig',
                'filename': self.filename,
                'label': self.top.id,
                'id': self.id,
                'desc': self.desc,
                'width': 1.0,
            }
    
class TextBlock2(BaseplotBlock2):
    """TextBlock2

    Block for text output, currently the only target / backend is *latex*.

    ..note:: This block duplicates Block2's :func:`Block2.latex_close` functionality. FIXME: merge
    """
    defaults = {
        'block_group': ['output', 'measure'],
        'savetype': 'tex',
        'title': None,
        'desc': None,
        'layout': None,
        'colwidth': 0.15, # default column width in textwidths
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # update child class 'self' defaults
        defaults = {}
        defaults.update(AnalysisBlock2.defaults, **self.defaults)
        self.defaults = defaults
        
        # super init
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # prepare for saving
        # self.prepare_saveplot()
        if self.saveplot:
            # self.filename = '%s_%s.%s' % (self.top.datafile_expr, self.id, self.savetype)
            self.filename = '%s/%s_%s.%s' % (self.top.datadir_expr, self.top.id, self.id, self.savetype)
            self._info('Filename set to %s' % (self.filename, ))

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
            top_id = re.sub(r'_', r'-', self.top.id)
            table_title = 'Results: %s' % (top_id)
            if self.title is not None:
                table_title = self.title
            table_id = '%s-%s' % (top_id, self.id)
            table_caption = self.desc
            table_width = self.colwidth * 2
            table_cellalign = 'r' * 2
            table_rowlables = ['Measure', 'Value']
            
            inputkeys = list(self.inputs.keys())
            inputkeys.sort()
            table_collables = inputkeys # [ink for ink in inputkeys]

            table_cells = [[ink] for ink in inputkeys]

            # table geometry
            # - default is two columns: | name | value |, one row for each input item
            # - if table spec is given, use that
            if self.layout is not None:
                self._debug('layout = %s' % (self.layout, ))
                assert 'cells' in self.layout, "Layout needs entries: numrows, numcols, rowlables, collables, cells"
                
                table_width = self.colwidth * (len(self.layout['cells'][0]) + 1)
                table_cellalign = 'r' * (len(self.layout['cells'][0]) + 1)
                if table_width > 0.99:
                    self._warning('table too wide for portrait textwidth with %f' % (table_width, ))

                table_rowlables = self.layout['rowlabels']
                table_collables = self.layout['collabels']

                table_cells = self.layout['cells']
                
            # table head
            # self.textbuf = '\\begin{tabularx}{\\textwidth}{|r|X|}\n'
            self.textbuf = """\\bigskip
\\begin{minipage}{\\linewidth}
\\centering
\\captionof{table}{%s} \\label{tab:%s}
\\small
\\begin{tabularx}{%f\\textwidth}{%s}\\toprule[1.0pt]\n""" % (table_title, table_id, table_width, table_cellalign)

            # table row labels
            self.textbuf += '\\textbf{{{0}}}'.format(table_rowlables[0])
            for rowlabel in table_rowlables[1:]:
                self.textbuf += '& \\textbf{{{0}}}'.format(rowlabel)
            self.textbuf += '\\\\\n\\midrule\n'
            
            # # for ink, inv in self.inputs.items():
            # inputkeys = self.inputs.keys()
            # inputkeys.sort()
            # for ink in inputkeys:
            #     inv = self.inputs[ink]
            #     self._info('ink = %s, inv = %s' % (ink, inv))
            #     self.textbuf += '{:} & ${:10.4f}$ \\\\\n'.format(re.sub(r'_', r'\_', ink), inv['val'].flatten()[0])

            for r, tablerow in enumerate(table_cells):
                self.textbuf += '{:}'.format(re.sub(r'_', r'\_', table_collables[r]))
                for tablecol in tablerow:
                    # print "cell", tablerow, tablecol
                    if tablecol is not None:
                        cellv = self.inputs[tablecol]['val'].flatten()[0]
                        self.textbuf += '& ${:10.4f}$'.format(cellv)
                    else:
                        self.textbuf += '& '
                self.textbuf +=  '\\\\\n' # row terminate
                    
            self.textbuf += '\\bottomrule[2pt]\n\\end{tabularx}\n'

            self.textbuf += """\\par
\\normalsize
\\bigskip
%s
\\end{minipage}\n""" % (table_caption)
            
#             self.textbuf += """\n
# \\bigskip
# \\begin{minipage}{\\linewidth}
# \\centering
# \\captionof{table}{Table Title} \\label{tab:title2} 
# \\begin{tabularx}{\\linewidth}{@{} C{1in} C{.85in} *4X @{}}\\toprule[1.5pt]
# \\bf Variable Name & \\bf Regression 1 & \\bf Mean & \\bf Std. Dev & \\bf Min & \\bf Max\\\\\\midrule
# text        &  text     & text      &  text     &  text     &text\\\\
# \\bottomrule[1.25pt]
# \\end {tabularx}\\par
# \\bigskip
# Should be a caption
# \\end{minipage}
# """

            if self.saveplot:
                self.save()

    def save(self):
        """Save the analysis, redirect to corresponding class method, passing the instance
        """
        try:
            f = open(self.filename, 'w')
            f.write(self.textbuf)
            f.flush()
            f.close()
            self._info('Saved texbuf (%d) to file %s' % (len(self.textbuf), self.filename))
        except Exception as e:
            self._error('Saving texbuf to file %s failed with %s at %s' % (self.filename, e, inspect.getframeinfo(inspect.currentframe()).lineno))

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
    defaults = {
        'bbox_inches': None, # 'tight',
        'wspace': 0.0,
        'hspace': 0.0,
        'axesspec': None,
    }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # update child class 'self' defaults
        defaults = {}
        defaults.update(AnalysisBlock2.defaults, **self.defaults)
        self.defaults = defaults

        # # defaults
        # self.wspace = 0.0
        # self.hspace = 0.0
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # configure figure and plot axes
        if hasattr(self, 'axesspec') and self.axesspec is not None and hasattr(self, 'fig_rows') and hasattr(self, 'fig_cols'):
            pass
        else:
            self.fig_rows = len(self.subplots)
            self.fig_cols = len(self.subplots[0])
            self.axesspec = None # (self.fig_rows, self.fig_cols)

        # create figure
        self.fig = makefig(
            rows = self.fig_rows, cols = self.fig_cols,
            wspace = self.wspace, hspace = self.hspace,
            title = self.title, axesspec=self.axesspec)
        # self.fig.tight_layout(pad = 1.0)
        # self.debug_print("fig.axes = %s", (self.fig.axes, ))

        # paint timestamp into plot
        if len(self.fig.axes) > 0:
            self.fig.axes[0].text(-0.05, 1.05, time.strftime('%Y%m%d-%H%M%S'), alpha = 0.5, transform=self.fig.axes[0].transAxes)
        
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
        # subplotstr = ''
        # if len(plotinst.subplots) > 0 and len(plotinst.subplots[0]) > 0 and plotinst.subplots[0][0].has_key('input'):
        #     subplotstr = "_".join(np.array(
        #         [[
        #             "r%d_c%d_%s" % (r, c, "_".join(subplot_input_fix(sbc['input'])),) for c,sbc in enumerate(sbr)
        #             ] for r, sbr in enumerate(plotinst.subplots)
        #         ]).flatten())

        # get filename from instance
        filename = plotinst.filename

        plotinst._debug("%s-%s.save filename = %s" % (plotinst.cname, plotinst.id, filename))

        if not hasattr(plotinst, 'savesize'):
            savescale = 3
            plotinst.savesize = (
                min(plotinst.fig_cols * 2.5 * savescale, 24),
                min(plotinst.fig_rows * 1.25 * savescale, 12))

        plotinst._debug('fig savesize w/h = %f/%f' % (plotinst.savesize[0], plotinst.savesize[1]))
        plotinst._debug('fig cols/rows = %s/%s' % (plotinst.fig_cols, plotinst.fig_rows))
        plotinst._debug('fig bbox_inches = %s' % (plotinst.bbox_inches))
        plotinst.fig.set_size_inches(plotinst.savesize)

        # write the figure to file
        try:
            plotinst._info("%s.savefig saving plot %s to filename = %s" % (plotinst.id, re.sub('\n', ' ', plotinst.title), filename))
            plotinst.fig.savefig(filename, dpi=300, bbox_inches=plotinst.bbox_inches)
        except Exception as e:
            logger.error("%s.savefig saving failed with %s" % (plotinst.id, e))

        # # FIXME: needed or obsolete?
        # # register figure for latex output
        # try:
        #     logger.debug(
        #         "%s.savefig, top = %s, has outputs = %s with keys %s",
        #         plotinst.id, plotinst.top,
        #         hasattr(plotinst.top, 'outputs'),
        #         plotinst.top.outputs.keys())
        #     # if plotinst.top.
        #     plotinst.top.outputs['latex']['figures'][plotinst.id] = {
        #         'filename': filename,
        #         'label': plotinst.top.id,
        #         'id': plotinst.id,
        #         'desc': plotinst.desc}
        #     # plotinst.fig.savefig(filename, dpi=300)
        # except Exception, e:
        #     logger.error(
        #         "%s.savefig configuring top.outputs['latex']['figures'] failed with %s at %s",
        #         plotinst.id, e,
        #         inspect.getframeinfo(inspect.currentframe()).lineno
        #     )
            
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
                print("Using inputs from log.log_store = %s with keys = %s instead of bus" % (log.log_store.filename, list(log.log_store.keys()), ))
                # commit data
                log.log_pd_store()
                # iterate input items
                for ink, inv in list(self.inputs.items()):
                    bus = '/%s' % (inv['bus'], )
                    # print "ink", ink, "inv", inv['bus'], inv['shape'], inv['val'].shape
                    # check if a log exists
                    if bus in list(log.log_store.keys()):
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

    def get_xaxis(self, subplotconf, xslice, plotlen, idxtup = None):
        """get the plot xaxis

        If configured as an input key, that input modulo the xslice param is taken.
        if configured but not input key, taken as literal array
        if not configured take minimum of plotlen and xlsice
        """
        if idxtup is not None:
            (i, j, k) = idxtup

        # xslice.start = int(xslice.start)
        # xslice.stop = int(xslice.stop)
             
        # configure x axis, default implicit number of steps
        if 'xaxis' in subplotconf:
            if type(subplotconf['xaxis']) is str and subplotconf['xaxis'] in list(self.inputs.keys()):
                t = self.inputs[subplotconf['xaxis']]['val'].T[xslice] # []
            else:
                t = subplotconf['xaxis'] # self.inputs[ink]['val'].T[xslice] # []
                # self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s xaxis setting t = %s from subplotconf['xaxis']" % (
                #     i, j, k, ink, t, ))
        else:
            if xslice.stop > plotlen:
                t = np.linspace(0, plotlen - 1, plotlen)
            else:
                # print('xslice = {0}'.format(xslice))
                t = np.linspace(xslice.start, xslice.start+plotlen-1, plotlen)[xslice]
        return t
                
    def subplotconf2kwargs(self, subplotconf, i, j, k=None):
        kwargs = {}
        for kw in [
                'aspect', 'orientation', 'labels',
                'title_pos',
                'xlabel', 'xlim', 'xticks', 'xticklabels', 'xinvert', 'xtwin',
                'ylabel', 'ylim', 'yticks', 'yticklabels', 'yinvert', 'ytwin',
                'lineseg_val', 'lineseg_idx',
            ]:
            if kw in subplotconf:
                if k is not None:
                    kwargs[kw] = listify(subplotconf[kw], k)
                else:
                    kwargs[kw] = subplotconf[kw]
            elif hasattr(self, kw):
                if k is not None:
                    kwargs[kw] = listify(getattr(self, kw), k)
                else:
                    kwargs[kw] = getattr(self, kw)

            # if k is not None:
            #     kwargs[kw] = listify(kwargs[kw], k)
                
            self._debug("plot_subplots pass 1 subplot[%d,%d] kwargs = %s" % (i, j, kwargs))
        return kwargs
            
    def plot_subplots(self):
        """FigPlotBlock2.plot_subplots

        This is a stub and has to be implement by children classes.
        """
        print("%s-%s.plot_subplots(): implement me" % (self.cname, self.id,))

class PlotBlock2(FigPlotBlock2):
    """PlotBlock2 class
    
    Block for plotting timeseries and histograms

    - FIXME: check_list (convert scalars to lists)
    - FIXME: dict_safe  (safe dict getter return None on key fail)
    - FIXME: dict_safe_if (True if dict has key and dict[key] True)
    """
    # PlotBlock2.defaults
    defaults = {
        # 'inputs': {
        #     'x': {'bus': 'x'}, # FIXME: how can this be known? id-1?
        # },
        'blocksize': 1,
        'xlim_share': True,
        'ylim_share': True,
        # 'subplots': [[{'input': ['x'], 'plot': timeseries}]],
        'subplots': [[{}]],
        'plot_subplots_pass_1_flag': False,
        'plot_subplots_pass_2_flag': False,
        'bbox_inches': 'tight',
        'wspace': 0.0,
        'hspace': 0.0,
        
    }

    defaults_subplotconf = {
        'input': [],
        'xlabel': None,
        'ylabel': None,
        'loc': 'left',
        'cmap': ['rainbow'],
        # 'xticks': False,
        # 'yticks': False,
    }
        
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_coding_lineseg(self, subplotconf, input_ink, xxx_todo_changeme):
        """PlotBlock2.plot_coding_lineseg
        """
        (i, j, k) = xxx_todo_changeme
        if 'lineseg_val' in subplotconf and 'lineseg_idx' in subplotconf:
            lineseg_val_k = listify(subplotconf['lineseg_val'], k)
            if lineseg_val_k is None: return input_ink

            # else, do work
            l_ = []
            for lineseg_input in lineseg_val_k:
                self._debug('lineseg_input = %s' % (lineseg_input))
                l_.append(self.inputs[lineseg_input]['val'])
            self._debug('lineseg_input l_ = %s' % (l_))
            input_ink['val'] = np.vstack(l_)
            input_ink['shape'] = input_ink['val'].shape
            self._debug('input_ink val   = %s' % (input_ink['val']))
            self._debug('input_ink shape = %s' % (input_ink['shape'],))
        return input_ink
        
    def plot_coding_event(self, subplotconf, input_ink, xxx_todo_changeme1):
        """PlotBlock2.plot_coding_event

        Event based recoding of incoming timeseries. Events are
        identified by a condition e.g. y[t] != 0 and two new arrays
        are created coding the time and the values / properties of the
        event. These have a new shape with length equal to the number
        of events.

        TODO: Generalize from plot internal to eventcoding block, but
        this involves touching the entire execution model, which we
        want to be event aware anyway.
        """
        (i, j, k) = xxx_todo_changeme1
        if 'event' not in subplotconf: return input_ink

        # FIXME: robust dict
        event = listify(subplotconf['event'], k)
                
        # return if event type is not set for this subplot input channel
        if not event: return input_ink

        # recode as event (aer) sequence if this subplot input channel
        # is configured as event type
        data = myt(input_ink['val'])
        
        # apply condition, default condition is non-zeroness, y[x] != 0
        datanz = (data != 0.0).ravel()
        self._debug('datanz = %s' % (datanz))
        self._debug('data = %s' % (data.shape,))
        # create full index t axis / axis
        t_ = np.arange(0, data.shape[0])
        # grab index positions of condition matches from the full index
        datanz_idx = t_[datanz]
        # grab the event data at the corresponding index positions
        datanz_val = data[datanz]
        self._debug('datanz_idx = %s' % (datanz_idx.shape,))
        self._debug('datanz_val = %s' % (datanz_val.shape,))
        # modify input item
        input_ink['t'] = datanz_idx
        input_ink['val'] = datanz_val
        input_ink['shape'] = datanz_val.T.shape
        # fix plotlen, xslice, shape
        self._debug('subplotconf %s' % (list(subplotconf.keys()),))
        self._debug('  input_ink %s' % (list(input_ink.keys()),))
        
        # return the modified input item 'input_ink'
        return input_ink
       
    def plot_subplots(self):
        """PlotBlock2.plot_subplots

        Loop over configured subplots and plot the input data
        according to the configuration.

        The function does not take any arguments. Instead, the args
        are taken from the :data:`subplots` member.

        subplots is a list of lists, specifying a subplot
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
        # self._debug("%s plot_subplots self.inputs = %s", self.cname, self.inputs)

        # subplots pass 0: remember ax limits
        sb_rows = len(self.subplots)
        sb_cols = len(self.subplots[0])
        
        rows_ylim_max = [(1e9, -1e9) for _ in range(sb_rows)]
        cols_xlim_max = [(1e9, -1e9) for _ in range(sb_cols)]

        # set default plot size when we know the subplot geometry
        default_plot_scale = 3
        default_plot_size = (sb_cols * 2.5 * default_plot_scale, sb_rows * 1 * default_plot_scale)
        self.fig.set_size_inches(default_plot_size)

        # subplots pass 1: the hard work, iterate over subplot config and build the plot
        for i, subplot in enumerate(self.subplots):    # rows are lists (of dicts)
            for j, subplotconf_ in enumerate(subplot): # cols are dicts (of k,v tuples; columns)

                # subplotconf defaults
                if type(subplotconf_) is dict:
                    subplotconf = {}
                    subplotconf.update(self.defaults_subplotconf)
                    subplotconf.update(subplotconf_)
                    subplotconf_.update(subplotconf)
                    
                # empty gridspec cell
                if subplotconf is None or len(subplotconf) < 1:
                    self._warning('plot_subplots pass 1 subplot[%d,%d] no plot configured, subplotconf = %s' % (i, j, subplotconf))
                    return

                # convert conf items into list if they aren't (convenience)
                for input_spec_key in [
                        'input', 'ndslice', 'shape',
                        'xslice', 'event',
                    ]:
                    if input_spec_key in subplotconf:
                        subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                        # print "    id: %s, subplotconf[%s] = %s" % (self.id, input_spec_key, subplotconf[input_spec_key])

                # linear (flattened) axes index from subplot row_i * col_j
                # idx = (i*self.fig_cols)+j
                idx = (i*len(subplot))+j
                    
                # remember axes and their labels created during pass 1 e.g. by twin[xy]()
                axs = {
                    'main': {
                        'ax': self.fig.axes[idx],
                        'labels': []
                    }
                }

                # remember input data processed for plot input in plotdata
                plotdata = OrderedDict()
                # remember input data processed for plot input in plotdatad, dict storing more info on the plot
                plotdatad = OrderedDict()
                # remember distinct input variables
                plotvar = ' '
                
                # get this subplot's plotfunc configuration and make sure its a list
                plotfunc_conf = self.check_plot_type(subplotconf)
                # print "%s-%s plotfunc_conf = %s" % (self.cname, self.id, plotfunc_conf)
                assert type(plotfunc_conf) is list, "plotfunc_conf must be a list, not %s" % (type(plotfunc_conf), )

                # create a plot title
                title = ''
                if 'title' in subplotconf:
                    if subplotconf['title'] is not None: title += subplotconf['title']
                else:
                    # add plotfunc type to default title
                    # if title == '':
                    title += self.get_title_from_plot_type(plotfunc_conf)

                # generate labels all at once
                # l = [['%s[%d]' % (ink, invd) for invd in range(inv.shape[1])] for ink, inv in plotdata.items()]
                # self._debug("pre labels l = %s, items = %s" % (l, plotdata.items(), ))
                # labels = reduce(lambda x, y: x + y, l)
                labels = []

                ################################################################################
                # loop over subplot 'input'
                for k, ink in enumerate(subplotconf['input']):
                    
                    # FIXME: array'ize this loop
                    # vars: input, ndslice, shape, xslice, ...
                    input_ink = self.check_plot_input(ink, [i, j, k])
                    if not input_ink: continue

                    # check for event recoding
                    input_ink = self.plot_coding_event(subplotconf, input_ink, (i, j, k))

                    # check for linesegment recoding
                    input_ink = self.plot_coding_lineseg(subplotconf, input_ink, (i, j, k))

                    # get numsteps of data for the input
                    if 'shape' not in input_ink:
                        input_ink['shape'] = input_ink['val'].shape
                    # plotlen intrinsic
                    plotlen = input_ink['shape'][-1] # numsteps at shape[-1]
                    # plotlen extent

                    # set default slice
                    xslice = slice(0, plotlen)
                    # compute final shape of plot data, custom transpose from horiz time to row time
                    plotshape = mytupleroll(input_ink['shape'])
                    
                    # print "%s.subplots defaults: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                
                    # x axis slice spec
                    if 'xslice' in subplotconf:
                        # get slice conf
                        # if type(subplotconf['xslice']) is list:
                        #     subplotconf_xslice = subplotconf['xslice'][k]
                        # else:
                        #     subplotconf_xslice = subplotconf['xslice']
                        subplotconf_xslice = listify(subplotconf['xslice'], k)
                        
                        # set slice
                        xslice = slice(subplotconf_xslice[0], subplotconf_xslice[1])
                        
                        # update plot length
                        plotlen = xslice.stop - xslice.start
                        
                        # and plot shape
                        plotshape = (plotlen, ) + tuple((b for b in plotshape[1:]))
                    
                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s xslice xslice = %s, plotlen = %d, plotshape = %s" % (
                        i, j, k, ink, xslice, plotlen, plotshape))

                    # explicit shape key
                    # FIXME: shape overrides xslice
                    if 'shape' in subplotconf:
                        # if len(subplotconf['shape']) > 1:
                        #     subplotconf_shape = subplotconf['shape'][k]
                        # else:
                        #     subplotconf_shape = subplotconf['shape'][0]
                        subplotconf_shape = listify(subplotconf['shape'], k)
                            
                        # update the plot shape 'plotshape' via custom transpose from column t [-1] to row t [0] repr
                        plotshape = mytupleroll(subplotconf_shape)
                        
                        # update plot length 'plotlen'
                        plotlen = int(plotshape[0])
                        
                        # update x-slice 'xslice'
                        xslice = slice(0, plotlen)

                    # convert to int
                    plotshape = tuple2inttuple(plotshape)

                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s shape xslice = %s, plotlen = %d, plotshape = %s" % (
                        i, j, k, ink, xslice, plotlen, plotshape))
                    
                    # # configure x axis, default implicit number of steps
                    # if subplotconf.has_key('xaxis'):
                    #     if type(subplotconf['xaxis']) is str and subplotconf['xaxis'] in self.inputs.keys():
                    #         t = self.inputs[subplotconf['xaxis']]['val'].T[xslice] # []
                    #     else:
                    #         t = subplotconf['xaxis'] # self.inputs[ink]['val'].T[xslice] # []
                    #         self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s xaxis setting t = %s from subplotconf['xaxis']" % (
                    #             i, j, k, ink, t, ))
                    # else:
                    #     if xslice.stop > plotlen:
                    #         t = np.linspace(0, plotlen - 1, plotlen)
                    #     else:
                    #         t = np.linspace(xslice.start, xslice.start+plotlen-1, plotlen)[xslice]

                    # get t axis / xaxis
                    if 't' in input_ink:
                        t_ = input_ink['t']
                    else:
                        t_ = self.get_xaxis(subplotconf, xslice, plotlen, (i, j, k))
                    
                    # print "%s.plot_subplots k = %s, ink = %s" % (self.cname, k, ink)
                    # plotdata[ink] = input_ink['val'].T[xslice]
                    # if ink == 'd0':
                    #     print "plotblock2", input_ink['val'].shape
                    #     print "plotblock2", input_ink['val'][0,...,:]
                    # ink_ = "%s_%d" % (ink, k)
                    ink_ = "%d-%s" % (k + 1, ink)
                    # print "      input shape %s: %s" % (ink, input_ink['val'].shape)

                    # if explicit n-dimensional slice is given
                    if 'ndslice' in subplotconf:
                        # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                        # slice the data to spec, custom transpose from h to v time
                        # ndslice = subplotconf['ndslice'][k]
                        ndslice = listify(subplotconf['ndslice'], k)
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice ndslice = %s" % (
                            i, j, k, ink, ndslice))
                        
                        # get the data nd-sliced
                        plotdata[ink_] = myt(input_ink['val'])[ndslice]
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice sb['ndslice'] = %s, numslice = %d" % (
                            i, j, k, ink, subplotconf['ndslice'][k], len(subplotconf['ndslice'])))
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice plotdata[ink_] = %s, input = %s" % (
                            i, j, k, ink, plotdata[ink_].shape, input_ink['val'].shape, ))
                    else:
                        # get the data en bloc
                        plotdata[ink_] = myt(input_ink['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))

                    # dual plotdata record hack
                    axd = axs['main']
                    # ax_ = axs['main']['ax']
                    if 'xtwin' in subplotconf:
                        subplotconf_xtwin = listify(subplotconf['xtwin'], k)
                        if subplotconf_xtwin:
                            if 'xtwin' not in axs:
                                axs['xtwin'] = {'ax': axs['main']['ax'].twinx(), 'labels': []}
                            axd = axs['xtwin'] # ['ax']
                            
                        # if type(subplotconf['xtwin']) is list:
                        #     if subplotconf['xtwin'][k]:
                        #         if not axs.has_key('xtwin'):
                        #             axs['xtwin'] = {'ax': axs['main']['ax'].twinx(), 'labels': []}
                        #         axd = axs['xtwin'] # ['ax']
                                
                        # else:
                        #     if subplotconf['xtwin']:
                        #         if not axs.has_key('xtwin'):
                        #             axs['xtwin'] = {'ax': axs['main']['ax'].twinx(), 'labels': []}
                        #         axd = axs['xtwin'] # ['ax']

                    assert plotdata[ink_].shape != (0,), "no data to plot"
                    # print "      input = %s" % input_ink['val']
                    # print "      id %s, ink = %s, plotdata = %s, plotshape = %s" % (self.id, ink_, plotdata[ink_], plotshape)
                    # plotdata[ink_] = plotdata[ink_].reshape((plotshape[1], plotshape[0])).T

                    # reshape the data
                    plotdata[ink_] = plotdata[ink_].reshape(plotshape)
                    
                    # fix nans
                    plotdata[ink_][np.isnan(plotdata[ink_])] = -1.0
                    plotvar += "%s, " % (input_ink['bus'],)

                    # generate labels
                    # FIXME: range(?) when len(shape) != 2
                    numlabels = plotdata[ink_].shape[0]
                    if len(plotdata[ink_].shape) > 1:
                        numlabels = plotdata[ink_].shape[-1]
                    l1  = ['%s[%d]' % (ink, invd) for invd in range(numlabels)]
                    l2 = reduce(lambda x, y: x+y, l1)
                    
                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s labels numlabels = %s, l1 = %s, l2 = %s" % (
                        i, j, k, ink, numlabels, l1, l2, ))
                    l = l1
                    labels.append(l)
                    
                    axd['labels'] += l # .append(l)
                    ax_ = axd['ax']
                    
                    # store fully prepared / ready to plot subplot
                    # input, e.g. for pass 2 consolidation. Storing: ax, data, legend labels, t axis
                    # FIXME: in sync with subplotconf, initialize as
                    #        copy and updated from option triggered computations
                    plotdatad[ink_] = {
                        'ax': ax_,
                        'data': plotdata[ink_],
                        'labels': l,
                        't': t_,
                    }
                # end loop over subplot input
                ################################################################################

                ############################################################
                # finalize labels
                if len(labels) == 1:
                    labels = labels[0]
                elif len(labels) > 1:
                    l3 = reduce(lambda x, y: x+y, labels)
                    labels = l3
                    
                self._debug("plot_subplots pass 1 subplot[%d,%d] labels after subplotconf.input = %s" % (
                    i, j, labels, ))
                subplotconf['labels'] = labels
                
                ################################################################################
                # mode: stacking, combine inputs into one backend
                #       plot call to automate color cycling etc
                if 'mode' in subplotconf:
                    """FIXME: fix dangling effects of stacking"""
                    # ivecs = tuple(myt(input_ink['val'])[xslice] for k, ink in enumerate(subplotconf['input']))
                    ivecs = [plotdatav for plotdatak, plotdatav in list(plotdata.items())]
                    # plotdata = {}
                    if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                        plotdata['_stacked'] = np.hstack(ivecs)
                        plotdatad['_stacked'] = {
                            'ax': plotdatad[list(plotdata.keys())[0]]['ax'],
                            'data': plotdata['_stacked'],
                            'labels': labels,
                            't': t_,
                        }

                # get explicit xaxis (t)
                # FIXME: overrides xslice (?), shape (?), plotshape/plotlen, event
                if 'xaxis' in subplotconf:
                    # xaxis given as another input signal via bus key
                    if type(subplotconf['xaxis']) is str and subplotconf['xaxis'] in list(self.inputs.keys()):
                        inv = self.inputs[subplotconf['xaxis']]
                    # xaxis given directly here (check)
                    else:
                        inv = self.inputs[ink]

                    # titles
                    if 'bus' in inv:
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

                # transfer plot_subplot configuration keywords subplotconf to plot kwargs
                kwargs = self.subplotconf2kwargs(subplotconf, i, j)
                
                # prep axis
                ax = axs['main']['ax']
                # self.fig.axes[idx].clear()
                ax.clear()
                inkc = 0
                
                # colors
                num_cgroups = 5
                num_cgroup_color = 5
                num_cgroup_dist = 255/num_cgroups
                # cmap_str = 'cyclic_mrybm_35_75_c68'
                # cmap_str = 'colorwheel'
                cmap_str = subplotconf['cmap'][0] # 'rainbow'
                cmap_idx = None
                cmap_off = [0 for _ in range(len(plotdata) + 1)]
                cmap_off_group  = [0 for _ in range(len(plotdata) + 1)]
                cmap_off_single = [0 for _ in range(len(plotdata) + 1)]
                
                if 'cmap_idx' in subplotconf:
                    cmap_idx = subplotconf['cmap_idx']
                self._debug("plot_subplots pass 1 subplot[%d,%d] cmap_idx = %s" % (i, j, cmap_idx))
                if 'cmap_off' in subplotconf:
                    assert type(subplotconf['cmap_off']) is list, "cmap_off param needs to be a list"
                    cmap_off = subplotconf['cmap_off']
                self._debug("plot_subplots pass 1 subplot[%d,%d] cmap_off = %s" % (i, j, cmap_off))

                ax.set_prop_cycle(
                    get_colorcycler(
                        cmap_str = cmap_str, cmap_idx = cmap_idx,
                        c_s = (inkc + cmap_off[inkc]) * num_cgroup_dist, c_e = (inkc + cmap_off[inkc] + 1) * num_cgroup_dist, c_n = num_cgroup_color
                    )
                )
                
                # stacked data plot
                if '_stacked' in plotdata:
                    self._debug("plot_subplots pass 1 subplot[%d,%d] plotting stacked" % (i, j, ))
                    # ordinate = t, t axis, xaxis
                    t_ = plotdatad['_stacked']['t']
                    data_ = plotdatad['_stacked']['data'] # plotdata['_stacked']
                    plotfunc_conf[0](ax, data=data_, ordinate=t_, title=title, **kwargs)
                    # interaction
                    fig_interaction(self.fig, ax, plotdata['_stacked'])

                
                # iterate over plotdata items
                title_ = title
                inv_accum = []
                for ink, inv in list(plotdata.items()):
                    ax = plotdatad[ink]['ax']
                    t_ = plotdatad[ink]['t']
                    self._debug("plot_subplots pass 1 subplot[%d,%d] plotdata[%s] = inv.sh = %s, plotvar = %s, t.sh = %s" % (
                        i, j, ink, inv.shape, plotvar, t_.shape))

                    # if multiple input groups, increment color group and compensate for implicit cycle state
                    if inkc > 0:
                        if len(subplotconf['cmap']) > 1:
                            cmap_str = subplotconf['cmap'][inkc]
                            
                        ax.set_prop_cycle(
                            get_colorcycler(
                                cmap_str = cmap_str, cmap_idx = cmap_idx,
                                c_s = (inkc + cmap_off[inkc])     * num_cgroup_dist - (inkc * num_cgroup_color),
                                c_e = (inkc + cmap_off[inkc] + 1) * num_cgroup_dist - (inkc * num_cgroup_color),
                                c_n = num_cgroup_color,
                                # c_s = (inkc + 1 + cmap_off[inkc]) * num_cgroup_dist, c_e = (inkc + cmap_off[inkc] + 2) * num_cgroup_dist, c_n = num_cgroup_color
                                # c_s = (inkc + 1) * num_cgroup_dist, c_e = (inkc + cmap_off[inkc]) * num_cgroup_dist, c_n = num_cgroup_color
                            ),
                        )

                    # select single element at first slot or increment index with plotdata items
                    plotfunc_idx = inkc % len(plotfunc_conf)
                    
                    # # ax.set_prop_cycle(get_colorcycler(cmap_str = tmp_cmaps_[inkc]))
                    # for invd in range(inv.shape[1]):
                    #     label_ = "%s[%d]" % (ink, invd + 1)
                    #     if len(label_) > 16:
                    #         label_ = label_[:16]
                    #     labels.append(label_)

                    # # get t / xaxis from plotdatad
                    # if 't' in plotdatad:
                    #     t_ = plotdatad['t']
                    # else:
                    #     t_ = t

                    # transfer plot_subplot configuration keywords subplotconf to plot kwargs
                    kwargs = self.subplotconf2kwargs(subplotconf, i, j, k)
                    
                    # this is the plot function array from the config
                    if '_stacked' not in plotdata:
                        # print "    plot_subplots plotfunc", plotfunc_conf[plotfunc_idx]
                        # print "                      args", ax, inv, t, title, kwargs
                        plotfunc_conf[plotfunc_idx](
                            ax = ax, data = inv, ordinate = t_,
                            title = title_, **kwargs)
                        # avoid setting title multiple times
                        # title_ = None

                    # label = "%s" % ink, title = title
                    # tmp_cmaps_ = [k for k in cc.cm.keys() if 'cyclic' in k and not 'grey' in k]

                    inv_accum.append(inv)
                        
                    # metadata
                    inkc += 1

                if len(plotdata) > 0:
                    # changed 20180507: don't stack but dict
                    # inv_accum_ = np.hstack(inv_accum)
                    
                    # interaction
                    # fig_interaction(self.fig, ax, inv_accum_)
                    # call with data dict plotdatad
                    fig_interaction(self.fig, ax, plotdatad)
                    
                # reset to main axis
                ax = axs['main']['ax']
                # store the final plot data
                # print "sbdict", self.subplots[i][j]
                sb = self.subplots[i][j]
                sb['p1_plottitle'] = title
                sb['p1_plotdata'] = plotdata
                sb['p1_plotvar'] = plotvar
                sb['p1_plotlabels'] = labels
                sb['p1_plotxlim'] = ax.get_xlim()
                sb['p1_plotylim'] = ax.get_ylim()
                sb['p1_axs'] = axs
                sb['p1_plotdatad'] = plotdatad

                # save axis limits
                # print "xlim", ax.get_xlim()
                if sb['p1_plotxlim'][0] < cols_xlim_max[j][0]: cols_xlim_max[j] = (sb['p1_plotxlim'][0], cols_xlim_max[j][1])
                if sb['p1_plotxlim'][1] > cols_xlim_max[j][1]: cols_xlim_max[j] = (cols_xlim_max[j][0], sb['p1_plotxlim'][1])
                if sb['p1_plotylim'][0] < rows_ylim_max[i][0]: rows_ylim_max[i] = (sb['p1_plotylim'][0], rows_ylim_max[i][1])
                if sb['p1_plotylim'][1] > rows_ylim_max[i][1]: rows_ylim_max[i] = (rows_ylim_max[i][0], sb['p1_plotylim'][1])

                    
                # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)
        # subplots pass 1: done

        # check for and run pass 2
        if not self.plot_subplots_pass_2_flag:
            self.plot_subplots_pass_2(cols_xlim_max, rows_ylim_max)
            self.plot_subplots_pass_2_flag = True

        self._debug("plot_subplots len fig.axes = %d" % (len(self.fig.axes)))

        # plt mechanics
        plt.draw()
        plt.pause(1e-9)

    def plot_subplots_pass_2(self, cols_xlim_max, rows_ylim_max):
        ################################################################################
        # subplots pass 2: clean up and compute globally shared dynamic vars
        # adjust xaxis
        for i, subplot in enumerate(self.subplots):
            idx = (i*self.fig_cols)            
            # idx = (i*len(fig.axes)/len(self.subplots))
            for j, subplotconf in enumerate(subplot):
                # subplot handle shortcut
                sb = self.subplots[i][j]
                
                self._debug("    0 subplotconf.keys = %s" % (list(subplotconf.keys()), ))
                
                # subplot index from rows*cols
                # idx = (i*self.fig_cols)+j
                idx = (i*len(subplot))+j
                    
                # axis handle shortcut
                ax = self.fig.axes[idx]

                # check empty input
                if len(subplotconf['input']) < 1:
                    # ax = self.fig.axes[idx]
                    # ax = fig.gca()
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    continue
                    
                # consolidate axis limits
                if self.xlim_share and 'xlim' not in subplotconf:
                    # self._debug("subplots pass 2 consolidate ax[%d,%d] = %s" % (i, j, ax, cols_xlim_max[j]))
                    # self._debug("subplots pass 2             xlim = %s" % (cols_xlim_max[j]))
                    # self._debug("subplots pass 2             subplotconf.keys = %s" % (subplotconf.keys()))
                    ax.set_xlim(cols_xlim_max[j])
                if self.ylim_share and 'ylim' not in subplotconf:
                    # self._debug("subplots pass 2 consolidate ax[%d,%d] = %s" % (i, j, ax, rows_ylim_max[j]))
                    # self._debug("subplots pass 2             ylim = %s" % (rows_ylim_max[j]))
                    # self._debug("subplots pass 2             subplotconf.keys = %s" % (subplotconf.keys()))
                    ax.set_ylim(rows_ylim_max[i])

                # check axis inversion
                ax_invert(ax, **subplotconf)
                
                # legend: location
                loc = 'left'
                if 'legend_loc' in sb:
                    loc = sb['legend_loc']

                # legend: spacing
                legend_space = 0.9
                if 'legend_space' in sb:
                    legend_space = sb['legend_space']
                    
                # check for twin axes
                if len(sb['p1_axs']) == 1:
                    # single axis
                    if 'legend' in sb and sb['legend']:
                        assert type(sb['legend']) is dict, 'Legend param needs to be a dict = {\'label\': [ax handles, ...]}'
                        labels = list(sb['legend'].keys())
                        # artists
                        lines = list(sb['p1_axs']['main']['ax'].get_lines())
                        self._info('|lines| = %s' % (len(lines)))
                        if len(lines) < 1:
                            handles = None
                        else:
                            handles = [lines[l] for l in list(sb['legend'].values())]
                    else:
                        labels = sb['p1_plotlabels']
                        handles = None

                    if 'legend' in sb and not sb['legend']:
                        pass
                    else:
                        custom_legend(
                            labels = labels,
                            handles = handles,
                            ax = ax, resize_by = legend_space,
                            loc = loc)
                        
                    ax_set_aspect(ax, **subplotconf)
                else:
                    # twin axes
                    lg_ = None
                    # for axk, ax in sb['p1_axs'].items():
                    # for pdk, pdv in sb['plotdatad'].items():
                    #    ax = pdv['ax']
                    for k, axk in enumerate(sb['p1_axs']):
                        ax = sb['p1_axs'][axk]['ax']
                        labels = sb['p1_axs'][axk]['labels']
                        # if axk == 'main' and sb['p1_axs'].has_key('xtwin'): loc_ = 'right'
                        # else: loc_ = loc
                        # if lg_ is not None:
                        #     print "lg_.loc", lg_
                        # if loc == 'left': locx = 1.05
                        # elif loc == 'right': locx = -0.15
                        # else: locx = 0.0
                        # loc_ = (locx, (k * 0.45))
                        loc_ = loc
                        custom_legend(
                            labels = labels, # sb['p1_plotlabels'],
                            ax = ax, resize_by = legend_space,
                            loc = loc_, lg = lg_)
                        lg_ = ax.get_legend()

                        # set aspect after placing legend
                        # self._debug("    1 subplotconf.keys = %s" % (subplotconf.keys(), ))
                        ax_set_aspect(ax, **subplotconf)
                
                # put_legend_out_top(labels = labels, ax = ax, resize_by = 0.8)

# plot a matrix via imshow/pcolor
class ImgPlotBlock2(FigPlotBlock2):
    """ImgPlotBlock2 class
    
    Block for plotting 2-dimensional data as an image (matrices, scans, ...)

    Args:
     - conf(dict): configuration dictionary
     - paren(Block2): pointer to parent (graph)
     - paren(Block2): pointer to topblock (graph)

    Returns:
     - None

    An example configuration snippet for :class:`ImgPlotBlock2`

.. code:: python

    (
        'blockid', # freestyle string from [A-Za-z0-9_\-]
        {
            'block': ImgPlotBlock2,
            'params': {
                'blocksize': numsteps,
                'saveplot': False,
                'savetype': 'pdf',
                'savesize': (width x height) * scale constant,
                'wspace': width pad,
                'hspace': height pad,
                'desc': Description of the plot which is put into the figure caption during latex output,
                'vlim_share': True, # share pixel value limits over entire figure?
                'inputs': {
                    'datain': {'bus': 'data/x', 'shape': (32, 32)},
                    # ...
                },
                # subplots is a list of lists, first axis are subplot rows, second axis are subplot columns
                'subplots': [

                    [
                        {
                            'input': ['datain'],
                            'vlim_share': True, # share pixel value limits over FIXME
                            'plot': [plotfunc],
                            'title': 'subplot title',
                            # x/y-label, lim, ticks, see also PlotBlock2
                        }
                    ],

                ],
                'logging': False,
                'debug': False,
            }
        }
    )
    """
    defaults = {
        'vlim_share': True, # globally share pixel value limits over subplots
    }
        
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        """ImgPlotBlock2 worker func

        Iterate subplots and translate config to matplotlib img
        plotting using pcolor.
        """
        self._debug("plot_subplots self.inputs = %s" % (self.inputs, ))
        
        # subplots pass 0: preprocessing
        # - compute value ranges and limits, e.g. for shared normalization

        # get outer geometry from subplots list
        numrows = len(self.subplots)
        numcols = len(self.subplots[0])

        # alloc extrema
        extrema = np.zeros((2, numrows, numcols))

        # value min/max for subplot columns
        vmins_sb = [[] for i in range(numcols)]
        vmaxs_sb = [[] for i in range(numcols)]

        # value min/max for subplot columns
        vmins = [None for i in range(numcols)]
        vmaxs = [None for i in range(numcols)]
        # value min/max for subplot rows
        vmins_r = [None for i in range(numrows)]
        vmaxs_r = [None for i in range(numrows)]

        # iterate subplots
        for i, subplot in enumerate(self.subplots): # rows
            for j, subplotconf in enumerate(subplot): # cols
                # check conditions
                assert 'shape' in subplotconf, "image plot needs 'shape' entry in subplotconf but only has %s" % (list(subplotconf.keys()),)
                
                # make it a list if it isn't
                for input_spec_key in ['input', 'ndslice', 'shape']:
                    if input_spec_key in subplotconf:
                        subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                        
                # for img plot use only first input item (FIXME: mixing?)
                subplotin = self.inputs[subplotconf['input'][0]]
                # print "subplotin[%d,%d].shape = %s / %s" % (i, j, subplotin['val'].shape, subplotin['shape'])
                vmins_sb[j].append(np.min(subplotin['val']))
                vmaxs_sb[j].append(np.max(subplotin['val']))
                extrema[0,i,j] = np.min(subplotin['val'])
                extrema[1,i,j] = np.max(subplotin['val'])
                # print "i", i, "j", j, vmins_sb, vmaxs_sb

        # debugging
        self._debug("%s mins = %s" % (self.id, extrema[0], ))
        self._debug("%s maxs = %s" % (self.id, extrema[1], ))
        # convert to ndarray
        vmins_sb = np.array(vmins_sb)
        vmaxs_sb = np.array(vmaxs_sb)
        # print "vmins_sb, vmaxs_sb", i, j, vmins_sb.shape, vmaxs_sb.shape

        # iterate columns
        for i in range(numcols):
            # row min is min over all cols
            vmins[i] = np.min(vmins_sb[i])
            # vmins[1] = np.min(vmins_sb[1])
            # row max is max over all cols
            vmaxs[i] = np.max(vmaxs_sb[i])
            # vmaxs[1] = np.max(vmaxs_sb[1])

        # for i in range(numrows):
        #     vmins_r[i] = np.min(vmins_sb[i])
        #     # vmins[1] = np.min(vmins_sb[1])
        #     vmaxs_r[i] = np.max(vmaxs_sb[i])
        #     # vmaxs[1] = np.max(vmaxs_sb[1])

        # # get extrema
        # rowmins = np.min(extrema[0], axis = 0) 
        # rowmaxs = np.max(extrema[1], axis = 0) 
        # colmins = np.min(extrema[0], axis = 1) 
        # colmaxs = np.max(extrema[1], axis = 1)
        
        # self._debug("plot_subplots rowmins = %s, rowmaxs = %s, colmins = %s, colmaxs = %s" % (rowmins, rowmaxs, colmins, colmaxs))
        
        if True:
            for i, subplot in enumerate(self.subplots): # rows
                for j, subplotconf in enumerate(subplot): # cols

                    # map loop indices to gridspec linear index
                    idx = (i*self.fig_cols)+j
                    # print "self.inputs[subplotconf['input']][0].shape", self.inputs[subplotconf['input'][0]]['val'].shape, self.inputs[subplotconf['input'][0]]['shape']

                    xslice = slice(None)
                    yslice = slice(None)

                    # transfer plot_subplot configuration keywords subplotconf to plot kwargs
                    kwargs = self.subplotconf2kwargs(subplotconf, i, j)

                    # check for slice specs
                    if 'xslice' in subplotconf:
                        xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                        # print "xslice", xslice, self.inputs[subplotconf['input']][0].shape

                    if 'yslice' in subplotconf:
                        yslice = slice(subplotconf['yslice'][0], subplotconf['yslice'][1])
                        # print "yslice", yslice, self.inputs[subplotconf['input']][0].shape

                    # min, max values for colormap
                    axis = 0
                    aidx = j
                    if 'vaxis' in subplotconf:
                        if subplotconf['vaxis'] == 'rows':
                            axis = 1
                            aidx = i

                    vmin = None
                    vmax = None
                            
                    if type(self.vlim_share) is bool and self.vlim_share:
                        # get subplot item vmin, vmax from global plot extrema
                        vmin = np.min(extrema[0], axis = axis)[aidx]
                        vmax = np.max(extrema[1], axis = axis)[aidx]
                    elif type(self.vlim_share) is list and self.vlim_share[i,j]:
                        vmin = np.min(extrema[0], axis = axis)[aidx]
                        vmax = np.max(extrema[1], axis = axis)[aidx]
                        
                    # override subplot item vmin, vmax from subplotconf
                    if 'vmin' in subplotconf:
                        vmin = subplotconf['vmin']
                    if 'vmax' in subplotconf:
                        vmax = subplotconf['vmax']
                    
                    self._debug('subplot vlim default: i = %d, j = %d, vmin = %s, vmax = %s' % (i, j, vmin, vmax))
                    self._debug('subplot vlim extrama = %s' % (extrema))
                    # vmin = vmins[sbidx]
                    # vmax = vmaxs[sbidx]
                    # vmin = extrema[0]

                    # self._debug('subplot vlim override: i = %d, j = %d, vmin = %s, vmax = %s' % (i, j, vmin, vmax))
                        
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][xslice,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,xslice]
                    
                    # print "%s plot_subplots self.inputs[subplotconf['input'][0]]['val'].shape = %s" % (self.cname, self.inputs[subplotconf['input'][0]]['val'].shape)
                    # old version
                    # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][yslice,xslice]
                    
                    k = 0
                    ink = subplotconf['input'][k]
                    input_ink = self.check_plot_input(ink, [i, j, k])
                    if not input_ink: continue
                        
                    # FIXME completeness if input is ndim, currently only first dim is handled
                    if 'ndslice' in subplotconf:
                        # di = subplotconf['ndslice'][0]
                        # dj = subplotconf['ndslice'][1]
                        # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][di, dj, :, -1]
                        # ink = subplotconf['input'][0]
                        plotdata_cand = myt(input_ink['val'])[subplotconf['ndslice'][0]]
                        # print "%s[%d]-%s.step plotdata_cand.shape = %s, ndslice = %s, shape = %s, xslice = %s, yslice = %s" % (self.cname, self.cnt, self.id, plotdata_cand.shape, subplotconf['ndslice'], subplotconf['shape'], xslice, yslice)
                        # print "plotdata_cand", plotdata_cand
                    else:
                        try:
                            # plotdata_cand = myt(self.inputs[subplotconf['input'][0]]['val'])[xslice,yslice]
                            plotdata_cand = myt(input_ink['val'])[xslice,yslice]
                        except Exception as e:
                            print(self.cname, self.id, self.cnt, self.inputs, subplotconf['input'])
                            # print "%s[%d]-%s.step, inputs = %s, %s " % (self.cname, self.cnt, self.id, self.inputs[subplotconf['input']][0].shape)
                            print(e)
                    #                                         self.inputs[subplotconf['input']][0])
                    # print "plotdata_cand", plotdata_cand.shape

                    ################################################################################
                    # digitize a random sample (continuous arguments, continuous values)
                    # to an argument grid and average the values
                    # FIXME: to separate function
                    if 'digitize' in subplotconf:
                        argdims = subplotconf['digitize']['argdims']
                        numbins = subplotconf['digitize']['numbins']
                        valdims = subplotconf['digitize']['valdim']

                        # print "%s.plot_subplots(): digitize argdims = %s, numbins = %s, valdims = %s" % (self.cname, argdims, numbins, valdims)
                        
                        # plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims)
                        plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims, f_fval = np.mean)
                    plotdata = {}

                    # if we're dimstacking, now is the time
                    if 'dimstack' in subplotconf:
                        plotdata['i_%d_%d' % (i, j)] = dimensional_stacking(plotdata_cand, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                        # print "plotdata[" + 'i_%d_%d' % (i, j) + "].shape", plotdata['i_%d_%d' % (i, j)].shape
                        # print "%s.plot_subplots(): dimstack x = %s, y = %s" % (self.cname, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                    else:
                        plotdata['i_%d_%d' % (i, j)] = plotdata_cand.reshape(subplotconf['shape'][0])
                        
                    if 'ylog' in subplotconf:
                        # plotdata['i_%d_%d' % (i, j)] = np.log(plotdata['i_%d_%d' % (i, j)] + 1.0)
                        # print plotdata['i_%d_%d' % (i, j)]
                        yscale = 'log'
                    else:
                        yscale = 'linear'
                    if 'bus' in self.inputs[subplotconf['input'][0]]:
                        plotvar = self.inputs[subplotconf['input'][0]]['bus']
                    else:
                        plotvar = input_ink # 'const%d-%d' % ()
                    
                    plotlen = input_ink['shape'][-1] # numsteps at shape[-1]

                    title = "img plot"
                    if 'title' in subplotconf: title = subplotconf['title']

                    # colorbar = False
                    for kwk in ['colorbar', 'colorbar_orientation', 'cax']:
                        if kwk in subplotconf: kwargs[kwk] = subplotconf[kwk]

                    # get t axis
                    # t = self.get_xaxis(subplotconf, xslice, plotlen, (i, j, k))
                        
                    # for k, ink in enumerate(subplotconf['input']):
                    #     plotdata[ink] = input_ink[0].T[xslice]
                    #     # fix nans
                    #     plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                    #     plotvar += "%s, " % (input_ink[2],)
                    # title += plotvar

                    # colormap
                    if 'cmap' not in subplotconf:
                        subplotconf['cmap'] = 'gray'
                    cmap = plt.get_cmap(subplotconf['cmap'])
                                                                
                    # plot the plotdata
                    for ink, inv in list(plotdata.items()):
                        # FIXME: put the image plotting code into function
                        ax = self.fig.axes[idx]
                        
                        inv[np.isnan(inv)] = -1.0

                        # Linv = np.log(inv + 1)
                        Linv = inv
                        # print "Linv.shape", Linv.shape
                        # print "Linv", np.sum(np.abs(Linv))
                        plotfunc = "pcolorfast"
                        kwargs_ = plot_img(
                            ax = ax, data = Linv, plotfunc = plotfunc, # , ordinate = t
                            vmin = vmin, vmax = vmax, cmap = cmap,
                            title = title, **kwargs)

                    # remember stuff for subplot
                    # self.subplots[i][j].update()
                    # self.subplots[i][j]['ax'] = kwargs_['ax']
                    if 'cax' in kwargs_:
                        self.subplots[i][j]['cax'] = kwargs_['cax']
                    
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
    defaults = {
        # diagonal plot function: sns.kdeplot, uniform_divergence, plt.hist, plt.hexbin, plt.plot
        'plotf_diag':    plt.hist, # partial(uniform_divergence, f = subplotconf_plot),    # sns.kdeplot,
        'plotf_offdiag': plt.hist2d, # partial(uniform_divergence, f = subplotconf_plot), # uniform_divergence,
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # self.saveplot = False
        # self.savetype = 'jpg'
        defaults = {}
        defaults.update(AnalysisBlock2.defaults)
        defaults.update(BaseplotBlock2.defaults)
        defaults.update(self.defaults)
        self.defaults = defaults
        
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.prepare_saveplot()
        
    @decStep()
    def step(self, x = None):
        # print "%s.step inputs: %s"  % (self.cname, self.inputs.keys())

        subplotconf = self.subplots[0][0]
        
        # vector combination
        if 'mode' not in subplotconf:
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
            if 'ndslice' in subplotconf:
                # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                print("      ndslice %s: %s, numslice = %d" % (ink, subplotconf['ndslice'][k], len(subplotconf['ndslice'])))
                plotdata = myt(self.inputs[ink]['val'])
                print("      ndslice plotdata", plotdata.shape)
                plotdata = plotdata[subplotconf['ndslice'][k]]
                print("      ndslice plotdata", plotdata.shape)
            else:
                plotdata = myt(self.inputs[ink]['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))
            print("       ndslice plotdata", plotdata.shape)
            
            # apply shape
            if 'shape' in subplotconf:
                if type(subplotconf['shape']) is list:
                    plotdata_shape = subplotconf['shape'][k]
                else:
                    plotdata_shape = subplotconf['shape']
                print("       ndslice plotshape", plotdata_shape)
            else:
                plotdata_shape = plotdata.T.shape

            plotdata = myt(plotdata).reshape(plotdata_shape)
            print("        shape plotdata", plotdata.shape)
    
            return plotdata    
            
        ivecs = []
        ilbls = []
        for k, ink in enumerate(subplotconf['input']):
            # ivec = myt(self.inputs[ink]['val'])
            ivec = unwrap_ndslice(self, subplotconf, k, ink)
            ivecs.append(ivec)
            ilbls += ['%s%d' % (self.inputs[ink]['bus'], j) for j in range(ivec.shape[0])] # range(self.inputs[ink]['shape'][0])]
            # ilbls.append(ilbl)
        print("ilbls", ilbls)
        
        # ivecs = tuple(myt(self.inputs[ink]['val']) for k, ink in enumerate(subplotconf['input']))
        # for ivec in ivecs:
        #     print "ivec.shape", ivec.shape
        plotdata = {}
        if subplotconf['mode'] in ['stack', 'combine', 'concat']:
            # plotdata['all'] = np.hstack(ivecs)
            for ivec in ivecs:
                self._debug('ivec = %s', ivec.shape)
            plotdata['all'] = np.vstack(ivecs).T

        data = plotdata['all']
        print("data", data)
        
        print("SnsPlotBlock2:", data.shape)
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
        # self.plotf_diag = sns.kdeplot # lw = 3, legend = False
        # self.plotf_diag = subplotconf_plot[0]
        # self.plotf_diag = partial(uniform_divergence, f = subplotconf_plot)        
        self.plotf_diag = partial(plt.hist, histtype = 'step')
        g.map_diag(self.plotf_diag)
        # g.map_diag(plotf)

        # g = g.map_diag(sns.kdeplot, lw=3, legend=False)
        # g.map_offdiag(plotf) #, cmap="gray", gridsize=40, bins='log')
        # g.map_offdiag(plt.hexbin, cmap="gray", gridsize=40, bins="log")
        # g.map_offdiag(plt.histogram2d, cmap="gray", bins=30)
        # g.map_offdiag(plt.plot, linestyle = "None", marker = "o", alpha = 0.5) # , bins="log");
        # self.plotf_offdiag = subplotconf_plot[0]
        self.plotf_offdiag = partial(plt.hexbin, cmap='gray', gridsize=40, bins='log')
        g.map_offdiag(self.plotf_offdiag)

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
        
                    
