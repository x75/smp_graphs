"""**smp_graphs** measure blocks

.. moduleauthor:: Oswald Berthold

.. warning:: Work in progress

Measures are quantitative expressions of some quality and its amount
of presence in the input data. A measure operator can be unary like
mean() and var(), binary like a distance measure d(x1, x2), or n-ary
like an area spanned by vertices.

.. note::

    Unary measures can also be seen as a special case of a binary
    operator applied to the same object in both arguments. Doing this
    a binary measure becomes an 'auto' version of the original measure
    like cross-correlation corr(x1, x2) can become auto-correlation
    with corr(x1, x1)

.. note::

    Opposite of distance is similarity (closeness, proximity, difference, ...)

Currently the measure stack consists of:
 - the xcorr scan measurements defined here, TODO move to smp_base
 - the infth scan measurements defined in block_meas_infth.py and smp_base/measures_infth.py
 - direct: MSEBlock2, MomentBlock2
 - TFBlock2 approach: measure as basis transforms, DFT style
 - indirect: FuncBlock2, ModelBlock2
 - legacy and external stuff in: evoplast, smp, ...
 - check [Diss_Draft, thesis_smp, ...] for measure notes

Plan:
 - consolidate the stack
 - make general matrix scan pattern to fill with primitive measure callback

"""

import numpy as np

from smp_base.common import get_module_logger

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2

from logging import DEBUG as LOGLEVEL
logger = get_module_logger(modulename = 'block_meas', loglevel = LOGLEVEL)

def compute_xcor_matrix_src_dst(data, dst, src, shift = (-10, 11)):
    """compute_xcor_matrix_src_dst

    Compute the cross-correlation function for one source/destination pair. The destination is shifted by convention.
    """
    # # this makes an implicit diff on the y signal
    # return np.array([
    #     np.correlate(
    #         np.roll(data['x'].T[1:,dst], shift = i),
    #         np.diff(data['y'].T[:,src], axis = 0)) for i in range(shift[0], shift[1])
    #     ])
    # don't do that
    x = data['x'].T[:,dst]
    y = data['y'].T[:,src]
    # print "corr compute_xcor_matrix_src_dst shapes", x.shape, y.shape
    assert len(x.shape) == 1 and len(y.shape) == 1
    # normalization
    len_inv  = 1.0/x.shape[0] # 1 / N
    x = (x - np.mean(x)) / (np.std(x) * x.shape[0]) # 1 / N applied once, scaled by stddev
    y = (y - np.mean(y)) / np.std(y) # scaled by stddev
    # compute correlation
    corr = np.array([
        np.correlate(np.roll(x, shift = i), y) for i in range(shift[0], shift[1])
    ])
    return corr
    
def compute_xcor_matrix_src(data = {}, src = 0, shift = (-10, 11), dst_dim = 1):
    # compute the cross-correlation matrix for one source looping over destinations (sensors)
    return np.array([compute_xcor_matrix_src_dst(data = data, dst = j, src = src, shift = shift) for j in range(dst_dim)])

def compute_xcor_matrix(data = {}, shift = (-1, 0), src_dim = 1, dst_dim = 1):
    # np.array([compute_xcor_matrix_src(d, k, shift = self.shift, xdim = self.xdim) for k in range(self.ydim)]).reshape((self.ydim, self.xdim, self.shift[1] - self.shift[0]))
    return np.array([compute_xcor_matrix_src(data, src = k, shift = shift, dst_dim = dst_dim) for k in range(src_dim)]).reshape((src_dim, dst_dim, shift[1] - shift[0]))

class XCorrBlock2(PrimBlock2):
    """XCorrBlock2

    Compute cross-correlation functions all pairs of variables in dataset and a given timeshift

    Arguments:
        - conf: block configuration dict
        - paren: block's parent reference
        - graph's top node reference

    Sub-Arguments:
        - conf['params']
            - id: some id string
            - blocksize: number of timesteps between block computations
            - inputs: input dict
            - shift: timeshift start and stop, shift is applied to last axis of input tensor
            - outputs: output dict
    """
    def __init__(self, conf = {}, paren = None, top = None):
        # # FIXME: legacy        
        # params = conf['params']
        # for i in range(params['outputs']['xcorr'][0][0]):
        #     # self.params['outputs'][]
        #     for j in range(params['outputs']['xcorr'][0][1]):
        #         params['outputs']['xcorr_%d_%d' % (i, j)] = [(1, params['blocksize'])]

        # super init
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # shortcut to variable dimensions params
        self.xdim = self.bus[self.inputs['x']['bus']].shape[0]
        self.ydim = self.bus[self.inputs['y']['bus']].shape[0]

    @decStep()
    def step(self, x = None):
        """XCorrBlock2.step

        Arguments:
            - x: legacy, None is ok
        """
        # init dict
        d = {}
        
        # predefined cross correlation inputs are x and y
        for k in ['x', 'y']:
            d[k] = self.inputs[k]['val']
            # print "%s.step d[%s] = %s / %s" % (self.cname, k, d[k].shape, self.inputs[k][0])

        # compute the entire cross-correlation matrix looping over sources (motors)
        arraytosumraw = compute_xcor_matrix(data = d, shift = self.shift, src_dim = self.ydim, dst_dim = self.xdim)
        
        # print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw.shape)
        # for i in range(self.ydim):
        #     print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw[0,i,:])
        # arraytosum = arraytosumraw.reshape((24, 41))
        # thesum = np.sum(arraytosum, axis=0)
        # plotdata = np.log(thesum)

        # prepare required output shape
        # print "arraytosum.sh", arraytosumraw.shape
        print("xcorr.shape", self.xcorr.shape)
        self.xcorr = arraytosumraw.reshape(self.outputs['xcorr']['shape']) # + (1,))
        print("xcorr.shape", self.xcorr.shape) #, self.xcorr
        
        # # fig = makefig(self.xdim, self.ydim)
        # for i in range(self.ydim):
        #     for j in range(self.xdim):
        #         outk  = 'xcorr_%d_%d' % (i, j)
        #         outsh = arraytosumraw[i,[j]].shape
        #         # print "outk = %s, outsh = %s" % (outk, outsh,)
        #         # self.outputs[outk][0] = arraytosumraw[i,[j]]
        #         outv = getattr(self, outk)
        #         outsl = slice(0, self.shift[1] - self.shift[0])
        #         outv[:,outsl] = arraytosumraw[i,[j]]
        #         # setattr(self, outk, arraytosumraw[i,[j]])
        #         # print "self.%s = %s" % (outk, getattr(self, outk))

        #         plotdata = arraytosumraw[i,j]
        #         ax = fig.axes[j+(i*self.xdim)]
        #         ax.plot(range(self.shift[0], self.shift[1]), plotdata, "ko", alpha = 0.5, label = "corr(x[%d], y[%d])" % (j, i))
        #         ax.legend()
        # plt.show()

class WindowedBlock2(PrimBlock2):
    """WindowedBlock2 class

    Uniform random numbers: output is uniform random vector
    FIXME: this one is not really needed as sliding window just comes out of blocksize vs. output blocksize
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        for k, v in list(self.inputs.items()):
            self._debug("%s.step[%d] k = %s, v = %s/%s, bus = %s" % (self.cname, self.cnt, k, v['val'].shape, v['shape'], self.bus[v['bus']].shape))
            setattr(self, k, np.abs(TFBlock2.step_fft({k: v['val']})[k.upper()]) / float(self.blocksize))
            # print "step_fft", getattr(self, k).shape
        
class TFBlock2(PrimBlock2):
    @staticmethod
    def step_fft(data = {}):
        DATA = {}
        for k, v in list(data.items()):
            DATA[k.upper()] = np.fft.fft(a = v)
        return DATA

################################################################################
# good old plain measures: MSE, \int MSE, statistical moments, ...
from smp_base.measures import measures as measures_available
from smp_base.measures import meas_mse, meas_hist, div_kl, div_chisquare
# from smp_base.measures import meas_sub

# MSE
class MSEBlock2(PrimBlock2):
    """MSEBlock2 class

    Compute mean squared error between inputs 'x' and 'x_' over window of size 'winsize'
    """
    defaults = {
        'inputs': {
            'blocksize': 1,
            'x': {'shape': (1, 1), 'val': np.zeros((1, 1))},
            'x_': {'shape': (1, 1), 'val': np.zeros((1, 1))},
            }
        }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        # for k, v in self.inputs.items():
        x = self.inputs['x']['val'].T
        x_ = self.inputs['x_']['val'].T
        # print "MSEBlock2.step[%d] x,x_ = %s,%s" % (self.cnt, x.shape, x_.shape)
        setattr(self, 'y', meas_mse(x = x, x_ = x_).T)
        # print "MSEBlock2.step[%d] y = %s" % (self.cnt, self.y)
    
# statistical moments: mean, var, kurt, entropy, min, max
class MomentBlock2(PrimBlock2):
    """MomentBlock2 class

    Compute statistical moments: mean, var, min, max
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        # print "MomentBlock2"
        for k, v in list(self.inputs.items()):
            logstr = "%s.step[%d] k = %s, v = %s/%s, bus = %s" % (self.cname, self.cnt, k, v['val'].shape, v['shape'], self.bus[v['bus']].shape)
            self._debug(logstr)
            data = v['val']
            axis = -1
            # print "data", data.shape
            k_mu = k + "_mu"
            k_var = k + "_var"
            k_min = k + "_min"
            k_max = k + "_max"
            if hasattr(self, 'transpose') and self.transpose:
                setattr(self, k_mu, np.mean(v['val'], axis = axis, keepdims = True).T)
                setattr(self, k_var, np.var(v['val'], axis = axis, keepdims = True).T)
                setattr(self, k_min, np.min(v['val'], axis = axis, keepdims = True).T)
                setattr(self, k_max, np.max(v['val'], axis = axis, keepdims = True).T)
            else:
                setattr(self, k_mu, np.mean(v['val'], axis = axis, keepdims = True))
                setattr(self, k_var, np.var(v['val'], axis = axis, keepdims = True))
                setattr(self, k_min, np.min(v['val'], axis = axis, keepdims = True))
                setattr(self, k_max, np.max(v['val'], axis = axis, keepdims = True))
            logstr = "%s%s-%s[%d/%d]\n%s    %s mu = %s, var = %s, min = %s, max = %s" % (
                self.nesting_indent, self.cname, self.id, self.cnt, self.top.blocksize_min,
                self.nesting_indent, k,
                getattr(self, k_mu), getattr(self, k_var), getattr(self, k_min), getattr(self, k_max)
            )
            self._debug(logstr)

class MeasBlock2(PrimBlock2):
    """MeasBlock2 - measure block

    MeasBlock2 is a generic container for measures, see the module
    description in :file:`smp_base/measures.py`.

    The idea is that the MeasBlock2 operation is governed by the
    `mode`, `scope` and `meas` attributes.

    Modes are: primitive (single measure), scan (multiple measures),
    histogram (vector quant), basis transforms, ...

    Scopes are: local (component-wise) and global (summed over all
    axes). Using parametric embedding / convolution / summation-rules,
    local-to-global can be made into a continuous spectrum.

    Measures are: all common distance metrics (error, manhattan,
    euclid, l1, l2, linf, min, max, cosine, ...), histo and
    probabilistic distance metrics (KLD, chi-square, EMD, mahalanobis,
    ...), information theoretic (entropic) measures, basis transforms,
    ...
    """
    modes = {
        'basic': {},
        'hist': {},
    }
    # use available measures from smp_base.measures
    measures = measures_available
    # measures = {
    #     'sub': {'func': np.subtract},
    #     'mse': {'func': meas_mse},
    #     'hist': {'func': meas_hist}, # compute histogram
    #     'kld':  {'func': div_kl},
    #     'chisq':  {'func': div_chisquare},
    # }
    output_modifiers = {
        'proba': '_p',
        'value': '_x',
    }
    
    defaults = {
        'inputs': {
            'x1': {'shape': (1, 1)},
            'x2': {'shape': (1, 1)},
        },
        'outputs': {
            'y': {'shape': (1, 1)}
        },
        'mode': 'basic',
        'scope': 'local',
        'meas': 'mse',
        'bins': 21, # 'auto',
    }

    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # default step is basic
        self._step = self.step_basic

        # mode specific setup
        if self.mode == 'basic':
            if type(self.meas) is list:
                for meas in self.meas:
                    # create outputs for measures
                    pass
        elif self.mode == 'hist':
            self._step = self.step_hist
            # fix outputs
            # self.outputs
            if type(self.bins) is int:
                self.bins = np.linspace(-1.1, 1.1, self.bins + 1)
        # mode specific setup
        elif self.mode == 'div':
            self._step = self.step_div
        
        # FIXME: mangle conf for input/output dimension inference
        
    @decStep()
    def step(self, x = None):
        self._step(x)
        
    def step_basic(self, x = None):
        
        # # print "MeasBlock2"
        # for k, v in self.inputs.items():
        #     logstr = 'k = {0}, v = {1}'.format(k, v)
        #     self._debug(logstr)

        x1 = self.get_input('x1').astype(np.float)
        x2 = self.get_input('x2').astype(np.float)
        self._debug('self.measures     is type = %s, length = %s' % (type(self.measures), len(self.measures)))
        self._debug('self.measures[%s] is type = %s, length = %s' % (self.meas, type(self.measures[self.meas]), len(self.measures[self.meas])))
        self._debug('calling %s on (x1 = %s, x2 = %s)' % (self.measures[self.meas]['func'], x1.shape, x2.shape))
        self._debug('               x1 = %s' % (x1, ))
        self._debug('               x2 = %s' % (x2, ))
        setattr(self, 'y', self.measures[self.meas]['func'](x1, x2))
        
        self._debug('y = measures[%s](x1, x2) = %s' % (self.meas, str(self.y)[:300]))

    def step_div(self, x = None):
        x1_p = self.get_input('x1_p').astype(np.float)
        x2_p = self.get_input('x2_p').astype(np.float)
        x1_x = self.get_input('x1_x').astype(np.float)
        x2_x = self.get_input('x2_x').astype(np.float)
        
        self._debug('self.measures     is type = %s, length = %s' % (type(self.measures), len(self.measures)))
        self._debug('self.measures[%s] is type = %s, length = %s' % (self.meas, type(self.measures[self.meas]), len(self.measures[self.meas])))
        self._debug('    step_div calling %s on (x1 = %s, x2 = %s)' % (self.measures[self.meas]['func'], x1_p.shape, x2_p.shape))
        self._debug('               x1 = %s' % (x1_p, ))
        self._debug('               x2 = %s' % (x2_p, ))
        
        if len(x1_x.shape) > 1: x1_x = x1_x[0,:]
        if len(x2_x.shape) > 1: x2_x = x2_x[0,:]
        assert len(x1_x.shape) == 1, "Assuming 1d bin specs but got %s in MeasBlock2.step_div from block id = %s " % (x1_x.shape, self.id)
        # distmat d(x1_i, x2_j)
        x1_x_ = x1_x
        x2_x_ = x2_x
        if  x1_x.shape[0] != x1_p.shape[0]: # bin limits
            # x1_x_ = x1_x[:-1] + np.mean(np.abs(np.diff(x1_x)))/2.0
            x1_x_ = x1_x[:-1] + np.abs(np.diff(x1_x))/2.0
        if x2_x.shape[0] != x2_p.shape[0]: # bin limits
            # x2_x_ = x2_x[:-1] + np.mean(np.abs(np.diff(x2_x)))/2.0
            x2_x_ = x2_x[:-1] + np.abs(np.diff(x2_x))/2.0
            
        distmat = x1_x_[None,:] - x2_x_[:,None]
        self._debug('    distmat = %s' % (distmat.shape, ))

        div, flow = self.measures[self.meas]['func'](x1_p, x2_p, distmat, flow = True)
        if self.scope == 'local':
            # setattr(self, 'y', np.sum(flow, axis = 0))
            setattr(self, 'y', flow)
            # setattr(self, 'y', np.array(flow[0]))
        else:
            setattr(self, 'y', div)
            
        self._debug('y = measures[%s](x1, x2) = %s' % (self.meas, str(self.y)[:300]))
        
    def step_hist(self, x = None):
        """step for histogram

        Compute the histogram for all input items.

        Inputs:
         - x(ndarray): data

        Params:
         - bins(None, str, int, array-like): None, 'auto', number of bins or precomputed bin array
        """
        self._debug('self.measures     is type = %s, length = %s' % (type(self.measures), len(self.measures)))
        self._debug('self.measures[%s] is type = %s, length = %s' % (self.meas, type(self.measures[self.meas]), len(self.measures[self.meas])))
        for ink, inv in list(self.inputs.items()):
            x = self.get_input(ink)
            self._debug('    calling %s on (x = %s, bins = %s)' % (self.measures[self.meas]['func'], x.shape, self.bins))
            _h = self.measures[self.meas]['func'](x, bins = self.bins)
            # h_ is a tuple (counts, bins)
            self._debug('    _h[0].shape = %s, _h[1].shape = %s, bins = %s' % (_h[0].shape, _h[1].shape, self.bins))
            setattr(self, '%s%s' % (ink, MeasBlock2.output_modifiers['proba']), _h[0])
            setattr(self, '%s%s' % (ink, MeasBlock2.output_modifiers['value']), _h[1])
            self._debug('    _h%s = measures[%s](x, bins) = %s' % (ink, self.meas, str(getattr(self, '%s%s' % (ink, MeasBlock2.output_modifiers['proba'])))[:100]))
            
        # x1 = self.get_input('x1')
        # x2 = self.get_input('x2')
        
        # setattr(self, 'y', self.measures[self.meas]['func'](x1, x2))
        
        
