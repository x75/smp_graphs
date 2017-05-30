
import numpy as np


from smp_graphs.block import decInit, decStep, Block2, PrimBlock2


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
    # scov = np.std(x) * np.std(y)
    # scov_inv = np.sqrt(1.0/scov)
    len_inv  = 1.0/x.shape[0]
    x = (x - np.mean(x)) / (np.std(x) * x.shape[0]) # * scov_inv
    # x /= scov
    y = (y - np.mean(y)) / np.std(y) #  scov_inv
    # y /= scov
    corr = np.array([
        np.correlate(np.roll(x, shift = i), y) for i in range(shift[0], shift[1])
    ])
    # corr_normalized = corr * len_inv
    # this is correct if the inputs are the same size
    # print "compute_xcor_matrix_src_dst corr = %s" % (corr,)
    return corr # _normalized
    
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
        print "xcorr.shape", self.xcorr.shape
        self.xcorr = arraytosumraw.reshape(self.outputs['xcorr']['shape']) # + (1,))
        print "xcorr.shape", self.xcorr.shape
        
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
    """!@brief Uniform random numbers: output is uniform random vector
    FIXME: this one is not really needed as sliding window just comes out of blocksize vs. output blocksize
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        for k, v in self.inputs.items():
            print "%s.step[%d] k = %s, v = %s/%s, bus = %s" % (self.cname, self.cnt, k, v['val'].shape, v['shape'], self.bus[v['bus']].shape)
            setattr(self, k, np.abs(TFBlock2.step_fft({k: v['val']})[k.upper()]) / float(self.blocksize))
            # print "step_fft", getattr(self, k).shape
        
class TFBlock2(PrimBlock2):
    @staticmethod
    def step_fft(data = {}):
        DATA = {}
        for k, v in data.items():
            DATA[k.upper()] = np.fft.fft(a = v)
        return DATA
