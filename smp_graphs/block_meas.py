
import numpy as np
from smp_graphs.block import decInit, decStep, Block2, PrimBlock2

class XCorrBlock2(PrimBlock2):
    """Compute cross-correlation functions among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        params = conf['params']
        for i in range(params['outputs']['xcorr'][0][0]):
            # self.params['outputs'][]
            for j in range(params['outputs']['xcorr'][0][1]):
                params['outputs']['xcorr_%d_%d' % (i, j)] = [(1, params['blocksize'])]
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        self.xdim = self.bus[self.inputs['x'][0]].shape[0]
        self.ydim = self.bus[self.inputs['y'][0]].shape[0]
        # print "xcorrblock", self.xdim, self.ydim

    @decStep()
    def step(self, x = None):
        d = {}
        for k in ['x', 'y']:
            d[k] = self.inputs[k][0]
            # print "%s.step d[%s] = %s / %s" % (self.cname, k, d[k].shape, self.inputs[k][0])

        arraytosumraw = np.array([f2(d, k, shift = self.shift, xdim = self.xdim) for k in range(self.ydim)]).reshape((self.ydim, self.xdim, self.shift[1] - self.shift[0]))
        # print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw.shape)
        # for i in range(self.ydim):
        #     print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw[0,i,:])
        # arraytosum = arraytosumraw.reshape((24, 41))
        # thesum = np.sum(arraytosum, axis=0)
        # plotdata = np.log(thesum)

        # fig = makefig(self.xdim, self.ydim)
        for i in range(self.ydim):
            for j in range(self.xdim):
                outk  = 'xcorr_%d_%d' % (i, j)
                outsh = arraytosumraw[i,[j]].shape
                # print "outk = %s, outsh = %s" % (outk, outsh,)
                # self.outputs[outk][0] = arraytosumraw[i,[j]]
                outv = getattr(self, outk)
                outsl = slice(0, self.shift[1] - self.shift[0])
                outv[:,outsl] = arraytosumraw[i,[j]]
                # setattr(self, outk, arraytosumraw[i,[j]])
                # print "self.%s = %s" % (outk, getattr(self, outk))

        #         plotdata = arraytosumraw[i,j]
        #         ax = fig.axes[j+(i*self.xdim)]
        #         ax.plot(range(self.shift[0], self.shift[1]), plotdata, "ko", alpha = 0.5, label = "corr(x[%d], y[%d])" % (j, i))
        #         ax.legend()
        # plt.show()

def f1(d, j, k, shift = (-10, 11)):
    # # this makes an implicit diff on the y signal
    # return np.array([
    #     np.correlate(
    #         np.roll(d['x'].T[1:,j], shift = i),
    #         np.diff(d['y'].T[:,k], axis = 0)) for i in range(shift[0], shift[1])
    #     ])
    # don't do that
    x = d['x'].T[:,j]
    y = d['y'].T[:,k]
    scov = np.std(x) * np.std(y)
    x -= np.mean(x)
    x /= scov
    y -= np.mean(y)
    y /= scov
    corr = np.array([
        np.correlate(np.roll(x, shift = i), y) for i in range(shift[0], shift[1])
    ])/y.shape[0]
    # this is correct if the inputs are the same size
    # print "f1 corr = %s" % (corr,)
    return corr
    
def f2(d, k, shift = (-10, 11), xdim = 1):
    return np.array([f1(d, j, k, shift = shift) for j in range(xdim)])
