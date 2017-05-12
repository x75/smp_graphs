
import numpy as np

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2



class XCorrBlock2(PrimBlock2):
    """Compute cross-correlation functions among all variables in dataset"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        d = {}
        for k in ['x', 'y']:
            d[k] = self.inputs[k][0]
            print "%s.step d[%s] = %s" % (self.cname, k, d[k].shape)

        arraytosumraw = np.array([
            np.array([
                np.abs(np.array([
                    np.correlate(
                        np.roll(d['x'][1:,j], shift = i),
                        np.diff(d['y'][:,k], axis = 0)) for i in range(-20, 21)
                ])) for j in range(6)]) for k in range(4)
            ])
        print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw.shape)
        print "%s.step arraytosumraw.sh = %s" % (self.cname, arraytosumraw[0,0,:])
        arraytosum = arraytosumraw.reshape((24, 41))
        thesum = np.sum(arraytosum, axis=0)
        plotdata = np.log(thesum)
        plt.plot(plotdata, "ko", alpha = 0.5)
        plt.show()
