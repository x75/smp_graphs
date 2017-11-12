import numpy as np

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2

# essentia
HAVE_ESSENTIA = False
try:
    import essentia as e
    import essentia.standard as estd
except ImportError, e:
    print "Failed to import essentia and essentia.standard", e

class EssentiaBlock2(PrimBlock2):
    """EssentiaBlock2 class

    Compute :mod:`essentia` features on input data
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
    
        self.pool = e.Pool()
        self.pool2 = []
        self.w = estd.Windowing(type = 'hamming')
        # FFT() would return the complex FFT, here we just want the magnitude spectrum        
        self.spectrum = estd.Spectrum()
        self.mfcc = estd.MFCC()
        self.danceability = estd.Danceability(
            maxTau = 10000, minTau = 300, sampleRate = self.samplerate)
        
    @decStep()
    def step(self, x = None):
        # print "MomentBlock2"
        for k, v in self.inputs.items():
            print "EssentiaBlock2 step[%d] input %s = %s" % (self.cnt, k, v.keys())

            frame_ = v['val']
            print "                        frame_ = %s" % (frame_.shape, )
            
            # spec = self.spectrum(self.w(frame))
            # mfcc_bands, mfcc_coeffs = self.mfcc(spec)

            # print "type(spec)", type(spec)
            # print "spec.shape", spec.shape


            # compute the centroid for all frames in our audio and add it to the pool
            # for frame in estd.FrameGenerator(
            #         frame_, frameSize = 2 * self.samplerate,
            #         hopSize = 1 * self.samplerate):
            for frameidx in range(self.blocksize / self.samplerate):
                frame = frame_[frameidx * self.samplerate:(frameidx + 1) * self.samplerate]
                print "                            frame = %s" % (frame.shape, )
                dreal, ddfa = self.danceability(self.w(frame))
                print "d", dreal # , "frame", frame
                self.pool.add('rhythm.danceability', dreal)
                self.pool2.append(dreal)

            print "rhythm.danceability", type(self.pool['rhythm.danceability']), self.pool['rhythm.danceability'].shape
            self.y = np.atleast_2d(np.array(self.pool2))
            print "y.shape", self.y.shape

            # setattr(self, 'y', spec)
            # setattr(self, 'y', self.pool['rhythm.danceability'])
            # setattr(self, 'y', y)

