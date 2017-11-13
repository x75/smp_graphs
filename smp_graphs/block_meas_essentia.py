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
        self.pool2 = {}
        self.w = estd.Windowing(type = 'hamming')
        
        # FFT() would return the complex FFT, here we just want the magnitude spectrum
        self.spectrum = estd.Spectrum()
        
        for outk, outv in self.outputs.items():
            if not outv.has_key('etype'): continue
            if outv['etype'] == 'mfcc':
                outv['func'] = estd.MFCC()
            elif outv['etype'] == 'centroid':
                outv['func'] = estd.Centroid()
            elif outv['etype'] == 'sbic':
                outv['func'] = estd.SBic()
            elif outv['etype'] == 'danceability':
                outv['func'] = estd.Danceability(
                    maxTau = 10000, minTau = 300, sampleRate = self.samplerate)
        
    @decStep()
    def step(self, x = None):
        # print "MomentBlock2"
        
        # # silently assuming only one item named 'x'
        # for k, v in self.inputs.items():
        #     pass

        # inputs
        print "EssentiaBlock2 step[%d] input %s = %s" % (
            self.cnt, 'x', self.inputs['x'].keys())
        frame_ = self.inputs['x']['val']

        spec = []
        for frame in estd.FrameGenerator(
                frame_, frameSize = 2 * self.samplerate,
                hopSize = 1 * self.samplerate):
            # common funcs
            spec_ = self.spectrum(self.w(frame))
            spec.append(spec_)
        self.spec = np.array(spec)
            
        # outputs
        for outk, outv in self.outputs.items():
            print "EssentiaBlock2 step[%d] output %s = %s" % (
                self.cnt, outk, outv.keys())

            # frame_ = v['val']
            # print "                        frame_ = %s" % (frame_.shape, )
            
            # spec = self.spectrum(self.w(frame))
            # mfcc_bands, mfcc_coeffs = self.mfcc(spec)

            # print "type(spec)", type(spec)
            # print "spec.shape", spec.shape


            # compute the centroid for all frames in our audio and add it to the pool
            # for frame in estd.FrameGenerator(
            #         frame_, frameSize = 2 * self.samplerate,
            #         hopSize = 1 * self.samplerate):
            
            # for frameidx in range(self.blocksize / self.samplerate):
            #     frame = frame_[frameidx * self.samplerate:(frameidx + 1) * self.samplerate]
            outtypek = '%s-%s' % (outk, outv['etype'])
            self.pool2[outtypek] = []
            print "    spec.shape", self.spec.shape
            for spec_ in self.spec:
                frame = spec_
                print "                            frame = %s" % (frame.shape, )
                # dreal, ddfa = self.danceability(self.w(frame))
                # print "d", dreal # , "frame", frame
                # self.pool.add('rhythm.danceability', dreal)
                # self.pool2.append(dreal)
                # self.pool.add('rhythm.danceability', dreal)
                y_ = outv['func'](frame)
                if outv['etype'] == 'mfcc':
                    y_ = y_[1][1:]
                # print "        EssentiaBlock2 compute", y_
                self.pool2[outtypek].append(y_)

            # print "rhythm.danceability", type(self.pool['rhythm.danceability']), self.pool['rhythm.danceability'].shape

            y_ = np.atleast_2d(np.array(self.pool2[outtypek]))
            if outv['etype'] == 'mfcc':
                y_ = y_.T

            setattr(self, outk, y_.copy())
            print "%s/%s = %s" % (self.id, outk, getattr(self, outk))
            # setattr(self, 'y', self.pool['rhythm.danceability'])
            # setattr(self, 'y', y)

