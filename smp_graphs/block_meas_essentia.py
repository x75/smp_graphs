"""wrap essentia in blocks

this is an ad-hoc sketch for a particular processing pipeline, wip
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# essentia
# FIXME: try selective imports of only those modules needed
HAVE_ESSENTIA = False
try:
    import essentia as e
    import essentia.standard as estd
    # import essentia.streaming as estd
    HAVE_ESSENTIA = True
except ImportError as e:
    print("Failed to import essentia and essentia.standard", e)

from smp_graphs.block import decInit, decStep, Block2, PrimBlock2
from smp_graphs.block_ols import FileBlock2

class eFileBlock2(FileBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FileBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def init_load_file(self, filetype, lfile, conf):
        if filetype == 'mp3':

            # default samplerate
            if 'samplerate' not in conf['params']['file']:
                conf['params']['file']['samplerate'] = 44100

            # load data
            loader = estd.MonoLoader(filename = lfile, sampleRate = conf['params']['file']['samplerate'])
            data = loader.compute()

            # if not length is given, create random slice of 60 sec minimal length if file length allows
            if 'length' not in conf['params']['file'] or conf['params']['file']['length'] is None or conf['params']['file']['length'] == 0:
                conf['params']['file']['length'] = min(
                    data.shape[0],
                    np.random.randint(
                        conf['params']['file']['samplerate'] * 60,
                        data.shape[0] - conf['params']['file']['offset']))

            # compute slice
            sl = slice(conf['params']['file']['offset'], conf['params']['file']['offset'] + conf['params']['file']['length'])
            print("%sFileBlock2-%s fileypte mp3 sl = %s" % (self.nesting_indent, self.id, sl, ))
            # select data
            self.data = {'x': data[sl]} # , 'y': data[sl]}
            print("%sFileBlock2-%s data = %s, self.data['x'] = %s" % (self.nesting_indent, self.id, data.shape, self.data['x'].shape))
            # set step callback
            self.step = self.step_wav
    
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
        
        for outk, outv in list(self.outputs.items()):
            if 'etype' not in outv: continue
            if outv['etype'] == 'mfcc':
                outv['func'] = estd.MFCC(numberCoefficients = 13)
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
        # print "EssentiaBlock2 step[%d] input %s = %s" % (
        #     self.cnt, 'x', self.inputs['x'].keys())
        frame_ = self.inputs['x']['val']
        print("EssentiaBlock2 step[%d] input.val = %s" % (
            self.cnt, self.inputs['x']['val'].shape))

        spec = []
        for frame in estd.FrameGenerator(
                frame_,
                frameSize = 2 * self.samplerate,
                hopSize = 1 * self.samplerate,
                startFromZero = True):
            # common funcs
            spec_ = self.spectrum(self.w(frame))
            # print "    EssentiaBlock2 from frame %s computing spectrum %s" % (frame.shape, spec_.shape)
            spec.append(spec_)
        self.spec = np.array(spec)

        print("   EssentiaBlock2 spectrum = %s" % (self.spec.shape, ))
            
        # outputs
        for outk, outv in list(self.outputs.items()):
            print("    EssentiaBlock2 step[%d] output %s = %s" % (
                self.cnt, outk, list(outv.keys())))

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
            # print "    spec.shape", self.spec.shape
            for spec_ in self.spec:
                frame = spec_
                # print "                            frame = %s" % (frame.shape, )
                # dreal, ddfa = self.danceability(self.w(frame))
                # print "d", dreal # , "frame", frame
                # self.pool.add('rhythm.danceability', dreal)
                # self.pool2.append(dreal)
                # self.pool.add('rhythm.danceability', dreal)
                y_ = outv['func'](frame)
                if outv['etype'] == 'mfcc':
                    # y_ = y_[1][1:]
                    y_ = y_[1]
                # print "        EssentiaBlock2 compute", y_
                self.pool2[outtypek].append(y_)

            # print "rhythm.danceability", type(self.pool['rhythm.danceability']), self.pool['rhythm.danceability'].shape

            y_ = np.atleast_2d(np.array(self.pool2[outtypek]))
            if outv['etype'] == 'mfcc':
                y_ = y_.T

            setattr(self, outk, y_.copy())
            print("    %s/%s = %s" % (self.id, outk, getattr(self, outk).shape))
            # setattr(self, 'y', self.pool['rhythm.danceability'])
            # setattr(self, 'y', y)

class AdhocMixBlock2(PrimBlock2):
    """AdhocMixBlock2 class

    Custom ad-hoc computation block
     - input is feature matrix, first half of rows is start-features, second half of rows is end-features
     - compute 1D t-SNE on the entire matrix
     - plot end and start
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # self.tsne = TSNE(n_components = 1, metric = 'correlation')
        # self.tsne = TSNE(n_components = 1, metric = 'l2')
        self.tsne = TSNE(n_components = 1, metric = 'l2')
        
    @decStep()
    def step(self, x = None):
        print("adhoc inputs", self.inputs['mus']['val'].shape)
        x = self.inputs['mus']['val']
        # print "x", x
        x_zeromean = x - np.mean(x, axis = 0)
        x_std = x_zeromean / np.std(x_zeromean, axis = 0)

        print("x_std", x_std)
        
        # x_emb = self.tsne.fit_transform(x_std)
        x_emb = x
        x_emb[:,1] = 0
        x_emb_start = x_emb[:x.shape[0]/2]
        x_emb_end   = x_emb[x.shape[0]/2:]
        
        points = []
        points_start = []
        d_mat = {
            'd_l2': np.zeros((x.shape[0]/2, x.shape[0]/2)),
            'd_corr': np.zeros((x.shape[0]/2, x.shape[0]/2)),
        }
            
        for i, x_e in enumerate(x_emb_end):
            for j, x_s in enumerate(x_emb_start):
                # if j == i: continue
                # points.append(np.hstack([x_e, x_s]))
                d_l2 = np.linalg.norm(x_e - x_s)
                # d_corr = np.correlate(x_e, x_s)
                d_corr = 1 - np.corrcoef(x_e, x_s)[0,1]
                d_mat['d_l2'][i,j] = d_l2
                d_mat['d_corr'][i,j] = d_corr
                points.append(d_corr)
                points_start.append((i, j))

        print("adhoc inputs x_emb_start", x_emb_start)
        print("adhoc inputs x_emb_end", x_emb_end)

        # write mix sequence
        trk_cnt = 0
        trk_visited = []
        trk_do = True
        # trk_mat = d_mat['d_corr'].copy()
        trk_mat = d_mat['d_l2'].copy()
        
        trk = np.random.randint(x.shape[0]/2)
        trk_mat[:,trk] = 1000
        print("trk_mat", trk_mat)
        while trk_do:
            trk_visited.append((trk, self.filearray[trk][0]))
            if len(trk_visited) == x.shape[0]/2:
                trk_do = False
            # trk_ = np.argmin(d_mat['d_corr'][trk,:])
            trk_ = np.argmin(trk_mat[trk,:])
            trk_mat[:,trk_] = 1000
            trk = trk_
            print("trk_mat", trk_mat)

        trk_seq = "trk_seq = [\n"
        for seqnum, seqfile in trk_visited:
            trk_seq += "    (%d, '%s'),\n" % (seqnum, seqfile)
        trk_seq += "]\n"
        
        print("trk_visited", trk_seq)
        f = open('trk_seq_%d.txt' % np.random.randint(1000), 'w')
        f.write(trk_seq)
        f.flush()
        f.close()
        print("trk_mat", trk_mat)
            
        # setattr(self, outk, y_.copy())

        plt.ion()

        fig1 = plt.figure()
        ax0 = fig1.add_subplot(1,3,1)
        # ax0.plot(x_std.T, 'ko')
        ax0.boxplot(x_std)
        
        ax1 = fig1.add_subplot(1,3,2)
        ax1.imshow(d_mat['d_corr'], interpolation = 'none', cmap = plt.get_cmap('Reds'))

        ax2 = fig1.add_subplot(1,3,3)
        ax2.imshow(d_mat['d_l2'], interpolation = 'none', cmap = plt.get_cmap('Reds'))
        
        plt.draw()
        plt.pause(1e-9)
        
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(1,1,1)
        # ax1.plot(x_emb_start, np.zeros_like(x_emb_start), 'ko')
        # ax1.plot(x_emb_end, np.zeros_like(x_emb_end), 'ro')
        # plt.draw()
        # plt.pause(1e-9)

        # # points = zip(x_emb_end, x_emb_start)
        # # points = np.hstack((x_emb_end, x_emb_start))
        
        # points = np.array(points)
        # print "points", points

        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(1,1,1)
        # # plt.plot(points, 'ko')
        # ax2.scatter(points[:,0], points[:,1], alpha = 0.5)
        # for i, p in enumerate(points):
        #     ax2.text(p[0], p[1], '%d/%d' % (points_start[i][0], points_start[i][1]))
        # plt.draw()
        # plt.pause(1e-9)
        
