# MFCCAlt.py

fileLoc = '/Users/varunnayyar/Documents/Speech Stuff/LDC Cleaned/Bored/BoredCL1_1.wav'

from features.base import mfcc

import scipy.io.wavfile as wav
import numpy as np

(rate, sig) = wav.read(fileLoc)

sig = sig.astype(np.float32)
sig = 0.5 * (sig.T[0] + sig.T[1])

# sig*=5
# print "Sig.dtype: ", sig.dtype
print "Sig.shape", sig.shape
# print "SigLen: ", len(sig)


mfcc_feat = mfcc(sig, rate, winlen=0.02, winstep=0.01, ceplifter=22, appendEnergy=True)
# fbank_feat = logfbank(sig,rate)

print "Mfcc.shape: ", mfcc_feat.shape
print mfcc_feat[1:3]

from matplotlib import pyplot as plt

plt.plot(sig)
plt.show()