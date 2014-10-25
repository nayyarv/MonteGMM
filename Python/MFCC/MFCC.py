import essentia

from essentia.standard import MonoLoader, AudioLoader, FrameGenerator
from essentia.standard import Windowing, Spectrum, MFCC
from matplotlib import pyplot as plt
import numpy as np
# import essentia.streaming


loader = MonoLoader(filename='/Users/varunnayyar/Documents/Speech Stuff/LDC Cleaned/Bored/BoredCC1_1.wav')

try:
    audioChan, sampleRate, nChannels = loader()
    audio = (audioChan.T[0] + audioChan.T[1]) / 2.0
    print audio

except ValueError:
    audio = loader()
    sampleRate = 44100
    nChannels = 1

# audio/=np.abs(audio).max()

print len(audio)
# plt.plot(audio)


i = 0
skippedFrames = 0
windowMS = 20
frameSize = sampleRate * windowMS // (1000)
overalpMs = 10
hopSize = sampleRate * overalpMs // (1000)

w = Windowing(type='square')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC(inputSize=hopSize + 1, numberBands=26, sampleRate=sampleRate)

MFCCList = []

print frameSize, hopSize

for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):

    # print i, (i-1)*hopSize, (i-1)*hopSize + frameSize, frame.var()
    # if frame.var()<0.01*: #threshold detection
    # if not 1:
    skippedFrames += 1

else:  # we have a fram
    # a = w(frame)
    # b = spectrum(a)
    # c = mfcc(b)
    # print i, (i-1)*hopSize, (i-1)*hopSize + frameSize
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    MFCCList.append(mfcc_coeffs)
    # print mfcc_bands
    # print mfcc_coeffs
    # print len(mfcc_bands), len(mfcc_coeffs)

i += 1

print skippedFrames, i
print np.array(MFCCList)[1:3]
# plt.show()


