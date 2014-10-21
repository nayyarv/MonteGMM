from features import mfcc
import scipy.io.wavfile as wav

from matplotlib import pyplot as plt
import numpy as np

from scipy import signal
import os




#constants
sampleRate = 22050
windowMS = 20
frameSize = sampleRate*windowMS//(1000)
overlapMs = 10
hopSize = sampleRate*overlapMs//(1000)


try:
	emotionList = os.listdir(CleanedCorpus)
except OSError:
	#Wrong location
	print "Fix location"
	exit()

try:
	emotionList.remove('.DS_Store')
except ValueError:
	# It didn't need to be removed
	assert(len(emotionList)==5)


def audioFileYielder():
	print "Ready to scrape"

	i=0
	for emotion in emotionList:
		emotionDir = os.listdir(os.path.join(CleanedCorpus, emotion))

		try:
			emotionDir.remove('.DS_Store')
		except ValueError:
			print "No DS_Store, this try block was needed"
			# Nothing to do

		for speechFile in emotionDir:
			speakerID = speechFile.lstrip(emotion)[:3] #grab first 3 letters
			fileNum = speechFile.lstrip(emotion+speakerID).lstrip("_").rstrip('.wav')
			# print speakerID, fileNum, speechFile
			# print os.path.join(CleanedCorpus, emotion, speechFile)
			

			yield speakerID, fileNum, audio



def main():
	w = Windowing(type = 'hann')	
	spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
	mfcc = MFCC(inputSize= hopSize+1, numberBands = 17, sampleRate = sampleRate)

	for speakerID, fileNum, audio in audioFileYielder():
		#filter audio
		preEmphaAudio = signal.lfilter([1 ,-1], [1], audio)

		MFCCList = [] #Empty List for each file
		
		for frame in FrameGenerator(preEmphaAudio, frameSize = frameSize, hopSize = hopSize):

			if frame.var()>=0.001: #threshold
				mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
				MFCCList.append(mfcc_coeffs)

		#Now store the list somewhere









if __name__ == '__main__':
	main()