from features import mfcc
import scipy.io.wavfile as wav

from matplotlib import pyplot as plt
import numpy as np

from scipy import signal
import os

import cPickle



#constants
sampleRate = 22050
windowMS = 20
frameSize = sampleRate*windowMS//(1000)
overlapMs = 10
hopSize = sampleRate*overlapMs//(1000)
CleanedCorpus = "/Users/varunnayyar/Documents/Speech Stuff/LDC Cleaned"


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


def audioFileReader():
	print "Ready to scrape"

	i=0
	for emotion in emotionList:
		emotionDir = os.listdir(os.path.join(CleanedCorpus, emotion))

		try:
			emotionDir.remove('.DS_Store')
		except ValueError:
			print "No DS_Store, this try block was needed"
			# Nothing to do
		print "Currently doing: ", emotion

		mfccDict = {}

		for speechFile in emotionDir:
			speakerID = speechFile.lstrip(emotion)[:3] #grab first 3 letters
			fileNum = speechFile.lstrip(emotion+speakerID).lstrip("_").rstrip('.wav')
			# print speakerID, fileNum, speechFile
			fileloc = os.path.join(CleanedCorpus, emotion, speechFile)
			

			rate, sig = wav.read(fileloc)
			sig=sig.astype(np.float32)
			sig= 0.5*(sig.T[0]+sig.T[1]) # do a simple mix

			mfcc_feat = mfcc(sig,rate, winlen = 0.02, winstep = 0.01, nfilt=17, ceplifter = 22, appendEnergy = True)

			try:
				mfccDict[speakerID].append(mfcc_feat)
			except KeyError:
				mfccDict[speakerID] = [mfcc_feat]

		
		for k,v in mfccDict.iteritems():
			fileName = os.path.join("../MFCCData", emotion+"_"+k)
			print "Writing to: ", fileName
			with open(fileName, 'w') as f:
				cPickle.dump(v, f)




def main():
	audioFileReader()








if __name__ == '__main__':
	main()