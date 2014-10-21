#Wave  reader


import os
import wave
Boreddir = "/Users/varunnayyar/Documents/Speech Stuff/LDC Cleaned/Sad"

for files in os.listdir(Boreddir):
	if not files.endswith(".wav"): 
		continue
	# print os.path.join(Boreddir, files)
	f =  wave.open(os.path.join(Boreddir, files))
	print f.getnchannels(), f.getframerate()
	# print "hi"	