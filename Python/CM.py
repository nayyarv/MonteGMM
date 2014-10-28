__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from MFCCArrayGen import emotions as origEmotions
from MFCCArrayGen import speakers

# emotions = origEmotions[1:] + [origEmotions[0]]
emotions = ["Bored", "Happy", "HotAnger", "Neutral", "Sad"]

speakers = speakers[:3]

trueDict = {}
predDict = {}


trueDict["CC1"] = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']
predDict["CC1"] = ['Bored', 'Sad', 'Bored', 'Bored', 'Bored', 'Neutral', 'Bored', 'Bored', 'Neutral', 'Sad', 'Bored', 'Sad', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Neutral', 'Neutral', 'Sad', 'Bored', 'Sad', 'Bored', 'Bored', 'Neutral', 'Bored', 'Sad', 'Neutral', 'Sad', 'Neutral', 'Bored', 'Happy', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'HotAnger', 'HotAnger', 'Sad', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']


trueDict["CL1"] = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']
predDict["CL1"] = ['Sad', 'Neutral', 'Bored', 'Sad', 'Sad', 'Neutral', 'Sad', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Sad', 'Neutral', 'Bored', 'Sad', 'Sad', 'Sad', 'Bored', 'Sad', 'Sad', 'Happy', 'Sad', 'Neutral', 'Happy', 'Happy', 'Sad', 'Happy', 'Sad', 'Sad', 'HotAnger', 'HotAnger', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'Sad', 'Happy']

trueDict["GG1"] = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']
predDict["GG1"] = ['Neutral', 'Neutral', 'Happy', 'Sad', 'Neutral', 'Sad', 'Happy', 'Happy', 'Bored', 'Happy', 'Happy', 'Happy', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Bored', 'Happy', 'Bored', 'Bored', 'HotAnger', 'Sad', 'Happy', 'Bored', 'Happy', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'Happy', 'Happy', 'HotAnger', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']


trueDict["JG1"] = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']
predDict["JG1"] = ['Happy', 'Happy', 'Bored', 'Happy', 'Bored', 'Happy', 'Bored', 'Bored', 'Bored', 'Bored', 'Sad', 'Bored', 'Bored', 'Neutral', 'Bored', 'Happy', 'Happy', 'Neutral', 'Happy', 'Happy', 'Sad', 'Happy', 'Sad', 'Sad', 'Happy', 'Bored', 'Bored', 'Bored', 'Sad', 'Happy', 'Bored', 'Happy', 'Happy', 'Happy', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'Happy', 'Happy']



def main():
    print emotions

    cmTotal = np.zeros((len(emotions), len(emotions)))

    for speakerID in speakers:

        try:
            cm = confusion_matrix(trueDict[speakerID], predDict[speakerID], labels = emotions)
            cmTotal+=cm
            print speakerID
            print cm
            print (100.0*cm.T/cm.sum(1)).T.round(2)
        except KeyError:
            pass

    print emotions
    print cmTotal
    print (100.0*cmTotal.T/cmTotal.sum(1)).T.round(2)


if __name__ == "__main__":
    main()