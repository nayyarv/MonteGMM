__author__ = 'Varun Nayyar'

import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.MFCCArrayGen import speakers

# emotions = origEmotions[1:] + [origEmotions[0]]
emotions = ["Bored", "Happy", "HotAnger", "Neutral", "Sad"]

# from fullDataResuMC import trueDict, predDict
from deciDataResMC import trueDict, predDict


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
    cmtotper =  (100.0*cmTotal.T/cmTotal.sum(1)).T.round(2)
    print cmtotper
    print np.diag(cmtotper).mean()




if __name__ == "__main__":
    main()