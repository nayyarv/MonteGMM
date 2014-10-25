__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt
import os
import cPickle



ExtractedMFCCs = "../MFCCData"


def main():
    MFCCFiles = os.listdir(ExtractedMFCCs)
    try:
        MFCCFiles.remove('.DS_Store')
    except ValueError:
        pass
        # It didn't need to be removed
    MFCCVals = []
    for file in MFCCFiles:
        if "Sad" in file:
            print file
            with open(os.path.join(ExtractedMFCCs, file)) as f:
                speakerEmotion = cPickle.load(f)
                speakerEmotion = np.vstack(speakerEmotion)
                MFCCVals.append(speakerEmotion)

    SadCorpus = np.vstack(MFCCVals)

    print SadCorpus.shape


def SadCorpus():
    MFCCFiles = os.listdir(ExtractedMFCCs)
    try:
        MFCCFiles.remove('.DS_Store')
    except ValueError:
        pass
        # It didn't need to be removed
    MFCCVals = []
    for file in MFCCFiles:
        if "Sad" in file:
            print file
            with open(os.path.join(ExtractedMFCCs, file)) as f:
                speakerEmotion = cPickle.load(f)
                speakerEmotion = np.vstack(speakerEmotion)
                MFCCVals.append(speakerEmotion)

    return np.vstack(MFCCVals)



if __name__ == "__main__":
    main()