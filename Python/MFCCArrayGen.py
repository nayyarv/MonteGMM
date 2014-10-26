__author__ = 'Varun Nayyar'

import numpy as np
import os
import cPickle


emotions = ["Sad", "Bored", "Happy", "HotAnger", "Neutral"]
speakers = ["CC1", "CL1", "GG1", "JG1", "MF1", "MK1", "MM1"]


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
    """
    A list of emotions
    :return: list of len 7. speakers voice for the sad emotion
    """
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

def getIndiviudalData(emotion, speakerID):
    """
    Mostly for testing, so we will maintain lists
    This will be a list of that speakers utterances of that word
    :param emotion:
    :param speakerID:
    :return:
    """
    if emotion not in emotions or speakerID not in speakers:
        raise Exception("No Such speaker: {} or emotion: {}".format(speakerID, emotion))
    #errorCheck

    filename = emotion+"_"+speakerID
    with open(os.path.join(ExtractedMFCCs, filename)) as f:
        speakerEmotion = cPickle.load(f)

    #No Vstacking, getting the list of each utterance
    return speakerEmotion


def getCorpus(emotion, speakerID = None):
    """
    Return the 6 speakers files in a massive vstack
    :param emotion:
    :param speakerID:
    :return:
    """

    if emotion not in emotions or (speakerID is not None and speakerID not in speakers):
        raise Exception("No Such speaker: {} or emotion: {}".format(speakerID, emotion))
    #error check

    if speakerID is None:
        #return whole corpus
        speakerID = "Derpington"
        # should not be in file

    MFCCFiles = os.listdir(ExtractedMFCCs)

    try:
        MFCCFiles.remove('.DS_Store')
    except ValueError:
        pass
        # It didn't need to be removed
    MFCCVals = []
    for file in MFCCFiles:
        if emotion in file and speakerID not in file:
            print "Currently reading", file
            with open(os.path.join(ExtractedMFCCs, file)) as f:
                speakerEmotion = cPickle.load(f)
                speakerEmotion = np.vstack(speakerEmotion)
                MFCCVals.append(speakerEmotion)

    return np.vstack(MFCCVals)



if __name__ == "__main__":
    main()