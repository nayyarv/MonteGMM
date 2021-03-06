__author__ = 'Varun Nayyar'

from Utils.MFCCArrayGen import emotions, speakers, getCorpus
from MCMC import MCMCRun
from emailAlerter import alertMe


def main2(numRuns = 100000, numMixtures = 8, speakerIndex = 6):
    import time

    

    for emotion in emotions:
        start = time.ctime()

        Xpoints = getCorpus(emotion, speakers[speakerIndex])


        message = MCMCRun(Xpoints, emotion+"-"+speakers[speakerIndex], numRuns, numMixtures)

        message += "Start time: {}\nEnd Time: {}\n".format(start, time.ctime())

        message += "\nNumRuns: {}, numMixtures:{}\n ".format(numRuns, numMixtures)

        message += "\nEmotion: {}, speaker:{}\n".format(emotion, speakers[speakerIndex])

        alertMe(message)




if __name__ == "__main__":
    for i in xrange(len(speakers)):
        main2(numMixtures=8, speakerIndex=i)