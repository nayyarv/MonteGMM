# Wave  reader




def fun():
    import os
    import wave

    Boreddir = "/Users/varunnayyar/Documents/Speech Stuff/LDC Cleaned/Sad"

    for files in os.listdir(Boreddir):
        if not files.endswith(".wav"):
            continue
        # print os.path.join(Boreddir, files)
        f = wave.open(os.path.join(Boreddir, files))
        print f.getnchannels(), f.getframerate()

    # print "hi"


def main():
    import cPickle

    with open("../MFCCData/Bored_CL1", 'r') as f:
        boredCL = cPickle.load(f)

    nData = 0
    for utterances in boredCL:
        nData += utterances.shape[0]
    print nData


if __name__ == '__main__':
    main()