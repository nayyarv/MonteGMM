__author__ = 'Varun Nayyar'

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from MFCCArrayGen import emotions


CC1_true = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'Happy', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']
CC1_pred = ['Bored', 'Sad', 'Bored', 'Bored', 'Bored', 'Neutral', 'Bored', 'Bored', 'Neutral', 'Sad', 'Bored', 'Sad', 'Neutral', 'Sad', 'Sad', 'Sad', 'Sad', 'Sad', 'Bored', 'Neutral', 'Neutral', 'Sad', 'Bored', 'Sad', 'Bored', 'Bored', 'Neutral', 'Bored', 'Sad', 'Neutral', 'Sad', 'Neutral', 'Bored', 'Happy', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'Bored', 'HotAnger', 'HotAnger', 'Sad', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger', 'HotAnger']


def main():
    print emotions
    cm = confusion_matrix(CC1_true, CC1_pred, labels = emotions)
    print cm


if __name__ == "__main__":
    main()