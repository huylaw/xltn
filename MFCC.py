import numpy as np
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt

file = open('mfcc.txt', 'w')

for i in range(1, 101):
    audio_path = 'D:/19021305_LeBaGiaHuy/c%d.wav' % (i)
    com_path = 'D:/19021305_LeBaGiaHuy/c%d.txt' % (i)
    com_array = np.genfromtxt(com_path, delimiter='\t', dtype='unicode')
    for c in com_array:
        ipd.Audio(audio_path)
        signal, sr = librosa.load(path=audio_path, offset=float(c[0]), duration=float(c[1]) - float(c[0]))

        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        features = np.concatenate((mfccs, delta, delta2))

        file.writelines('%s %d %d\n' % (c[2], features.shape[0], features.shape[1]))
file.close()