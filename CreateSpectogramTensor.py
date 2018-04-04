import pandas as pd
%matplotlib inline
import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import numpy
import torch

#Change the Filename
data = pd.read_csv("train_500.csv")

N = len(data) #Number of samples we are using
print("N:",N)
train = data.iloc[0:N][:].values
print("Train shape",train.shape)
N_train = train.shape[0] #Same as N
NUM_SAMPLES = train.shape[1]-1 #Number of columns, sans the class label

X_train = train[:,:-1]
y_train = train[:,-1]
y_train = y_train.reshape(N_train,1)

# JUST SOME FOURIER TRANSFORM PARAMETERS
SAMPLE_RATE = 22050
BINS_OCTAVE = 12*2
N_OCTAVES = 7
NUM_BINS = BINS_OCTAVE * N_OCTAVES

# Given a wav time series, makes a mel spectrogram
# which is a short-time fourier transform with
# frequencies on the mel (log) scale.
def mel_spec(y):
    Q = librosa.cqt(y=y, sr=SAMPLE_RATE, bins_per_octave=BINS_OCTAVE,n_bins=NUM_BINS)
    Q_db = librosa.amplitude_to_db(Q,ref=np.max)
    return Q_db

# This means that the spectrograms are 168 rows (frequencies)
# By 173 columns (time frames)
song = X_train[0]
test_spec = mel_spec(song)
#print(test_spec.shape)
FEATS = test_spec.shape[0]
FRAMES = test_spec.shape[1]


#This creates a tensor, in which we have N_train examples, each of which is a mel-frequency "image"

tmp_train = np.zeros((N_train,FEATS,FRAMES))
for i in range(N_train):
    tmp_train[i,:,:] = mel_spec(X_train[i])

#Save the tensor, so that we can load it into other models
np.save('Data/xtrain_spec', tmp_train)
np.save('Data/ytrain_spec', y_train)

