import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.io.wavfile import write

print("Reading in data...")
N = 5 #Number of samples we are using: CHANGE THIS!!!
data = pd.read_csv("train.csv.gz", header = None, nrows = N) #CHANGE THE FILE NAME!!!
print("Done!")


print("Creating .wav files...")
print("N:",N)
train = data.iloc[0:N,1:].values
print("Train shape",train.shape)
N_train = train.shape[0] #Same as N
NUM_SAMPLES = train.shape[1]-1 #Number of columns, sans the class label

X_train = train[:,:-1]
y_train = train[:,-1]
y_train = y_train.reshape(N_train,1)


SAMPLE_RATE = 22050

for i in range(0, X_train.shape[0]):
    write("WaveFiles/xtrain_"+str(i)+".wav", SAMPLE_RATE, (32768*X_train[i]).astype(np.int16))

print("Done!")

