print("Loading in libraries...")

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.io.wavfile import write

import six
import tensorflow as tf

from os import listdir
from os.path import join

from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_postprocess
from vggish import vggish_slim

pca_params = "vggish/vggish_pca_params.npz"
checkpoint = "vggish/vggish_model.ckpt"

print("Done!")


print("Reading in data...")
data = pd.read_csv("train_500.csv", header = None)
print("Done!")


print("Creating .wav files...")

N = 10 #Number of samples we are using: CHANGE THIS!!!
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


print("Processing .wav files...")

wav_file_direc = "WaveFiles/"
wav_files = listdir(wav_file_direc)

#Initialize array of batches and read each wav_file in wav_files array
batches = []

for wav_file in wav_files:
    if "wav" in wav_file:
        examples_batch = vggish_input.wavfile_to_examples(join(wav_file_direc,wav_file))
        batches.append(examples_batch)

print("Done!")


print("Computing Tensorflow Embeddings...")
# Prepare a postprocessor to munge the model embeddings.
pproc = vggish_postprocess.Postprocessor(pca_params)

output_sequences = []
with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    for batch in batches:
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)
        output_sequences.append(postprocessed_batch)

output_sequences = np.array(output_sequences)
print("Done!")

print("Processing and saving as a pickle file...")

#Sort the Output Sequences
order = []
for wavfile in wav_files:
	if "wav" in wavfile:
		order.append(int(wavfile[7:-4]))

output_sequences_sorted = []
for i in range(0, N):
    arg = order.index(i)
    output_sequences_sorted.append(output_sequences[arg])

output_sequences_sorted = np.array(output_sequences_sorted)
np.save('Data/xtrain_vggish', output_sequences_sorted)
print("Output Shape: ",output_sequences_sorted.shape)
print("Done!")
print("---------")
print("All finished, we cookin'")




