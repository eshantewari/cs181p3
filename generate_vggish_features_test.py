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
N = 1000

'''
print("Reading in data...")
data = pd.read_csv("test.csv.gz", header = None)
print("Done!")


print("Creating .wav files...")

N = 1000 #Number of samples we are using
print("N:",N)
X_test = data.iloc[:,2:].values #For consistency with the one-off error in the training
indicies = data.iloc[:,0].values
print("Test shape",X_test.shape)
N_test = X_test.shape[0] #Same as N
NUM_SAMPLES = X_test.shape[1] #Number of columns

SAMPLE_RATE = 22050

for i in range(0, X_test.shape[0]):
    write("WaveFilesTest/xtest_"+str(i)+".wav", SAMPLE_RATE, (32768*X_test[i]).astype(np.int16))

print("Done!")
'''

print("Reading in data...")
N = 1000
data = pd.read_csv("test.csv.gz", header = None, usecols = [0], nrows = N) 
print("Done!")
indicies = data.iloc[:,0].values

np.save('Data/test_indicies', indicies)

print("Processing .wav files...")

wav_file_direc = "WaveFilesTest/"
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

    count = 0
    for batch in batches:
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)
        output_sequences.append(postprocessed_batch)
        count += 1
        if count % 100 == 0:
            print("At Embedding ",count,"/",N)

output_sequences = np.array(output_sequences)
print("Done!")

print("Processing and saving as a pickle file...")

#Sort the Output Sequences
order = []
for wavfile in wav_files:
	if "wav" in wavfile:
		order.append(int(wavfile[6:-4]))

output_sequences_sorted = []
for i in indicies.tolist():
    arg = order.index(i)
    output_sequences_sorted.append(output_sequences[arg])

output_sequences_sorted = np.array(output_sequences_sorted)
np.save('Data/xtest_vggish', output_sequences_sorted)

print("Output Shape: ",output_sequences_sorted.shape)
print("Done!")
print("---------")
print("All finished, we cookin'")

