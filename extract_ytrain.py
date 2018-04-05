import numpy as np
import pandas as pd


print("Reading in data...")
N = 6325
data = pd.read_csv("train.csv.gz", header = None, usecols = [88200], nrows = N) 
print("Done!")


np.save('Data/y_train', data.iloc[:,0].values)


