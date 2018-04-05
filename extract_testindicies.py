import numpy as np
import pandas as pd


print("Reading in data...")
N = 1000
data = pd.read_csv("test.csv.gz", header = None, usecols = [0], nrows = N) 
print("Done!")


np.save('Data/test_indicies', data.iloc[:,0].values)


