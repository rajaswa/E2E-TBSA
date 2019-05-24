#Reading the txt file

import pandas as pd
df = pd.read_csv('glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove = {key: val.values for key, val in df.T.items()}

#converting to pickle file
import pickle
with open('glove.840B.300d.pkl', 'wb') as fp:
    pickle.dump(glove, fp)