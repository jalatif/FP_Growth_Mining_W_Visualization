__author__ = 'manshu'

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import csv

file_name = "/home/manshu/Templates/EXEs/CS412_Project/training.csv"

num_cols = [0, 1, 4, 5, 12, 14, 18, 19, 20, 21,22, 23, 24, 25, 26, 29, 30, 32, 33, 34]

enc = preprocessing.OneHotEncoder()
imp = Imputer(missing_values='NULL', strategy='most_frequent')

wheel_type = []

with open(file_name, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line_num = 0
    for row in spamreader:
        line_num += 1
        if line_num == 1:
            continue
        wheel_type.append(row[13])

print wheel_type
X = np.array(wheel_type)

print X.shape

uq_keys = np.unique(X)
bins = np.bincount(uq_keys.searchsorted(X))
binDict = {bins[i]:uq_keys[i] for i in range(0, len(uq_keys))}
most_freq = binDict[bins[0]]
print most_freq
new_X = [most_freq if val == "NULL" else val for val in X]
print(new_X)