import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
df = fetch_20newsgroups()
#print(df.target_names)
categories = df.target_names
print(categories)

train = fetch_20newsgroups(subset='train',categories=categories)
test =  fetch_20newsgroups(subset='test',categories=categories)
#print(train.data[5])
#print(test.data[5])
print(len(train.data))
print(len(test.data))