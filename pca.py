import  numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
"""
print(dataset.keys())
print(dataset.DESCR)
print(dataset['target'])
print(dataset['target_names'])
print(dataset.feature_names)
print(dataset.data)
print(dataset.filename)
"""

df = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
print(df)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(sc.fit(df))

scdata = sc.transform(df)
print(scdata)

#Reducing 30d into 2d [30 columns to 2 columns] using PCA
from sklearn.decomposition import PCA
pc = PCA(n_components=2)
print(pc.fit(scdata))
pcdata = pc.transform(scdata)
print(pcdata)
print(df.shape)
print(pcdata.shape)

import matplotlib.pyplot as plt
plt.scatter(pcdata[:,0],pcdata[:,1])
plt.xlabel("first component")
plt.ylabel("Second component")

