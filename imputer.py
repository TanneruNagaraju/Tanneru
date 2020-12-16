import numpy as np
import pandas as pd
"""

df = pd.read_csv("irisdataset.csv")
print(df)
print(df.columns)
print(df.species)

from sklearn.preprocessing import  LabelEncoder
encode = LabelEncoder()
print(encode.fit(df.species))
Y = encode.transform(df.species)
print(Y)
print(encode.fit_transform(df.species))
"""


data1 = pd.read_csv("Book1.csv")
print(data1)
print(data1.columns)
Y = data1[['marks']]
print(Y)
#Y = np.reshape(-1,1)
#Print(Y)

print(data1.isnull().sum())

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0,)
print(imp.fit(Y))
print(imp.transform(Y))
y1 = imp.fit_transform(Y)
print(y1)