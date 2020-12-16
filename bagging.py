import  numpy as np
import  pandas as pd
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('pima-indians-diabetes.csv',names = columns)
print(df)
print(df.shape)
print(df.size)
print(df.columns)
print(df.describe())
print(df.info())

X = df.iloc[:,:8].values
Y = df.iloc[:,8].values
print(X)
print(Y)
print(X.shape)
print(Y.shape)



