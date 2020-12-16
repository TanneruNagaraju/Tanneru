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



from sklearn.model_selection import KFold
kf = KFold(n_splits=10,random_state=55)


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(base_estimator=DTC,n_estimators=600,random_state=60)


from sklearn.model_selection import  cross_val_score
scores = cross_val_score(bagging,X,Y,cv=kf)
print(scores)
print("Accuracy :",scores.mean())




