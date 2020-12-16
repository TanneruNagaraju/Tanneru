import numpy as np
import pandas as pd


df = pd.read_csv("bill_authentication.csv")
"""
print(df)
print(df.shape)
print(df.size)
print(df.columns)
print(df.info())
print(df.head())
"""


X = df.iloc[:,:4].values
Y = df.iloc[:,4].values

"""
print(X)
print(Y)
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)
"""
from collections import  Counter
print(Counter(Y))
print(np.unique(Y))
print(df['Class'].value_counts())


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=65)

"""
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
print(X_train.size)
print(X_train.shape)
print(Y_train.size)
print(Y_train.shape)
print(X_test.size)
print(X_test.shape)
print(Y_test.size)
print(Y_test.shape)
"""

from sklearn.preprocessing  import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
print(model.fit(X_train,Y_train))

#Training predictions
ytrainpred = model.predict(X_train)
print("Training predictions",ytrainpred)


#Testing predictions
ypred = model.predict(X_test)
print("Ytesting predictions",ypred)

compare = pd.DataFrame({'Actual':Y_test,'Predict':ypred})
print(compare)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,ypred))


from sklearn.metrics import classification_report
print(classification_report(Y_test,ypred))


from sklearn.metrics import f1_score
print("Trianing F1 score : " ,f1_score(ytrainpred,Y_train))
print("Testing F1 score :",f1_score(ypred,Y_test))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_train,Y_train,cv=5)
print("cross validation:",scores)
print("Average for scores:" ,scores.mean())

from sklearn.metrics import accuracy_score
print("Accuracy Score : ",accuracy_score(Y_test,ypred))

#tuning parameters
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth' : [1,2,3,4,5],
              'min_samples_leaf': [1,2,3,4,5],
              'min_samples_split':[2,3,4,5],
              'criterion':['entropy','gini']}

grid = GridSearchCV(model,parameters,cv=5)
print(grid.fit(X_train,Y_train))

gspred = grid.predict(X_test)
print(gspred)


print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

print(accuracy_score(Y_test,gspred))




