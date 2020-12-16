import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("titanic_train.csv")

"""
print(df)
print(df.shape)
print(df.size)
print(df.info())
print(df.columns)
print(df.describe())
print(df.head(10))
print(df.isnull().sum())
"""

#df1 = df.drop('Survived',axis=1)
names = ['Name','Age','Cabin','Embarked','SibSp','Parch','Ticket','Survived']
#X = df.drop(labels=names,axis = 1)
X = df[['PassengerId','Pclass','Sex','Fare']]
Y = df['Survived']
print(X.shape)
#print(X.columns)
print(Y.shape)

#ocnverting Sex columns categorical into numerical & creating two separate two columns like onehot enocder
X = pd.get_dummies(X,columns=['Sex'])
print(X)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(Y_train.shape)
#print(X_test)


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=20,learning_rate=0.5,max_features=2,max_depth=2,random_state=0,)
print(model.fit(X_train,Y_train))

ypred = model.predict(X_test)
print(ypred)

compare = pd.DataFrame({'Actual' :Y_test,'Predicted':ypred})
print(compare)

print("Training score : ",model.score(X_train,Y_train))
print("Testing score :",model.score(X_test,Y_test))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,ypred))

from sklearn.metrics import classification_report
print(classification_report(Y_test,ypred))

from sklearn.metrics import accuracy_score
print("Accuracy score :",accuracy_score(Y_test,ypred))