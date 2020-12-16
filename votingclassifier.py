import  numpy as np
import  pandas as pd
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('pima-indians-diabetes.csv',names = columns )
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



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=45)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
from sklearn.tree import DecisionTreeClassifier

estimator = []
estimator.append(('LR',LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=200)))
estimator.append(('SVC',SVC(gamma='auto',probability=True)))
estimator.append(('DTC',DecisionTreeClassifier()))



from sklearn.ensemble import VotingClassifier
hardvoting =  VotingClassifier(estimators=estimator,voting = 'hard')
print(hardvoting.fit(X_train,Y_train))

hardypred = hardvoting.predict(X_test)
print("Hard y predict",hardypred)


from sklearn.metrics import accuracy_score
print("Hard voting Accuracy :",accuracy_score(Y_test,hardypred))



#soft voting
softvoting = VotingClassifier(estimators=estimator,voting='soft')
print(softvoting.fit(X_train,Y_train))

softypred = softvoting.predict(X_test)
print("Soft prediction",softypred)

print("Soft voting accuracy",accuracy_score(Y_test,softypred))

