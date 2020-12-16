import numpy as np
import pandas as pd


df = pd.read_csv("mushroom_csv.csv")
#print(df)
print(df.shape)
print(df.size)
print(df.columns)
print(df.info())
print(df.describe())
print(df.head())

from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

#Converting all columns string values into digits
for i in df.columns:
    df[i] = encode.fit(df[i]).transform(df[i])


print(df)


X = df.iloc[:,:22].values
Y = df.iloc[:,22].values
#print(X)
#print(Y)
print(X.shape)
print(Y.shape)




print(np.unique(Y))

from collections import Counter
print(Counter(Y))

print(df['class'].value_counts())



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.25,random_state=56)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=1)

from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(base_estimator=model,n_estimators=100,algorithm='SAMME')

print(boost.fit(X_train,Y_train))


ytrainpred = boost.predict(X_train)
ytestpred = boost.predict(X_test)

print("Ytrainpred",ytrainpred)
print("Ytestpred ",ytestpred)


compare = pd.DataFrame({'Actual':Y_test,'Predict':ytestpred})
print(compare)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,ytestpred))

from sklearn.metrics import classification_report
print(classification_report(Y_test,ytestpred))


from sklearn.metrics import accuracy_score
print("Accuracy score : " ,accuracy_score(Y_test,ytestpred))
