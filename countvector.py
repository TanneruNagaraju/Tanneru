import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#converting raw data into dataframe format by using sep & adding columns name to dataset
df=pd.read_csv('SMSSpamCollection',sep='\t',names=['status','message'])

"""
print(df)
print(df.columns)
print(df.shape)
print(df.size)
print(len(df))
print(df.head())
#print(df.duplicated().sum())
print(df.info())
"""


#individual value count for column status
from collections import Counter
print(Counter(df.status))

#Same like counter function
print(df.status.value_counts())

print("Spam",len(df[df.status == 'spam']))
print("Ham",len(df[df.status == 'ham']))


"""
df.loc[df['status'] == 'ham','status',] == 0
df.loc[df['status'] == 'spam','status',] == 1
print(df)
"""


X = df.message
Y = df.status
print(X)
print(Y)



#converting spam & ham into 0 & 1
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
Y = encode.fit_transform(Y)
print(Y)


#Splitting Data into Testing & Training
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=4)

"""
print("Xtrain ",X_train)
print("xtest ",X_test)
print("Ytrain ",Y_train)
print("Ytest ",Y_test)
print(X_train.size)
print(X_test.size)
print(Y_train.size)
print(Y_test.size)
"""

cv = CountVectorizer() #Countvector
"""
#sample example for countvector 
xy = cv.fit_transform(["Hi Bro How are you", "Hey Whats up bro", "come lets go" ])
print(xy)
print(xy.toarray())
print(cv.get_feature_names())
z = cv.fit_transform(["hi hi hi hi hi","what what","are you"])
print(z)
print(z.toarray())
print(cv.get_feature_names())
"""


#Training part
xtraincv = cv.fit_transform(X_train)   # converting Xtrain value word into numbers
#print(xtraincv)
#print("Feature names",cv.get_feature_names()) # Feature names of columns X[message]
a = xtraincv.toarray() # converting into array
print(a)
print(a[0]) #first index of a[xtraincv ] after converting word  into numbers
print(cv.inverse_transform(a[0]))   # reconverting number to words by using inverse transform
print(a.size)
print(type(a))
print(a.shape)
print(len(a))
print(a[10])
print(cv.inverse_transform(a[10]))



#Testing part

xtestcv = cv.transform(X_test)
#print(xtestcv)
#print(cv.get_feature_names())
b = xtestcv.toarray()
print(b)
print(b.size)
print(b.shape)
print(type(b))
print(len(b))

print(b[9])


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

"""
print(X_train.head())

print(Y_train)
print(type(X_train))
print(type(Y_train))
Y_train = pd.Series(Y_train)
print(type(Y_train))
print(Y_train)
print(Y_train.head())


"""
#Y_train = Y_train.astype('int')
#print(Y_train)
#print(Y_train)

xtraincv = xtraincv.toarray() #got an error so  converting to array
print(model.fit(xtraincv,Y_train))

xtestcv = xtestcv.toarray()  # got an error converting to array
ypred = model.predict(xtestcv)
print("ypredict",ypred)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,ypred))



#Classification report

from sklearn.metrics import classification_report
print(classification_report(Y_test,ypred))


#Accuracy Score
from sklearn.metrics import accuracy_score
print("Accuracy Score",accuracy_score(Y_test,ypred))
