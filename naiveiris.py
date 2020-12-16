import  numpy as np
import pandas as pd
df = pd.read_csv("irisdataset.csv")

"""
print(df)
print(df.info())
print(df.corr())
print(df.size)
print(df.shape)
print(df.describe())
print(df.columns)
"""

X = df.iloc[:,:4].values
Y = df.iloc[:,4].values
#print(X)
#print(Y)
print(np.unique(Y))
#print(df['species'].value_counts)

#converting names into numerical
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
Yenc = encode.fit_transform(Y)
print(Yenc)
from collections import Counter
print(Counter(Yenc))
print(Counter(Y))
print(np.unique(Yenc))



#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Yenc,test_size=0.25,random_state=45)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

#Coverting all training & testing values into range(-1 to 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_trainscale = scaler.fit_transform(X_train)
X_testscale = scaler.transform(X_test)
print(X_trainscale)
print(X_testscale)



#Applying naive bayes to data
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
print(model.fit(X_trainscale,Y_train))

#Prediction for  test data
ypred = model.predict(X_testscale)
print(ypred)

#confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,ypred)
print(cm)

#Classification REport
from sklearn.metrics import classification_report
cr = classification_report(Y_test,ypred)
print(cr)

#Accuracy score
from sklearn.metrics import accuracy_score
print("Accuracy score",accuracy_score(Y_test,ypred)*100)

#Dynamically Testing
n1 = float(input("Enter a sepal_length value : "))
n2 = float(input("Enter a sepal_width value : "))
n3 = float(input("Enter a petal_length value : "))
n4 = float(input("Enter a petal_width value : "))

test = scaler.transform([[n1,n2,n3,n4]])
youtput = model.predict(test)
print("Output",youtput)
