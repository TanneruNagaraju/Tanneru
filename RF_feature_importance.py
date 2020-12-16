import  numpy as np
import  pandas as pd

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

X = df.iloc[:, :4].values
Y = df.iloc[:, 4].values
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Yenc, test_size=0.25, random_state=45)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

#Coverting all training & testing values into range(-1 to 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,max_depth=3)

print(model.fit(X_train, Y_train))

ypred = model.predict(X_test)
print(ypred)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, ypred))

from sklearn.metrics import classification_report
print(classification_report(Y_test, ypred))

from sklearn.metrics import  accuracy_score
print("Before Accuracy score:",accuracy_score(Y_test, ypred))


print("--------------------------After selecting important features------------------------------- ")

#Selecting important features

feature_names = df.drop('species',axis=1).columns
print(feature_names)

feature_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(feature_imp)



"""
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
"""


X1 = df[['sepal_length', 'petal_length', 'petal_width']]
Y1 = df['species']
#print(X)
#print(Y)
print(np.unique(Y1))
#print(df['species'].value_counts)

#converting names into numerical
Yenc1 = encode.fit_transform(Y1)
print(Yenc1)
from collections import Counter
print(Counter(Yenc1))
print(Counter(Y1))
print(np.unique(Yenc1))



#Splitting data
from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Yenc1, test_size=0.25, random_state=45)
print(X_train1)
print(X_test1)
print(Y_train1)
print(Y_test1)

#Coverting all training & testing values into range(-1 to 1)
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)
print(X_train1)
print(X_test1)


from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100,max_depth=3)

print(model1.fit(X_train1, Y_train1))

ypred1 = model1.predict(X_test1)
print(ypred1)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test1, ypred1))

from sklearn.metrics import classification_report
print(classification_report(Y_test1, ypred1))

from sklearn.metrics import  accuracy_score
print("After Accuracy score:",accuracy_score(Y_test1, ypred1))






