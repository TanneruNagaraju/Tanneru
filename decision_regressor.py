import numpy as np
import  pandas as pd

df= pd.read_csv('petrol_consumption.csv')
"""
print(df)
print(df.shape)
print(df.size)
print(df.info())
print(df.describe())
print(df.columns)
print(df.head())
"""

X = df.drop('Petrol_Consumption',axis = 1)
Y = df['Petrol_Consumption']
print(X)
print(Y)



from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=55)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Xtrain:",X_train)
print("Xtest:",X_test)



from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

print(model.fit(X_train,Y_train))

ytrainpred = model.predict(X_train)
ytestpred = model.predict(X_test)

print("Ytrain:",ytrainpred)
print("YTest:",ytestpred)

compare = pd.DataFrame({'Actual':Y_test,'Predict':ytestpred})
print(compare)




from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("Mean_absolute error:",mean_absolute_error(Y_test,ytestpred))
print("Mean_squared error:",mean_squared_error(Y_test,ytestpred))
print("Root mean squared error:",np.sqrt(mean_squared_error(Y_test,ytestpred)))




