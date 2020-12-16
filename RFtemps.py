import  numpy as np
import  pandas as pd
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv("temps.csv")
df = df1[['year','month','day','week','temp_2','temp_1','average','actual','friend']]

"""
print(df)
print(df.shape)
print(df.size)
print(df.info())
print(df.describe())
print(df.columns)
print(df.head())
"""


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['week'] = le.fit_transform(df['week'])
#print(df['week'])
print(df.head(5))

y = df[['week']]
print(y)

df  = df.drop('week',axis = 1)
print(df)



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')
y  = encoder.fit_transform(y).toarray()
print(y.size)
print(y.shape)
print(y)
#print(y[0])
y = pd.DataFrame(y)



#names =['mon','tue','wed','thurs','fri','satur''sun']
cl = pd.DataFrame({'mon': y[0],'tue':y[1],'wed':y[2],'thurs':y[3],'fri':y[4],'satur':y[5],'sun':y[6]})
#cl = pd.DataFrame(y,columns = names)
print(cl)


z = [df,cl]
df = pd.concat(z,axis=1)
print(df)




X = df.drop('actual',axis=1)
Y = df['actual']
print(X)
print(Y)
"""
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)
"""



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000,random_state=42)
print(model.fit(X_train,Y_train))

ypred = model.predict(X_test)
print(ypred)

compare = pd.DataFrame({'Actual': Y_test,'Predicted': ypred})
print(compare)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print("Mean sqaured error",mean_squared_error(Y_test,ypred))
print("Mean absolute error",mean_absolute_error(Y_test,ypred))
print("Root mean sqaured error ",np.sqrt(mean_squared_error(Y_test,ypred)))




#errors
errors = abs(ypred-Y_test)
print(errors)

#Mean absolute error manually
print("Mean Absolute Error :",round(np.mean(errors),2))


#mean absolute percentage error
mape = 100*(errors/Y_test)
print(mape)

#Accuracy
Accuracy = 100 - np.mean(mape)
print("Accuracy:",Accuracy)

