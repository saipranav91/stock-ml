import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
data=pd.read_csv("stock market.csv")
df=data.copy()
df.drop("Date",axis='columns',inplace=True)
Y=df[['Close','Adj Close']]
df.drop("Adj Close",axis='columns',inplace=True)
df.drop('Close',axis='columns',inplace=True)
x_train,x_test,y_train,y_test=train_test_split(df,Y,test_size=0.35,random_state=0,shuffle=True)
scalar=StandardScaler() 
X_train=scalar.fit_transform(x_train)
X_test=scalar.transform(x_test)

model=RandomForestRegressor()
model.fit(X_train,y_train)


new=scalar.transform([[22.857143,23.149643,22.814644,442761200]])
print(new)
x=model.predict(new)
print(x)

#with open("stock.pkl",'wb') as file:
    #pickle.dump(model,file)

with open("scalar.pkl",'wb')as file:
    pickle.dump(scalar,file)

#with open("stock.pkl",'rb') as file:
    #model=pickle.load(file)

#p=model.predict(new)
#print(p)
 


