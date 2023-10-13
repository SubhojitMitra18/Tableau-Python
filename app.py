import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from io import StringIO
import matplotlib.pyplot as plt


df=pd.read_csv("C:\\Users\\nagbi\\OneDrive\\Desktop\\mobile price.csv") # upload the dataset in google collab files section
df.head()
df.dtypes
df.isnull().sum()
Target_ratio = df['price_range'].value_counts()/len(df)
print(Target_ratio)
plt.figure(figsize = (6,6))
plt.bar(Target_ratio.index,Target_ratio)
plt.ylabel('Percentage')
plt.show()
x = df.drop("price_range",axis=1)
y = df.price_range
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.2)
gaussian_nb=GaussianNB()
gaussian_nb.fit(x_train,y_train)
bernoulli_nb=BernoulliNB()
bernoulli_nb.fit(x_train,y_train)
pred=gaussian_nb.predict(x_test)
acc_gnb=accuracy_score(y_test,pred)
print('Accuracy Score for gaussian: ',acc_gnb)
pred=bernoulli_nb.predict(x_test)
acc_bnb=accuracy_score(y_test,pred)
print('Accuracy Score for bernoulli: ',acc_bnb)
plt.barh(['Gaussian','Bernoulli'],[acc_gnb,acc_bnb])
