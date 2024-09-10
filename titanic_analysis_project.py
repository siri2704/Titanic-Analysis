# -*- coding: utf-8 -*-
"""titanic analysis project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tuRAbc-KDXPwb95BMoNGtvS6KaVy4k1c
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import math
titanic_data = pd.read_csv('titanic.csv')
titanic_data.head()

sns.countplot(x='Survived',hue = "Sex", data=titanic_data)



sns.countplot(x="Survived",hue="Pclass",data=titanic_data)

titanic_data["Age"].plot.hist()

titanic_data['Fare'].plot.hist(bins=20,figsize=(10,5))
titanic_data.info()

sns.countplot(x='SibSp',data=titanic_data)

sns.countplot(x='Parch',data=titanic_data)

sns.countplot(x='Ticket',data=titanic_data)

sns.countplot(x='Cabin',data=titanic_data)

sns.countplot(x='Embarked',data=titanic_data)

#data wrangaling
titanic_data.isnull()

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)

sns.boxplot(x="Pclass" , y="Age" , data=titanic_data)

titanic_data.isnull().sum()

sns.boxplot(x="Pclass" , y="Age" , data=titanic_data)

titanic_data.head(5)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)

# Check if 'Cabin' column exists before trying to drop it
if 'Cabin' in titanic_data.columns:
  titanic_data.drop('Cabin',axis=1,inplace=True)
  print("Column 'Cabin' has been dropped successfully.")
else:
  print("Column 'Cabin' does not exist in the DataFrame.")

titanic_data.head()

titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)



titanic_data.isnull().sum()

titanic_data.head(5)

pd.get_dummies(titanic_data['Sex'])

Sex = pd.get_dummies(titanic_data['Sex'],drop_first=True) # Assign the result of the function to the variable Sex and set drop_first to True instead of 'True'
Sex.head(5)

emabrk=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
emabrk.head(5)

pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
pcl.head(5)

titanic_data=pd.concat([titanic_data,Sex,emabrk,pcl],axis=1)

titanic_data.head(5)

titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic_data.head(5)

titanic_data.drop(['Pclass'],axis=1,inplace=True)
titanic_data.head(5)

#train data
x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression( max_iter=10000)

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report

classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)

