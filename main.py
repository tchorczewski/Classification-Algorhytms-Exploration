import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

#Dropping columns that were deemed unimportant during visualization and data exploration
titanic_train.drop(['Age','PassengerId', 'Ticket', 'Name','SibSp','Cabin'], axis=1, inplace=True)
titanic_test.drop(['Age','PassengerId', 'Ticket', 'Name','SibSp','Cabin'], axis=1, inplace=True)
#Dealing with null values
titanic_train.dropna(subset=['Embarked'], axis=0, inplace=True)

#Encoding categorical data
label_encoder = LabelEncoder()
titanic_train['Sex'] = label_encoder.fit_transform(titanic_train['Sex'])
titanic_test['Sex'] = label_encoder.fit_transform(titanic_test['Sex'])

titanic_train = pd.get_dummies(titanic_train,columns=['Embarked'], drop_first=True)
titanic_test = pd.get_dummies(titanic_test,columns=['Embarked'], drop_first=True)
print(titanic_train.info())
#Preparing data for predictions
x = titanic_train.drop('Survived', axis=1)
y = titanic_train['Survived']

x_train, x_validate,y_train,y_validate = train_test_split(x,y, random_state=42, test_size=0.25)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)
logistic_regression_results = logistic_regression.predict(x_validate)
print('Logistic Regression')
print(confusion_matrix(y_validate,logistic_regression_results))
print(classification_report(y_validate,logistic_regression_results))

random_forest = RandomForestClassifier()
random_forest.fit(x_train,y_train)
random_forest_results = random_forest.predict(x_validate)
print('Random Forest')
print(confusion_matrix(y_validate,random_forest_results))
print(classification_report(y_validate,random_forest_results))

svc = SVC()
svc.fit(x_train,y_train)
svc_results = svc.predict(x_validate)
print('SVC')
print(confusion_matrix(y_validate,svc_results))
print(classification_report(y_validate,svc_results))

