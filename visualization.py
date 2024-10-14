import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.feature_selection import f_regression

sns.set_theme()

titanic_train = pd.read_csv('train.csv')

#Exploration of train data
print(titanic_train.info())
print(titanic_train.describe())
print(titanic_train.head())

#Checking for missing data
print(titanic_train.isna().sum())

#Visualization of missing data
sns.heatmap(titanic_train.isna(), cmap='viridis')
plt.show()

#Dropping the Cabin column as there is a lot of data missing
titanic_train.drop('Cabin', axis=1, inplace=True)

#Filling missing age data using mean
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train.groupby('Sex')['Age'].transform('mean'))
print(titanic_train.isna().sum())

#Checking 2 missing entries in Embarked column
print(titanic_train.loc[titanic_train['Embarked'].isna()])

#Removal of rows with missing entry in Embarked column
titanic_train.dropna(subset=['Embarked'], axis=0, inplace=True)
print(titanic_train.isna().sum())

#Plotting the data
sns.pairplot(data=titanic_train)
plt.show()

#Plotting survival rates
sns.displot(data=titanic_train, x='Sex', y='Pclass', hue='Survived')
plt.title('')

#Plotting Fare distribution
sns.displot(data=titanic_train, x='Fare')
plt.title('Fare distribution')
plt.show()

