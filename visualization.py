import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#Dropping ID, Ticket and Name columns as they hold no information beneficial for the classification
titanic_train.drop(['Ticket', 'Name', 'PassengerId'], inplace=True, axis=1)

#Plotting the data
sns.pairplot(data=titanic_train, corner=True)
plt.show()

#Plotting survival rates
sns.countplot(data=titanic_train, x='Survived')
plt.title('Amount of Survivors and Not Survivors')
plt.show()

#Survivors distributed by gender
sns.countplot(data=titanic_train, x='Sex', hue='Survived')
plt.title('Survival rate vs Sex')
plt.show()

#Survival rate including passengers class
sns.barplot(data=titanic_train, x='Sex', y='Survived', hue='Pclass')
plt.title('Survival rate for passengers with different class')
plt.show()

#Survival rate per age group
sns.swarmplot(data=titanic_train, x='Survived', y='Fare', hue='Survived')
plt.title('Fare of People who did or did not survive')
plt.show()

#Survival rate per age
sns.swarmplot(data=titanic_train, x='Survived', y='Age', hue='Sruvived')
plt.title('Age distribution for survivors')
plt.show()

#Exploring the correlation
sns.heatmap(data=titanic_train.corr(numeric_only=True), cmap='coolwarm', annot=True)
plt.title('Correlation heatmap')
plt.show()

#After analyzing the plots and correlations the columns Age and SibSp will be dropped as their relation to survival of a person is minimal