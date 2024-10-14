import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

spam = pd.read_csv('spam.csv')

print(spam.info())
print(spam.describe())
print(spam.head())

#Checking for missing data
print(spam.isna().sum())

