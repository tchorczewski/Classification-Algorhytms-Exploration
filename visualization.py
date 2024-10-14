import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

cancer = pd.read_csv('data.csv')

print(cancer.info())
print(cancer.describe())
print(cancer.head())