#import
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Arrange data frame
df_train = pd.read_csv(r'C:\Users\royru\Desktop\primrose\github\kaggle\train.csv')
df_test = pd.read_csv(r'C:\Users\royru\Desktop\primrose\github\kaggle\test.csv')

# to category
df_train = pd.get_dummies(df_train)

df_train = df_train.dropna(thresh=df_train.shape[0], axis=1)
df_train.drop(["Id"], inplace=True, axis=1)

df_train = df_train[df_train.T[df_train.dtypes!=np.object].index]

x = df_train.copy()
x.drop("SalePrice", axis=1, inplace=True)
y = df_train["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=120)

regr = RandomForestRegressor()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
regr.score(x_test, y_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
regr.score(x_test, y_test)

clf = SVR(C=1000, gamma = 0.001, kernel='linear')
clf.fit(x_train, y_train)
clf.score(x_test,y_test)
svm_y_pred = clf.predict(x_test)
# svm regression

## Correlation
#saleprice correlation matrix
plt.figure()
k = 10 #number of variables for heatmap
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
plt.show()

#scatterplot
plt.figure()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

df_train.isnull().sum().max() #just checking that there's no missing data missing...


