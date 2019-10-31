# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy as sp
import statistics
from sklearn import impute 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#########################################################################################
if __name__ == "__main__":
    house_price_df = pd.read_csv(r'C:\Users\royru\Desktop\primrose\github\kaggle\train.csv')
    test_df = pd.read_csv(r'C:\Users\royru\Desktop\primrose\github\kaggle\test.csv')
    nulls1 = house_price_df.isna().sum()
    nulls_test = test_df.isna().sum()
    house_price_df_data = house_price_df.iloc[:,1:-1]
    test_house_price_df_data = test_df.iloc[:,1:-1]
    house_price_df_labels = (house_price_df.iloc[:,-1])
    house_price_dummies = pd.get_dummies(house_price_df_data,drop_first = False)
    test_house_price_dummies = pd.get_dummies(test_house_price_df_data,drop_first = False)
    nulls = (house_price_dummies.isna()*1).sum()
    house_dummies_filled = house_price_dummies.copy()
    test_house_dummies_filled = test_house_price_dummies.copy()
    print ((house_dummies_filled.isna()*1).sum().max())
    print ((test_house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['LotFrontage'] = house_price_dummies['LotFrontage'].fillna(value = 0)
    test_house_dummies_filled['LotFrontage'] = test_house_price_dummies['LotFrontage'].fillna(value = 0)
    nulls = (house_dummies_filled.isna()*1).sum()
    print ((house_dummies_filled.isna()*1).sum().max())
    print ((test_house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['GarageYrBlt'] = house_dummies_filled['GarageYrBlt'].fillna(house_dummies_filled['YearBuilt']) 
    test_house_dummies_filled['GarageYrBlt'] = test_house_dummies_filled['GarageYrBlt'].fillna(house_dummies_filled['YearBuilt']) 
    nulls = (house_dummies_filled.isna()*1).sum()
    print ((house_dummies_filled.isna()*1).sum().max())
    print ((test_house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['MasVnrArea'] = house_price_dummies['MasVnrArea'].fillna(value = 0)
    test_house_dummies_filled['MasVnrArea'] = test_house_price_dummies['MasVnrArea'].fillna(value = 0)
    print ((house_dummies_filled.isna()*1).sum().max())
    print ((test_house_dummies_filled.isna()*1).sum().max())
    nulls = (test_house_dummies_filled.isna()*1).sum()
    test_house_dummies_filled['GarageArea'] = test_house_price_dummies['GarageArea'].fillna(value = 0)
    test_house_dummies_filled['GarageCars'] = test_house_price_dummies['GarageCars'].fillna(value = 0)
    test_house_dummies_filled['TotalBsmtSF'] = test_house_price_dummies['TotalBsmtSF'].fillna(value = 0)
    test_house_dummies_filled['GarageCars'] = test_house_price_dummies['GarageCars'].fillna(value = 0)
    nulls2 = (test_house_dummies_filled.isna()*1).sum()
    test_house_dummies_filled = test_house_price_dummies.fillna(value = 0)
    nulls2 = (test_house_dummies_filled.isna()*1).sum()
    test_house_dummies_filled = test_house_dummies_filled.dropna(axis=0, how='any')
    print ((test_house_dummies_filled.isna()*1).sum().max())
    
    trainDF, testDF = house_dummies_filled.align(test_house_dummies_filled, join='inner', axis=1)


x_train_fill, x_test_fill, y_train_fill, y_test_fill = train_test_split(trainDF, house_price_df_labels, test_size = 0.25, random_state=120)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

RFregr1 = RandomForestRegressor(n_estimators=1000)
RFregr1.fit(x_train_fill, y_train_fill)
y_pred = RFregr1.predict(x_test_fill)
RFtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)

print ('RF on cleaned data')
print (RFregr1.score(x_test_fill, y_test_fill))
print ('total error is RF on cleaned data: ' , RFtotal_error)


GBregr1 = GradientBoostingRegressor(n_estimators=1000)
GBregr1.fit(x_train_fill,y_train_fill)
y_pred = GBregr1.predict(x_test_fill)
GBtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)
print ('GB on cleaned data')
print (GBregr1.score(x_test_fill, y_test_fill))
print ('total error for GB on cleaned data: ' , GBtotal_error)
GBregr2 = GradientBoostingRegressor(n_estimators=3000)
GBregr2.fit(x_train_fill,y_train_fill)
y_pred = GBregr2.predict(x_test_fill)
GBtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)
print (GBregr2.score(x_test_fill, y_test_fill))
print ('total error with GD 3000 on cleaned data: ' , GBtotal_error)

submission_reg = GradientBoostingRegressor(n_estimators=2500)
submission_reg.fit(trainDF,house_price_df_labels)
submission = submission_reg.predict(testDF.to_numpy())

submission_df=pd.DataFrame({'Id':range(1461,2920),'SalePrice':submission})
submission_df.to_csv('submit.csv',index=False)




    
    