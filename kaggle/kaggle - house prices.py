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
    house_price_df_data = house_price_df.iloc[:,1:-1]
    house_price_df_labels = (house_price_df.iloc[:,-1])
    house_price_dummies = pd.get_dummies(house_price_df_data,drop_first = False)
    nulls = (house_price_dummies.isna()*1).sum()
    house_dummies_filled = house_price_dummies.copy()
    print ((house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['LotFrontage'] = house_price_dummies['LotFrontage'].fillna(value = 0)
    nulls = (house_dummies_filled.isna()*1).sum()
    print ((house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['GarageYrBlt'] = house_dummies_filled['GarageYrBlt'].fillna(house_dummies_filled['YearBuilt']) 
    nulls = (house_dummies_filled.isna()*1).sum()
    print ((house_dummies_filled.isna()*1).sum().max())
    house_dummies_filled['MasVnrArea'] = house_price_dummies['MasVnrArea'].fillna(value = 0)
    print ((house_dummies_filled.isna()*1).sum().max())

x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(house_price_df_data, house_price_df_labels, test_size = 0.25, random_state=120)
x_train_fill, x_test_fill, y_train_fill, y_test_fill = train_test_split(house_dummies_filled, house_price_df_labels, test_size = 0.25, random_state=120)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score
corr = x_train_raw.corr()
numeric_labels = corr.columns
x_train_raw_numeric = x_train_raw[numeric_labels]
x_test_raw_numeric = x_test_raw[numeric_labels]
x_train_raw_numeric.fillna(value = 0,inplace = True)
x_test_raw_numeric.fillna(value = 0,inplace = True)
RFregr = RandomForestRegressor(n_estimators=1000)
RFregr.fit(x_train_raw_numeric, y_train_raw)
#y_pred = RFregr.predict(x_train_raw_numeric)



RFregr1 = RandomForestRegressor(n_estimators=1000)
RFregr1.fit(x_train_fill, y_train_fill)
y_pred = RFregr1.predict(x_test_fill)
RFtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)
print ('RF on raw data')
print (RFregr.score(x_test_raw_numeric, y_test_raw))
print ('RF on cleaned data')
print (RFregr1.score(x_test_fill, y_test_fill))
print ('total error is RF on cleaned data: ' , RFtotal_error)


GDregr= GradientBoostingRegressor(n_estimators=1000)
GDregr1 = GradientBoostingRegressor(n_estimators=1000)
GDregr.fit(x_train_raw_numeric,y_train_raw)
GDregr1.fit(x_train_fill,y_train_fill)
y_pred = GDregr1.predict(x_test_fill)
GDtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)
print ('GD on raw data')
print (GDregr.score(x_test_raw_numeric, y_test_raw))
print ('GD on cleaned data')
print (GDregr1.score(x_test_fill, y_test_fill))
print ('total error is GD on cleaned data: ' , GDtotal_error)
GDregr2 = GradientBoostingRegressor(n_estimators=3000)
GDregr2.fit(house_dummies_filled,house_price_df_labels)
y_pred = GDregr2.predict(x_test_fill)
GDtotal_error = np.sum((y_pred-y_test_fill)**2)/len(y_pred)
print (GDregr2.score(x_test_fill, y_test_fill))
print ('total error with GD 3000 on cleaned data: ' , GDtotal_error)

parameters = {'max_depth':[3],'n_estimators':[3000],'max_features':['sqrt'],'loss':['huber'],'min_samples_leaf':[15],'min_samples_split':[10],'random_state':[0]}
model=GridSearchCV(GradientBoostingRegressor(),parameters,scoring='neg_mean_squared_error',cv=KFold(n_splits=7))
model.fit(house_dummies_filled,house_price_df_labels)
Y_test1=model.predict(test_df)


plt.figure(figsize=(7,7))
plt.scatter(y_test_fill,y_pred,color = 'blue')
plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('pred X label')
plt.xlabel('label - True Y')
plt.ylabel('Prediction')
plt.show()
GDregrlog = GradientBoostingRegressor(n_estimators=1000)
GDregrlog.fit(x_train_fill,np.log(y_train_fill))
y_pred_log = np.exp(GDregrlog.predict(x_test_fill))
GDtotal_error_log = np.sum((y_pred_log-y_test_fill)**2)/len(y_pred_log)
#print ('GD on log cleaned data')
#print (GDregrlog.score(x_test_fill, np.exp(y_test_fill)))
print ('total error on log GD cleaned data: ' , GDtotal_error_log)
print ('log - reg = ' ,GDtotal_error_log- GDtotal_error)
plt.figure(figsize=(7,7))
plt.scatter(y_test_fill,y_pred_log,color = 'blue')
plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('pred X labe - LOG')
plt.xlabel('label - True Y')
plt.ylabel('Prediction')
plt.show()


plt.scatter(x_train_fill['OverallQual'],y_train_fill,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X overallqual - LOG')
plt.xlabel('OverallQual')
plt.ylabel('y')
plt.show()

plt.scatter(x_train_fill['GrLivArea'],y_train_fill,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X vGrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('price')
plt.show()

plt.scatter(np.log(x_train_fill['GrLivArea']),y_train_fill,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X vGrLivArea - LOG')
plt.xlabel('GrLivArea')
plt.ylabel('price')
plt.show()

x_train_fill_cleaned = x_train_fill.drop(x_train_fill[x_train_fill['GrLivArea']>4500].index)
y_train_fill_cleaned = y_train_fill.drop(x_train_fill[x_train_fill['GrLivArea']>4500].index)
plt.scatter(x_train_fill_cleaned['GrLivArea'],y_train_fill_cleaned,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X vGrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('price')
plt.show()



plt.scatter(x_train_fill_cleaned['LotArea'],y_train_fill_cleaned,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X LotArea')
plt.xlabel('LotArea')
plt.ylabel('price')
plt.show()

x_train_fill_cleaned = x_train_fill.drop(y_train_fill[y_train_fill>590000].index)
y_train_fill_cleaned = y_train_fill.drop(y_train_fill[y_train_fill>590000].index)

plt.scatter(x_train_fill_cleaned['LotArea'],y_train_fill_cleaned,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X LotArea')
plt.xlabel('LotArea')
plt.ylabel('price')
plt.show()

x_train_fill_cleaned = x_train_fill.drop(x_train_fill[x_train_fill['LotArea']>100000].index)
y_train_fill_cleaned = y_train_fill.drop(x_train_fill[x_train_fill['LotArea']>100000].index)

plt.scatter(x_train_fill_cleaned['LotArea'],y_train_fill_cleaned,color = 'blue')
#plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('price X LotArea')
plt.xlabel('LotArea')
plt.ylabel('price')
plt.show()

GDregr_clean = GradientBoostingRegressor(n_estimators=2000)
GDregr_clean.fit(x_train_fill_cleaned,y_train_fill_cleaned)
y_pred_clean = GDregr_clean.predict(x_test_fill)
GDtotal_error_clean = np.sum((y_pred_clean-y_test_fill)**2)/len(y_pred_clean)
#print ('GD on log cleaned data')
#print (GDregrlog.score(x_test_fill, np.exp(y_test_fill)))
print ('total error of cleaner data: ' , GDtotal_error_log)
plt.figure(figsize=(7,7))
plt.scatter(y_test_fill,y_pred_log,color = 'blue')
plt.plot(y_test_fill, y_test_fill, color = 'red')
plt.title ('pred X label - cleand dataset')
plt.xlabel('label - True Y')
plt.ylabel('Prediction')
plt.show()

#corr = x_train_raw.corr()
#numeric_labels = corr.columns
#x_train_raw_numeric = x_train_raw[numeric_labels]
#x_test_raw_numeric = x_test_raw[numeric_labels]
#x_train_raw_numeric.fillna(value = 0,inplace = True)
#x_test_raw_numeric.fillna(value = 0,inplace = True)
#RFregr = RandomForestRegressor()
#RFregr.fit(x_train_raw_numeric, y_train_raw)
#y_pred = RFregr.predict(x_train_raw_numeric)
#RFregr.score(x_test_raw_numeric, y_test_raw)

    

    
    