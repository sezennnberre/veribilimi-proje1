# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:00:06 2023

@author: HP
"""

import numpy as np
import pandas as pd
import math
import re
from sklearn.ensemble import RandomForestRegressor


traffic05_07 = pd.read_csv("accidents_2005_to_2007.csv")
traffic09_11 = pd.read_csv("accidents_2009_to_2011.csv")
traffic12_14 = pd.read_csv("accidents_2012_to_2014.csv")

all_traffic = pd.concat([traffic05_07,traffic09_11,traffic12_14])

all_traffic.drop(["Accident_Index","Location_Easting_OSGR","Location_Northing_OSGR","Longitude","Latitude","Police_Force",
                 "Date","Day_of_Week","Time","Local_Authority_(Highway)","1st_Road_Class","1st_Road_Number","Road_Type",
                 "Junction_Detail","Junction_Control","2nd_Road_Class","2nd_Road_Number","Did_Police_Officer_Attend_Scene_of_Accident",
                 "LSOA_of_Accident_Location","Local_Authority_(District)","Number_of_Casualties"],axis=1,inplace=True)

all_traffic.dropna(inplace=True)

#-------------------------trafik akışının değişmesi kazaları nasıl etkiler?--------------------------------

siddet = all_traffic.iloc[:,0:1].values


veriseti = all_traffic.iloc[:,1:3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(veriseti,siddet,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestRegressor
r = RandomForestRegressor()
r.fit(x_train,y_train)
y_pred_r = r.predict(x_test)

# r^2 ile hata hesabı
from sklearn.metrics import r2_score
print("Random Forest R2 degeri")
print(r2_score(y_train,r.predict(x_train)))
print(r2_score(y_test, r.predict(x_test)))
#değerler her çalıştırmada farklılık gösterebilir neden bilmiyorum.

# RMSLE ile hata hesabı       squared false yapınca root oluyor.
from sklearn.metrics import mean_squared_log_error
print("Mean Squared Log Error degeri")
print(mean_squared_log_error(y_train,r.predict(x_train),squared=False))
print(mean_squared_log_error(y_test,r.predict(x_test),squared=False))

#RMSE ile hata hesabı
from sklearn.metrics import mean_squared_error
print("Mean Squared Error degeri")
print(mean_squared_error(y_train, r.predict(x_train),squared=False))
print(mean_squared_error(y_test, r.predict(x_test),squared=False))

#-----------------------------------kaza oranlarını ne arttırır?--------------------------------------

#---------------------------------zaman içinde kaza oranlarını tahmin edebilir miyiz?------------------

#-------------------------------kırsal ve kentsel alanlar nasıl farklılaşır?--------------------------


#açıkçası bu sorulara nasıl cevap vereceğimi anlamadım. prediction yapmak için ne kullanıcam ya da kullanacak mıyım
#anlamadım. bundan dolayı burada bırakıyorum. 








