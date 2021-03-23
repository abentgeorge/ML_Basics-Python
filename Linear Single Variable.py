# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:43:55 2021

@author: Aben George

# Credits - CodeBasics
"""


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# SK LEARN = SICKIT LEARN library used for machine learning 


df = pd.read_csv('homeprices.csv')

# SCATTER PLOT TO QUICKLY CHECK CORRELATION
plt.scatter(df.area, df.price)

# Look Good
plt.xlabel("Area(sq ft)")
plt.ylabel("Price($)")
plt.scatter(df.area, df.price, color ='red', marker = '+')

# LINEAR REGRESSION

# 1) CREATE AN REGRESSION OBJECT 

reg = linear_model.LinearRegression()

# 2)  FIT DATA INTO OBJECT
        # Add dataframe array(need to create a df of x axis using[]),
        # add y axis

reg.fit(df[['area']],df.price)

# THIS CREATES THE MODEL TO PREDICT PRICES

#       .predict() TAKES A 2D ARRAY, Matrix
# predict price of home whose area is 3300 sqft

# Make the price into an array and add a RESHAPE to make it into a matrix

#reg.predict(np.array([3300]).reshape(1,1))

test = np.array(3300).reshape(-1,1)


# 

print(reg.predict(test))

# YOU NEED TO PRINT THE PREDICTION IN SPYDER, VARIABLE EXPLORER WONT SHOW IT
# SPYDER WILL TELL YOU WHAT TO DO IF YOU JUST USER .predict(3300)

# PLOTTING THE REGRESSION

predtest = reg.predict(np.array(df[['area']]).reshape(-1,1))

plt.xlabel("Area(sq ft)")
plt.ylabel("Price($)")
plt.scatter(df.area, df.price, color ='red', marker = '+')
# but now we add the regression against it
plt.plot(df[['area']],predtest, color = 'blue')





# PREDICTING A LIST OF PRICES FROM A SEPERATE FILE

areas = pd.read_csv('areas.csv')

print(areas)

# reg has the regression model, so just supply areas

print(reg.predict(areas))  # THIS IS PREDICTING VALUES


p = reg.predict(areas)

# ASSIN P BACK INTO AREAS DF

areas['prices'] = p 
print(areas)

# PUT IN NEW EXCEL

areas.to_csv("Predicted Prices.csv")

# PUT INDEX TO REMOVE INDEX COLUMN EXPORTED

areas.to_csv("Predicted Prices No Index.csv", index = False)

