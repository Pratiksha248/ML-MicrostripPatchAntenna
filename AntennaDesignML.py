#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Import necessary libraries for data manipulation and machine learning tasks
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
data = pd.read_csv('/Users/sahoo/Desktop/DatasetAntenna.csv') #Read the CSV file
#Display the first few rows of the DataFrame
print("Preview of the dataset:") 
print(data.head())
#Display the number of unique values wrt. each column
print("\nNumber of unique values in each column:") 
print(data.nunique())
#Feature Engineering and Data Preprocessing
X1 = data.drop(['S11(Mag (dB))'], axis=1)  #Extract the features
y1 = data['S11(Mag (dB))']  #Extract the target variable
#Split the data into 70% training and 30% testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
#Define a list of regression models
models = [LinearRegression, ElasticNet, Lasso, DecisionTreeRegressor, RandomForestRegressor]
#Model Training and Evaluation
print("\nResults for the set of features:")
for model in models:
    #Create an instance of the current model
    reg = model()
    #Train the model using the training data
    reg.fit(X_train1, y_train1)
    #Make predictions on the test data using the trained model
    pred1 = reg.predict(X_test1)
    #Calculate RMSE and R2 score for the predictions
    err1 = mean_squared_error(y_test1, pred1) ** 0.5
    r2_value = np.mean(r2_score(y_test1, pred1))
    #Predict for the entire dataset
    data[f'Predicted_{model.__name__}'] = reg.predict(X1)  
    #Print evaluation metrics for the current model
    print(f'{model.__name__} Model:')
    print(f'RMSE: {err1}')
    print(f'R2 Score: {r2_value}')
    print(f'Accuracy: {r2_value*100}' '%')
    print('~' * 50)  #Separator
#Print the first few rows of the DataFrame with the predicted values
print("\nPredicted values for each algorithm:")
print(data.head())

