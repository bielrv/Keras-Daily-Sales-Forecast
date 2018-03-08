#Import libraries
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Import sample dataset containing date, year, day of the week, day of the montn, holydays and sales (i.e. #loans sold)
df = pd.read_csv('Daily_sales.csv',delimiter=';',index_col='date')

dates = df.index
#Split targets and features
Y = df.iloc[:,6]
X = df.iloc[:,0:6]

#Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)

#Count features for modelization
X_num_columns= len(X.columns)

#Define model
model = Sequential()

model.add(Dense(300,
                activation='relu',
                input_dim = X_num_columns))

model.add(Dense(90,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(30,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,
                activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model Created")

#Fit model to training data
model.fit(X_train, y_train, epochs=5000, batch_size=100)
print("Training completed")

#Save trained model
model.save("Sales_model.h5")
print("Sales_model.h5 saved model to disk in ",os.getcwd())

#Predict known daily sales in order to check results
predictions = model.predict(X)
predictions_list = map(lambda x: x[0], predictions)
predictions_series = pd.Series(predictions_list,index=dates)
dates_series =  pd.Series(dates)

#Import dates to be predicted
df_newDates = pd.read_csv('Upcoming_dates.csv',delimiter=';',index_col='date')
print("Upcoming dates imported")

#Predict upcoming sales using trained model and imported upcoming dates
Predicted_sales = model.predict(df_newDates)

#Export predicted sales
new_dates_series=pd.Series(df_newDates.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)
new_predictions_series.to_csv("predicted_sales.csv")
