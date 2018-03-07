#Importing libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split

#Importing sample dataset containing date, year,... #loans
df = pd.read_csv('Daily_sales.csv',delimiter=';',index_col='date')

#Sanity checks
df.head(3)
df.tail(3)

dates = df.index
#Splitting targets and features
Y = df.iloc[:,6]
X = df.iloc[:,0:6]

#Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)

#Counting number of features for modelization
X_num_columns= len(X.columns)

#Defining model

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

model.compile(optimizer='adam', loss='mse',metrics = ['accuracy'])

print("Model Created")

model.fit(X_train, y_train, epochs=5000, batch_size=100)
print("Training completed")

#Importing dates to be predicted
