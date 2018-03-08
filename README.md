# Keras Daily Sales Forecast

Using Keras and Tensorflow to predicted future daily sales.


## Desciption
A Daily Sales Forecast using Keras with Tensorflow is performed. The model take into account Day of the Week, Day of the Month, Week of the Month, Week of the Year, Year of the Month and could be easily improved using Holidays. Accuracy obtained over 92%. The current NN model has proved to improve performance over ARIMA models.

## Model Parameters

### Structure
5 layers:
 - Input layer: 300 relu neurons with no dropout
 - 1st hidden layer: 90 relu neurons with 20% dropout
 - 2nd hidden layer: 30 relu neurons with 20% dropout
 - 3rd hidden layer: 7 relu neurons with 20% dropout
 - Output layer: 1 linear relu

 optimizer used: adam
 loss measured using mean squared error

### Training
 - 5000 epochs using batch size 100

## Model Accuracy
Accuracy : 0.92

### Train
Accuracy (Train): 0.95

### Test
Accuracy (Test):  0.92

### Production
Accuracy (Production): 0.90