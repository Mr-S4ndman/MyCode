import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



frame = pd.read_csv('спички.csv')
array = frame.to_numpy()
y = array[0]-1
x = array[1:].T
y_cat = keras.utils.to_categorical(y, 30)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, train_size=0.67, random_state=123)

model = keras.Sequential([
    Flatten(input_shape=(300,1)),
    Dense(50, activation = 'relu'),
    Dense(300, activation = 'relu'),
    Dense(30, activation = 'softmax')
    ])
# print(model.summary()) 
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=5, epochs=500, validation_split=0)

model.evaluate(x_test, y_test)

y_pred = model.predict(x_train)

a = mean_squared_error(y_pred, y_train)

frame2 = pd.read_csv('тест1.csv')
array2 = frame2.to_numpy()
y2 = array2[0]-1
x2 = array2[1:].T

testpred = model.predict(x2)


# print(testpred)
