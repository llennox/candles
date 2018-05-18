import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

df = pd.read_csv("BTCRLC.csv")


df = df.drop(['start', "open", "id", "vwp", ], axis=1)

df['trades'] = df['trades'].astype('float64')
print(df.columns)


prediction_minutes = 60

df_train = df[:len(df)-prediction_minutes]
df_test= df[len(df)-prediction_minutes:]
print(df_test[0:5])

training_set = df_train.values
#print(training_set[0])
training_set = min_max_scaler.fit_transform(training_set)
#print(training_set[0])
#print(training_set)

x_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
#print(x_train.shape)
#print(len(y_train), 'here')


x_train = np.reshape(x_train, (len(x_train), 5, 1))




num_units = 5
activation_function = 'sigmoid'
optimizer = 'adam'
loss_function = 'mean_squared_error'
batch_size = 32
num_epochs = 1

# Initialize the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = num_units, activation = activation_function, input_shape=(5, 1)))

# Adding the output layer
regressor.add(Dense(units = 5))

# Compiling the RNN
regressor.compile(optimizer = optimizer, loss = loss_function)

# Using the training set to train the model
#print(x_train.shape)
#print(y_train.shape)
#print(len(y_train), 'and again')
#print(len(x_train))

regressor.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)

test_set = df_test.values
print(test_set.shape, 'test_set')
print(test_set[0:5], 'hi')
test_set = min_max_scaler.fit_transform(test_set)

inputs = np.reshape(test_set, (len(test_set), 5, 1))
#print(inputs.shape, 'input')
predicted_price = regressor.predict(inputs)
#print(predicted_price.shape, 'predicted price shape')
#print(predicted_price[0:5], 'before')
predicted_price = min_max_scaler.inverse_transform(predicted_price)


print(predicted_price[0:5], 'after')
print(predicted_price.shape)
quit()



plt.figure(figsize=(25, 25), dpi=80, facecolor = 'w', edgecolor = 'k')

plt.plot(test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize = 40)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize = 40)
plt.legend(loc = 'best')
plt.show()
