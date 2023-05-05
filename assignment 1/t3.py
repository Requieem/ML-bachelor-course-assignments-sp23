import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
data = np.load('data/data.npz')
x = data['x']
y = data['y']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[2]),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(x_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)