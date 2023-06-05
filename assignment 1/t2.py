import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

# Fit the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, verbose=0)

# Compute the predictions on the test set
y_pred = model.predict(x_test).flatten()

# Compute the mean squared error on the test set
mse = mean_squared_error(y_test, y_pred)

# Print the test MSE
print("Test MSE:", mse)

# Compare with the linear regression model from task 1
linear_mse = 0.152  # Assuming this is the MSE from the linear regression model in task 1
t_statistic, p_value = stats.ttest_ind(y_test, y_pred)
print("p-value of the t-test:", p_value)
if p_value < 0.05:
    print("The non-linear model is statistically better.")
else:
    print("The linear model is statistically better.")