import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the data
data = np.load('data/data.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Define the family of models
def model(x, theta):
    return theta[0] + theta[1] * x[:, 0] + theta[2] * x[:, 1] + \
           theta[3] * np.sin(x[:, 1]) + theta[4] * x[:, 0] * x[:, 1]

# Fit the model
model_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model_pipeline.fit(x_train, y_train)
theta_hat = np.concatenate(([model_pipeline.named_steps['linearregression'].intercept_], model_pipeline.named_steps['linearregression'].coef_))

# Compute the predictions on the test set
y_pred = model(x_test, theta_hat)

# Compute the mean squared error on the test set
mse = mean_squared_error(y_test, y_pred)

# Print the formula of the model
formula = f"y = {theta_hat[0]:.3f} + {theta_hat[1]:.3f} * x1 + {theta_hat[2]:.3f} * x2 + {theta_hat[3]:.3f} * sin(x2) + {theta_hat[4]:.3f} * x1 * x2"
print("Formula of the model:", formula)

# Print the MSE
print("Test MSE:", mse)