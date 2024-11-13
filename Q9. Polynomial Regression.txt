import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generating a synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random X values between 0 and 10
y = 2.5 * X**2 + 3 * X + np.random.randn(100, 1) * 10  # Quadratic relationship with some noise

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial Regression Model
poly_features = PolynomialFeatures(degree=2)  # degree=2 for quadratic relationship
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Performance Metrics
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print(f"Linear Regression Mean Squared Error: {linear_mse:.2f}")
print(f"Linear Regression R^2 Score: {linear_r2:.2f}")
print(f"Polynomial Regression Mean Squared Error: {poly_mse:.2f}")
print(f"Polynomial Regression R^2 Score: {poly_r2:.2f}")

# Visualization
plt.figure(figsize=(14, 6))

# Scatter plot of actual data
plt.scatter(X, y, color="blue", label="Actual data", alpha=0.5)

# Plotting Linear Regression predictions
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred_linear_full = linear_model.predict(X_range)
plt.plot(X_range, y_pred_linear_full, color="green", label="Linear Regression Prediction")

# Plotting Polynomial Regression predictions
X_range_poly = poly_features.transform(X_range)
y_pred_poly_full = poly_model.predict(X_range_poly)
plt.plot(X_range, y_pred_poly_full, color="red", label="Polynomial Regression Prediction")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
