# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the car dataset (replace with your dataset)
# Assuming a dataset with features like 'mileage', 'year', 'engine_size', and 'price'
data = {
    'mileage': [15000, 30000, 60000, 12000, 100000, 25000, 50000, 40000, 70000, 20000],
    'year': [2017, 2016, 2015, 2018, 2014, 2016, 2015, 2017, 2013, 2018],
    'engine_size': [1.5, 2.0, 1.6, 1.2, 2.5, 1.8, 1.5, 1.4, 2.2, 1.6],
    'price': [20000, 18000, 15000, 22000, 12000, 19000, 16000, 21000, 11000, 23000]
}
df = pd.DataFrame(data)

# Define features and target variable
X = df[['mileage', 'year', 'engine_size']]
y = df['price']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
