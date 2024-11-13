# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace with your dataset)
# Sample data for illustration purposes
data = {
    'square_footage': [2000, 1500, 2500, 1200, 3000, 1600, 2200, 1800, 3500, 1400],
    'num_bedrooms': [3, 2, 4, 2, 5, 3, 3, 2, 5, 2],
    'age_of_house': [10, 20, 5, 30, 2, 15, 10, 25, 3, 20],
    'price': [500000, 300000, 550000, 250000, 600000, 320000, 480000, 330000, 750000, 280000]
}
df = pd.DataFrame(data)

# Define features and target variable
X = df[['square_footage', 'num_bedrooms', 'age_of_house']]
y = df['price']

# Standardize the features
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
