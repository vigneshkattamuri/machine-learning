import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data - sales over a period (e.g., months)
data = {
    'month': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # Months (e.g., Jan to Dec)
    'sales': np.array([100, 120, 130, 150, 160, 180, 200, 210, 220, 230, 240, 250])  # Monthly sales in units
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the independent (X) and dependent (y) variables
X = df[['month']]  # Predictor variable (month)
y = df['sales']    # Response variable (sales)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
months_future = np.array([[13], [14], [15], [16], [17], [18]])  # Next six months
future_sales = model.predict(months_future)

# Print future sales predictions
for i, sales in enumerate(future_sales, start=13):
    print(f"Predicted sales for month {i}: {sales:.2f} units")

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', label='Trend Line')
plt.scatter(months_future, future_sales, color='green', label='Predicted Future Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()
