# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Sample data setup (replace this with your dataset)
# Here we're creating a synthetic dataset for illustration purposes
data = {
    'income': [50000, 60000, 30000, 80000, 20000, 70000, 40000, 100000, 30000, 75000],
    'age': [25, 45, 35, 50, 23, 37, 29, 55, 33, 40],
    'loan_amount': [10000, 15000, 7000, 20000, 5000, 12000, 9000, 25000, 8000, 14000],
    'credit_history': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],  # 1 = good, 0 = bad
    'employment_years': [3, 10, 4, 15, 2, 5, 3, 20, 4, 6],
    'credit_score': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # 1 = good, 0 = bad
}
df = pd.DataFrame(data)

# Define features and target
X = df[['income', 'age', 'loan_amount', 'credit_history', 'employment_years']]
y = df['credit_score']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
lr_classifier = LogisticRegression()

# Train the model
lr_classifier.fit(X_train, y_train)

# Make predictions
y_pred = lr_classifier.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
