import pandas as pd
import numpy as np

# Define a function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = sum([-(counts[i] / sum(counts)) * np.log2(counts[i] / sum(counts)) for i in range(len(elements))])
    return entropy_value

# Define a function to calculate information gain
def info_gain(data, split_attribute_name, target_name="Class"):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = sum(
        [(counts[i] / sum(counts)) * entropy(data[data[split_attribute_name] == values[i]][target_name]) 
         for i in range(len(values))]
    )
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Define the ID3 algorithm to build the decision tree
def id3(data, original_data, features, target_attribute_name="Class", parent_node_class=None):
    # If all target values have the same value, return that value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If dataset is empty, return the most frequent target value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # If features are empty, return the most frequent target value
    elif len(features) == 0:
        return parent_node_class

    # Otherwise, calculate the information gain for each feature
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        # Grow the tree
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree

# Function to classify a new sample
def classify(instance, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute]:
        return classify(instance, tree[attribute][instance[attribute]])
    else:
        return "Unknown"

# Load a sample dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Define the features and target column
features = data.columns[:-1]
target = 'Class'

# Build the decision tree
tree = id3(data, data, features, target)

print("Decision Tree:", tree)

# Classify a new sample
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
classification = classify(sample, tree)
print("Classification for new sample:", classification)
