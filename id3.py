import pandas as pd
import numpy as np
from collections import Counter

# Load datasets
train_data = pd.read_csv('kdd_train.csv')
test_data = pd.read_csv('kdd_test.csv')

# Preprocessing: Convert 'labels' to 'normal' and 'breach'
train_data['labels'] = train_data['labels'].apply(lambda x: 'normal' if x == 'normal' else 'breach')
test_data['labels'] = test_data['labels'].apply(lambda x: 'normal' if x == 'normal' else 'breach')

# Function to calculate entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count/total) * np.log2(count/total) for count in counts.values() if count > 0)

# Function to find the best feature to split on
def best_split(data, features):
    base_entropy = entropy(data['labels'])
    best_gain = 0
    best_feature = None
    
    for feature in features:
        values = data[feature].unique()
        feature_entropy = 0

        for value in values:
            subset = data[data[feature] == value]
            weight = len(subset) / len(data)
            feature_entropy += weight * entropy(subset['labels'])

        info_gain = base_entropy - feature_entropy
        
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = feature

    return best_feature

# Recursive function to build the decision tree
def build_tree(data, features):
    labels = list(data['labels'])
    # If all labels are the same, return the label
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    # If no more features to split, return the most common label
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]

    # Find the best feature to split
    best_feature = best_split(data, features)
    # If no best feature is found, return the most common label
    if best_feature is None:
        return Counter(labels).most_common(1)[0][0]

    tree = {best_feature: {}}

    # Split the data based on the best feature and build subtrees
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = build_tree(subset, [f for f in features if f != best_feature])
        tree[best_feature][value] = subtree

    return tree

# Function to classify a single instance using the decision tree
def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    feature_value = instance[feature]
    if feature_value in tree[feature]:
        return classify(tree[feature][feature_value], instance)
    else:
        return 'breach'  # Default to 'breach' if value not in tree

# Prepare data for training
features = train_data.columns.drop('labels')
decision_tree = build_tree(train_data, features)

# Predict and evaluate on test data
test_data['predicted'] = test_data.apply(lambda x: classify(decision_tree, x), axis=1)
accuracy = (test_data['predicted'] == test_data['labels']).mean()
print(f"Accuracy: {accuracy:.2f}")

print(test_data['labels'].value_counts())
print(test_data['predicted'].value_counts())
print(test_data[['labels', 'predicted']])
