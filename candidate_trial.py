import pandas as pd

# Load the datasets
train_data = pd.read_csv('kdd_train.csv')
test_data = pd.read_csv('kdd_test.csv')

# Separate features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Function to convert categorical features to a simplified form
def encode_features(X):
    """Encode categorical features to numeric values."""
    for col in X.columns:
        X[col] = X[col].astype('category').cat.codes
    return X

# Encode the training and test data
X_train_encoded = encode_features(X_train.copy())
X_test_encoded = encode_features(X_test.copy())

def is_consistent(hypothesis, example):
    """Check if the hypothesis is consistent with the example."""
    for h, e in zip(hypothesis, example):
        if h != '?' and h != e:
            return False
    return True

def generalize(hypothesis, example):
    """Generalize the hypothesis based on the example."""
    return [e if h == '?' else h for h, e in zip(hypothesis, example)]

def specialize(hypothesis, example):
    """Specialize the hypothesis based on the example."""
    new_hypotheses = []
    for i in range(len(hypothesis)):
        if hypothesis[i] != example[i]:
            new_hypothesis = hypothesis.copy()
            new_hypothesis[i] = example[i]  # Specialize to this value
            new_hypotheses.append(new_hypothesis)
    return new_hypotheses

def candidate_elimination(X, y):
    """Perform the Candidate Elimination algorithm."""
    S = [['?'] * X.shape[1]]  # Specific hypothesis
    G = [['?'] * X.shape[1]]  # General hypothesis

    for i in range(len(X)):
        x = X.iloc[i].values
        label = y.iloc[i]

        if label == 'normal':
            # Generalize S
            S = [generalize(s, x) for s in S if is_consistent(s, x)]
            # Remove hypotheses in G that are inconsistent with x
            G = [g for g in G if is_consistent(g, x)]
        else:
            # Remove hypotheses in S that are consistent with x
            S = [s for s in S if not is_consistent(s, x)]
            # Specialize G with the negative example
            new_G = []
            for g in G:
                if not is_consistent(g, x):
                    new_G.extend(specialize(g, x))
                else:
                    new_G.append(g)
            G = new_G

    return S, G

# Run the Candidate Elimination algorithm
S_final, G_final = candidate_elimination(X_train_encoded, y_train)

# Evaluate the model on the test set
correct = 0
total = len(X_test_encoded)

for i in range(total):
    x = X_test_encoded.iloc[i].values
    y = y_test.iloc[i]

    covered_by_S = any(is_consistent(s, x) for s in S_final)
    not_covered_by_G = all(not is_consistent(g, x) for g in G_final)

    # Check if the prediction is correct
    if (covered_by_S and y == 'normal') or (not covered_by_S and y != 'normal'):
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")