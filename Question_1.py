# This is the code for assignment 3 Question 2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Generate Gaussian Data for 4 classes
def generate_data(num_samples, means, covariances, priors):
    data = []
    labels = []
    for i in range(num_samples):
        class_label = np.random.choice(len(priors), p=priors)
        sample = np.random.multivariate_normal(means[class_label], covariances[class_label])
        data.append(sample)
        labels.append(class_label)
    return np.array(data), np.array(labels)

# Define means, covariances, and priors for 4 classes
means = [np.array([2, 2, 2]), np.array([-2, -2, -2]), np.array([2, -2, 2]), np.array([-2, 2, -2])]
covariances = [np.eye(3) for _ in range(4)]
priors = [0.25, 0.25, 0.25, 0.25]

# Generate training and test data
num_train_samples = 10000   # generate 100,500,1000, 5000,10000,samples.
X_train, y_train = generate_data(num_train_samples, means, covariances, priors)


num_test_samples = 100000
X_test, y_test = generate_data(num_test_samples, means, covariances, priors)

# Step 2: Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Step 3: Train the MLP model
def train_mlp(X_train, y_train, hidden_size, num_epochs=100, learning_rate=0.001):
    input_size = X_train.shape[1]
    output_size = 4  # 4 classes
    model = MLP(input_size, hidden_size, output_size).to(device)

    # Convert data to PyTorch tensors and move to device (GPU/CPU)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model

# Step 4: Evaluate the MLP model
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())  # Move tensors to CPU to calculate accuracy

    return accuracy

# Step 5: Cross-validation to select the best number of perceptrons (hidden units)
def cross_validate(X, y, hidden_sizes, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True)
    best_hidden_size = None
    best_accuracy = 0

    for hidden_size in hidden_sizes:
        fold_accuracies = []
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            model = train_mlp(X_train_fold, y_train_fold, hidden_size)
            accuracy = evaluate_model(model, X_val_fold, y_val_fold)
            fold_accuracies.append(accuracy)
        print(f"Fold accuracies for hidden size {hidden_size}: {fold_accuracies}")
        avg_accuracy = np.mean(fold_accuracies)
        print(f"Mean accuracies for hidden size {hidden_size}: {avg_accuracy}")
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_hidden_size = hidden_size
    return best_hidden_size

# Step 6: Train the final models on the full training set with selected hidden sizes
hidden_sizes = [5, 10, 20, 50, 100, 150]
best_hidden_size = cross_validate(X_train, y_train, hidden_sizes)
print(f"Best hidden size: {best_hidden_size}")
final_model = train_mlp(X_train, y_train, best_hidden_size)

# Step 7: Evaluate on test set
test_accuracy = evaluate_model(final_model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 8: Plot the results (this part depends on your experiment)
