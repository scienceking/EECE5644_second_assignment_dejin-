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

# Step 2: Generate training and test data
train_sample_sizes = [100, 500, 1000, 5000, 10000]  # List of different sample sizes
num_test_samples = 100000
X_test, y_test = generate_data(num_test_samples, means, covariances, priors)


# Step 3: Define MLP model
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


# Step 4: Train the MLP model
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


# Step 5: Evaluate the MLP model
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())  # Move tensors to CPU to calculate accuracy

    return accuracy


# Step 6: Train and evaluate models for different training sample sizes
hidden_size = 100  # Using a fixed hidden size for simplicity
empirical_errors = []

for num_train_samples in train_sample_sizes:
    # Generate training data
    X_train, y_train = generate_data(num_train_samples, means, covariances, priors)

    # Train the model
    model = train_mlp(X_train, y_train, hidden_size)

    # Evaluate the model
    test_accuracy = evaluate_model(model, X_test, y_test)
    p_error = 1 - test_accuracy
    empirical_errors.append(p_error)
    print(f"Number of training samples: {num_train_samples}, Test P(error): {p_error:.4f}")

# Step 7: Theoretical Optimal Classifier (assumed given)
theoretical_p_error = 0.024  # Assume theoretical optimal classifier error

# Step 8: Plot the results
plt.figure(figsize=(10, 6))
plt.semilogx(train_sample_sizes, empirical_errors, marker='o', linestyle='-', color='b', label='Empirical P(error)')
plt.axhline(y=theoretical_p_error, color='r', linestyle='--', label='Theoretical Optimal P(error)')
plt.xlabel('Number of Training Samples (log scale)')
plt.ylabel('P(error)')
plt.title('Empirical P(error) vs Number of Training Samples')
plt.legend()
plt.grid(True)
plt.show()
