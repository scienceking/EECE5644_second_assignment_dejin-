# This is the code for assignment 3 Question 2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar

# Step 1: Define the true GMM parameters with significant overlap and different weights
def generate_gmm_data(n_samples, means, covariances, weights):
    n_components = len(weights)
    data = []
    component_choices = np.random.choice(n_components, size=n_samples, p=weights)
    for i in range(n_samples):
        component = component_choices[i]
        sample = np.random.multivariate_normal(means[component], covariances[component])
        data.append(sample)
    return np.array(data)

# Modify means and covariances for significant overlap and different probabilities
means = [
    [0, 0],   # Component 1 mean
    [1, 1],   # Component 2 mean (overlaps with Component 1)
    [6, 0],   # Component 3 mean
    [3, 8]    # Component 4 mean
]

# Covariance matrices for each Gaussian component
covariances = [
    np.array([[1, 0.5], [0.5, 1]]),     # Component 1 covariance
    np.array([[1, 0.4], [0.4, 1]]),     # Component 2 covariance
    np.array([[0.8, 0.3], [0.3, 0.8]]), # Component 3 covariance
    np.array([[1.2, 0.6], [0.6, 1.2]])  # Component 4 covariance
]

# Component weights (unequal, sum to 1)
weights = [0.4, 0.3, 0.2, 0.1]

# Sample sizes
sample_sizes = [10, 100, 1000]

# Number of experiments
n_experiments =100

# Number of folds for cross-validation
n_folds = 10

# Maximum number of components to evaluate
max_components = 6


# Fit GMM with cross-validation
def fit_gmm_with_cross_validation(data, max_components=max_components, n_folds=n_folds):
    log_likelihoods = []

    # Create K-Fold splits
    kf = KFold(n_splits=min(n_folds, len(data)))

    avg_log_likelihoods = []

    for n_components in range(1, max_components + 1):
        fold_log_likelihoods = []

        # Loop over each fold
        for train_idx, val_idx in kf.split(data):
            train_data, val_data = data[train_idx], data[val_idx]

            # Skip fitting if the number of training samples is less than n_components
            if len(train_data) < n_components:
                continue

            # Fit GMM and calculate log-likelihood
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(train_data)
            fold_log_likelihoods.append(gmm.score(val_data))

        if fold_log_likelihoods:
            # Store average log-likelihood for this number of components
            avg_log_likelihoods.append(np.mean(fold_log_likelihoods))
        else:
            # If no valid fold, append a very low score to prevent this component number from being selected
            avg_log_likelihoods.append(-np.inf)

    # Determine the best model order based on the highest log-likelihood
    best_model_order = np.argmax(avg_log_likelihoods) + 1
    return best_model_order, avg_log_likelihoods

# Initialize results dictionary
results = {n: {'best_model_orders': [], 'log_likelihoods': []} for n in sample_sizes}

# Step 4: Repeat the experiment 100 times and collect results with a progress bar
for _ in tqdm(range(n_experiments), desc="Running Experiments"):
    for n_samples in sample_sizes:
        # Generate a new dataset for each experiment
        data = generate_gmm_data(n_samples, means, covariances, weights)



        best_model_order, log_likelihoods = fit_gmm_with_cross_validation(data)
        results[n_samples]['best_model_orders'].append(best_model_order)
        results[n_samples]['log_likelihoods'].append(log_likelihoods)

# Step 5: Report and visualize the results
for n_samples in sample_sizes:
    unique, counts = np.unique(results[n_samples]['best_model_orders'], return_counts=True)
    print(f"Sample size {n_samples}:")
    for u, c in zip(unique, counts):
        print(f"  Model order {u}: {c} times selected ({c / n_experiments:.2%})")

# Step 6: Visualize the results
plt.figure(figsize=(12, 6))

# Plot histograms of best model orders
for idx, n_samples in enumerate(sample_sizes):
    plt.subplot(1, 3, idx + 1)
    plt.hist(results[n_samples]['best_model_orders'], bins=np.arange(1, max_components + 2) - 0.5, edgecolor='black')
    plt.title(f'{n_samples} Samples')
    plt.xlabel('Model Order')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max_components + 1))

plt.tight_layout()
plt.show()

# Plot average log-likelihoods per model order
plt.figure(figsize=(12, 6))
for n_samples in sample_sizes:
    log_likelihoods = np.array(results[n_samples]['log_likelihoods'])
    mean_log_likelihoods = np.mean(log_likelihoods, axis=0)
    std_log_likelihoods = np.std(log_likelihoods, axis=0)
    plt.errorbar(range(1, max_components + 1), mean_log_likelihoods, label=f'{n_samples} Samples', capsize=5)

plt.xlabel('Model Order')
plt.ylabel('Average Log-Likelihood')
plt.title('Average Log-Likelihood vs Model Order')
#plt.ylim(-10,2)
plt.legend()
plt.show()
