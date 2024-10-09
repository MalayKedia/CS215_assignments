import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = pd.read_csv('glass.txt', delimiter=r'\s+')
data = data_file[['RI', 'Al']]
data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)
x_data = data['RI'].values
y_data = data['Al'].values


class UniformKDE:
    def __init__(self, bandwidth=1.0):
        """Initialize the KDE model with the given bandwidth."""
        self.bandwidth = bandwidth
        self.data = None
        self.y_data = None

    def fit(self, x_data, y_data):
        """Fit the KDE model with the given x and y data."""
        self.data = x_data
        self.y_data = y_data

    def __uniform_kernel(self, x):
        """Uniform kernel function."""
        return np.where(np.abs(x) <= 0.5, 1, 0)

    def kernel(self, x, xi):
        """Applies the uniform kernel to the data."""
        return self.__uniform_kernel((x - xi) / self.bandwidth)
    

    def predict(self, x):
        """Evaluate the Nadaraya-Watson estimator at points x."""
        func_val = []
        for point in x:
            weights = self.kernel(point, self.data)
            numerator = np.sum(weights * self.y_data)
            denominator = np.sum(weights)
            
            if denominator > 0:
                func_val.append(numerator / denominator)
            else:
                func_val.append(0)  # Handle division by zero case
        return np.array(func_val)


def calculate_risk(kde, x_data, y_true):
    """Calculate the mean squared error (MSE) as risk."""
    y_pred = kde.predict(x_data)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse



bandwidths = np.linspace(0, 10, 50)

# Store risks for each bandwidth
risks = []

# Loop over bandwidths to compute risk
for bandwidth in bandwidths:
    # Create an instance of the UniformKDE class
    kde = UniformKDE(bandwidth=bandwidth)

    # Fit the model with the data
    kde.fit(x_data, y_data)

    # Calculate risk (mean squared error)
    risk = calculate_risk(kde, x_data, y_data)
    risks.append(risk)
    
best_bandwidth = bandwidths[np.argmin(risks)]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot risk vs bandwidth
axs[0, 0].plot(bandwidths, risks, marker='o')
axs[0, 0].set_title('Risk vs Bandwidth (Uniform Kernel)')
axs[0, 0].set_xlabel('Bandwidth')
axs[0, 0].set_ylabel('Risk (MSE)')
axs[0, 0].grid(True)

# Oversmooth
kde = UniformKDE(bandwidth=best_bandwidth / 20)
kde.fit(x_data, y_data)
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = kde.predict(x_fit)
axs[0, 1].scatter(x_data, y_data, color='blue', label='Data points')
axs[0, 1].plot(x_fit, y_fit, color='red', label='Nadaraya-Watson fit (uniform kernel)')
axs[0, 1].set_title('Oversmooth')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].legend()

# perfect
kde = UniformKDE(bandwidth=best_bandwidth)
kde.fit(x_data, y_data)
y_fit = kde.predict(x_fit)
axs[1, 0].scatter(x_data, y_data, color='blue', label='Data points')
axs[1, 0].plot(x_fit, y_fit, color='red', label='Nadaraya-Watson fit (uniform kernel)')
axs[1, 0].set_title('Perfect')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')
axs[1, 0].legend()

# undersmooth
kde = UniformKDE(bandwidth=best_bandwidth * 20)
kde.fit(x_data, y_data)
y_fit = kde.predict(x_fit)
axs[1, 1].scatter(x_data, y_data, color='blue', label='Data points')
axs[1, 1].plot(x_fit, y_fit, color='red', label='Nadaraya-Watson fit (uniform kernel)')
axs[1, 1].set_title('Undersmooth')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

plt.savefig("uniform_kernel_regression.png")

print(f"bandwidth corresponding to minimum estimated risk is {best_bandwidth}")