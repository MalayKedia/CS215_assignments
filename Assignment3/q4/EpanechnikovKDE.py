import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = pd.read_csv('glass.txt', delimiter=r'\s+')
data = data_file[['RI', 'Al']]

#shuffle data
data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)

x_data = data['RI'].values
y_data = data['Al'].values


class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        """Initialize the KDE model with the given bandwidth."""
        self.bandwidth = bandwidth
        self.data = None
        self.y_data = None

    def fit(self, x_data, y_data):
        """Fit the KDE model with the given x and y data."""
        self.data = x_data
        self.y_data = y_data

    def __epanechnikov(self, x):
        # mod_x_squared=np.sum(x**2, axis=-1)
        return np.maximum(0, 0.75*(1-x**2))

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        
        return self.__epanechnikov((x-xi)/self.bandwidth)
        
    

    def predict(self, x):
        """Evaluate the Nadaraya-Watson estimator at points x."""
        func_val = []
        for point in x:
            weights = self.epanechnikov_kernel(point, self.data)
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
    
    # Create an instance of the EpanechnikovKDE class
    kde = EpanechnikovKDE(bandwidth=bandwidth)

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
kde = EpanechnikovKDE(bandwidth=best_bandwidth / 20)
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
kde = EpanechnikovKDE(bandwidth=best_bandwidth)
kde.fit(x_data, y_data)
y_fit = kde.predict(x_fit)
axs[1, 0].scatter(x_data, y_data, color='blue', label='Data points')
axs[1, 0].plot(x_fit, y_fit, color='red', label='Nadaraya-Watson fit (uniform kernel)')
axs[1, 0].set_title('Perfect')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')
axs[1, 0].legend()

# undersmooth
kde = EpanechnikovKDE(bandwidth=best_bandwidth * 20)
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

plt.savefig("epanechnikov_kernel_regression.png")

print(f"bandwidth corresponding to minimum estimated risk is {best_bandwidth}")