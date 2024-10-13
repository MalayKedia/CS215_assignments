import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv')
x_data = data.drop(columns=['id', 'yield']).values[:2000]
y_data = data['yield'].values[:2000]

test = pd.read_csv('data/test.csv')
x_test = test.drop(columns=['id']).values[:2000]


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
        dist_sq = np.sum((xi-x)**2, axis=-1)/ self.bandwidth
        print(dist_sq)
        return self.__uniform_kernel(dist_sq)

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

def risk_val(func):
    bandwidths = np.linspace(0, 10, 50)

    # Store risks for each bandwidth
    risks = []

    # Loop over bandwidths to compute risk
    for bandwidth in bandwidths:
        # Create an instance of the UniformKDE class
        kde = func(bandwidth=bandwidth)

        # Fit the model with the data
        kde.fit(x_data, y_data)

        # Calculate risk (mean squared error)
        risk = calculate_risk(kde, x_data, y_data)
        risks.append(risk)
        
    return bandwidths, risks, bandwidths[np.argmin(risks)]

bandwidths_Ukde, risks_Ukde, best_bandwidth_Ukde = risk_val(UniformKDE)

fig, axs = plt.subplots(figsize=(12, 6))

# Fit the model with the best bandwidth
kde = UniformKDE(bandwidth=best_bandwidth_Ukde)
kde.fit(x_data, y_data)

# Predict the yield
y_pred = kde.predict(x_test)

def MAE(data, pred):
    value = np.sum(np.abs(data - pred)) / len(data)
    return value

print(MAE(y_data, y_pred))

# Plot expected yield vs predicted yield
axs.scatter(y_data, y_pred, color='blue', label='Predicted vs Expected')
axs.plot([min(y_data), max(y_data)], [min(y_data), max(y_data)], color='red', label='Ideal fit')
axs.set_title('Expected Yield vs Predicted Yield')
axs.set_xlabel('Expected Yield')
axs.set_ylabel('Predicted Yield')
axs.legend()

plt.show()
