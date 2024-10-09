import numpy as np
import matplotlib.pyplot as plt


# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data=data

    def __epanechnikov(self, x):
        mod_x_squared=np.sum(x**2, axis=-1)
        return np.maximum(0, 0.75*(1-mod_x_squared))

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        return self.__epanechnikov((x-xi)/self.bandwidth)

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        func_val=[]
        for point in x:
            func_val.append(np.sum(self.epanechnikov_kernel(point, self.data))/(self.bandwidth*self.data.shape[0]))
        return func_val


# Load the data from the NPZ file
data_file = np.load('/home/yash7312/DAI/dai_assn/CS215_assignments/Assignment3/q2/transaction_data.npz')
data = data_file['data']

print(data)
# np.savetxt('points_data.txt', data, delimiter=',', fmt='%f')
# plt.plot(data.T[0], data.T[1], 'o')
# plt.show()
# Data lies in +-6 in both dims

# TODO: Initialize the EpanechnikovKDE class
kde=EpanechnikovKDE()

# TODO: Fit the data
kde.fit(data)

# TODO: Plot the estimated density in a 3D plot
x_range = np.linspace(-7, 8, 100)
y_range = np.linspace(-7, 8, 100)
X, Y = np.meshgrid(x_range, y_range)
xy_points = np.column_stack([X.ravel(), Y.ravel()])
Z=np.array(kde.evaluate(xy_points))
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Density Estimate')
plt.title("Transaction Distribution")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")

# TODO: Save the plot 
plt.savefig("transaction_distribution.png")
plt.show()