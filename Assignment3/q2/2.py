import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data=np.array(data)

    def __epanechnikov(self, x):      
        mod_x_squared=np.sum(x**2, axis=-1)
        return np.maximum(0, 0.75*(1-mod_x_squared))

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        assert(np.size(x) == 2)
        
        return self.__epanechnikov((x-xi)/self.bandwidth)

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        func_val=[]
        for point in x:
            func_val.append(np.sum(self.epanechnikov_kernel(point, self.data))/(self.bandwidth*self.data.shape[0]))
        
        return np.array(func_val)


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# np.savetxt('data.txt', data, delimiter=',', fmt='%f')
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
# X, Y = np.meshgrid(x_range, y_range)
# xy_points = np.column_stack([X.ravel(), Y.ravel()])
# Z=np.array(kde.evaluate(xy_points))
# Z = Z.reshape(X.shape)
xy_points = np.array([(x, y) for x in x_range for y in y_range])  
X = xy_points[:, 0].reshape(len(y_range), len(x_range))
Y = xy_points[:, 1].reshape(len(y_range), len(x_range))
Z = kde.evaluate(xy_points).reshape(len(x_range), len(y_range))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,  cmap='viridis', edgecolor='none')
ax.set_title('Distribution of Transactions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Estimated Probability Density')
ax.grid(True)

# TODO: Save the plot 
plt.savefig("transaction_distribution.png")
plt.show()