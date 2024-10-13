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



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # max_range = max(np.max(position_store[:,:,0]) - np.min(position_store[:,:,0]), np.max(position_store[:,:,1]) - np.min(position_store[:,:,1]), np.max(position_store[:,:,2]) - np.min(position_store[:,:,2]))
    # mid_x = np.mean(position_store[:, :, 0])
    # mid_y = np.mean(position_store[:, :, 1])
    # mid_z = np.mean(position_store[:, :, 2])

    # ax.set_xlim(mid_x - 0.6 * max_range, mid_x + 0.6 * max_range)
    # ax.set_ylim(mid_y - 0.6 * max_range, mid_y + 0.6 * max_range)
    # ax.set_zlim(mid_z - 0.6 * max_range, mid_z + 0.6 * max_range)

    # for i in range(no_of_bodies):
    #     Color = np.random.rand(3,)
    #     ax.plot(position_store[:,i,0],position_store[:,i,1], position_store[:,i,2], c=Color ,label=f'Body {i+1}')
    # ax.set_title("Trajectories of diff bodies")
    # ax.set_xlabel('X-coordinate (AU)')
    # ax.set_ylabel('Y-coordinate (AU)')
    # ax.set_zlabel('Z-coordinate (AU)')
    # ax.grid(True)
    # if (legend==True):
    #     ax.legend()
    # if (save==True):
    #     plt.savefig(f'{sys_time}_Trajectory".jpg')
    # plt.show()