import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 2 * np.pi, 100)  # X-axis: 100 points from 0 to 2π
times = np.linspace(0, 10, 20)  # Time points: 20 points from 0 to 10
frequencies = np.linspace(1, 5, len(times))  # Frequencies changing over time


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')


for t, f in zip(times, frequencies):
    y = np.sin(f * x)  # Generate a sine wave with varying frequency
    ax.plot(x, y, zs=t, zdir='y', label=f"t={t:.1f}")


ax.set_xlabel('X-axis')
ax.set_ylabel('Time')
ax.set_zlabel('Amplitude')
ax.set_title('Waterfall Plot Example')
ax.view_init(elev=20, azim=-60)  # Adjust view angle

plt.show()
