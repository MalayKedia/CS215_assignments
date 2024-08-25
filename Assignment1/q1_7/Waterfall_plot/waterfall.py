# import matplotlib.pyplot as plt
# import numpy as np

# # Parameters for synthetic data
# n_timepoints = 50  # Number of time points
# n_frequencies = 100  # Number of frequency points
# time = np.linspace(0, 1, n_timepoints)  # Simulated time axis (0 to 1 second)
# frequency = np.linspace(0, 500, n_frequencies)  # Frequency axis (0 to 500 Hz)

# # Generate synthetic spectrogram data (linear magnitude)
# spectrogram = np.sin(2 * np.pi * frequency[:, np.newaxis] * time) * np.exp(-time * 3)

# # Plotting
# plt.figure(figsize=(10, 6))
# offset = 5  # Vertical offset between lines

# for i in range(n_timepoints):
#     # Each time slice as a vertical stack, offset by its index
#     plt.plot(frequency, spectrogram[:, i] + i * offset, color='blue')
#     # Optionally fill under the curve for better visual effect
#     plt.fill_between(frequency, i * offset, spectrogram[:, i] + i * offset, color='blue', alpha=0.3)

# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Linear Magnitude (Uncal.) + Offset')
# plt.title('2D Waterfall Plot of Spectrogram Data')
# plt.grid(True)
# plt.show()

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
