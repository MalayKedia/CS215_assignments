import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def sample(loc, scale):
    uniform_sample = np.random.uniform(0, 1)
    gaussian_sample = norm.ppf(uniform_sample, loc=loc, scale=scale)
    return gaussian_sample

N = 10**5
params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]

gaussian_samples=[]
for i in range(N):
    entry=[]
    for param in params:
        entry.append(sample(param[0], param[1]))
    gaussian_samples.append(entry)
    
gaussian_samples=np.array(gaussian_samples).T

plt.figure(figsize=(10, 8))
plt.hist(gaussian_samples[0], bins=500, density=True, alpha=0.5, label="N(0, 0.2)", color='blue')
plt.hist(gaussian_samples[1], bins=500, density=True, alpha=0.5, label="N(0, 1.0)", color='red')
plt.hist(gaussian_samples[2], bins=500, density=True, alpha=0.5, label="N(0, 5.0)", color='yellow')
plt.hist(gaussian_samples[3], bins=500, density=True, alpha=0.5, label="N(-2, 0.5)", color='green')
plt.legend()
plt.title("Histogram of samples from different Gaussian distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("2c.png")
plt.show()
