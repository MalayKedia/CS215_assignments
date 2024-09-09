import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def sample(loc, scale, size):
    uniform_sample = np.random.uniform(0, 1, size)
    gaussian_sample = norm.ppf(uniform_sample, loc=loc, scale=math.sqrt(scale))
    return gaussian_sample

N = 10**5
params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]

gaussian_samples=[]
for param in params:
    gaussian_samples.append(sample(param[0], param[1], N))

plt.figure(figsize=(10, 8))
plt.xlim(-6,6)
plt.hist(gaussian_samples[0], bins='auto', density=True, alpha=0.5, label=r'$N(\mu=0, \sigma^2=0.2)$', color='blue')
plt.hist(gaussian_samples[1], bins='auto', density=True, alpha=0.5, label=r'$N(\mu=0, \sigma^2=1.0)$', color='red')
plt.hist(gaussian_samples[2], bins='auto', density=True, alpha=0.5, label=r'$N(\mu=0, \sigma^2=5.0)$', color='yellow')
plt.hist(gaussian_samples[3], bins='auto', density=True, alpha=0.5, label=r'$N(\mu=-2, \sigma^2=0.5)$', color='green')
plt.legend()
plt.title("Histogram of samples from different Gaussian distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("2c.png")
plt.show()
