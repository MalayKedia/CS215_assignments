import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading data from csv files and filtering data
csv_file = 'data.csv' 

df = pd.read_csv(csv_file, skiprows=12)

distances = df['D (Mpc)']
distances = distances[:1500]  # Keep only the first 1500 rows
distances=distances[distances<=4]   #keep only < 4Mpc

# Part (a): Plot a histogram with 10 bins
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=10, range=[0,4], alpha=0.6, color='blue', edgecolor='black')
plt.title('Histogram of Galaxy Distances (filtered)')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Number of Galaxies')
plt.savefig('images/10binhistogram.png')

# Calculating the estimated probabilities for each bin
hist, bin_edges = np.histogram(distances, bins=10, range=[0,4])
n = len(distances)
probabilities = hist / n

print("Printing probability distributions for each bin\n==============================================================================")
for i, prob in enumerate(probabilities):
    print(f'Bin {i + 1} [{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]: Probability = {prob:.4f}')
print("==============================================================================")

# Calculating cross validation scores
bin_counts = range(1, 1001)
cross_val_scores = []
bin_widths = []

for b in bin_counts:
    h = 4/ b      # calculating bin width as max-min/number_of_bins as max-min is 4 based on filtering of data
    bin_widths.append(h)
    hist, bin_edges = np.histogram(distances, bins=b, density=True, range=[0,4], )
    probabilities = hist * h
    sum_squared_probabilities = np.sum(probabilities**2)
    J_h = (2 / ((n - 1) * h)) - ((n + 1) / ((n - 1) * h)) * sum_squared_probabilities    #using formula for cross validation estimator
    cross_val_scores.append(J_h)

plt.figure(figsize=(10, 6))
plt.plot(bin_widths, cross_val_scores, color='blue')
plt.title('Cross-Validation Score vs. Bin Width')
plt.xlabel('Bin Width (h)')
plt.ylabel('Cross-Validation Score J(h)')
plt.grid()
plt.savefig('images/crossvalidation.png')

#Finding optimal number of bins and plotting the optimal histogram
optimal_index = np.argmin(cross_val_scores)     # minimising cross validation scores
optimal_bin_width = bin_widths[optimal_index]
optimal_bins= int(4/optimal_bin_width)

print("Optimal Bin width: ",optimal_bin_width)
print("Optimal Number of bins: ",optimal_bins)
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=optimal_bins, density=True,  range=[0,4], alpha=0.6, color='blue', edgecolor='black')
plt.title('Histogram of Distances with Optimal Bin Width')
plt.xlabel('Distance (D in Mpc)')
plt.ylabel('Probability Density')
plt.grid()
plt.savefig('images/optimalhistogram.png')