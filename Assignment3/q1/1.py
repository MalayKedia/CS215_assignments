#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Reading data from csv files and filtering data
csv_file = r'C:\Users\aksgu\Desktop\CS215_assignments\Assignment3\q1\data.csv' 
# csv_file = '/mnt/c/Users/aksgu/Desktop/CS215_assignments/Assignment3/q1/data.csv'

df = pd.read_csv(csv_file, skiprows=12)
distances = df['D (Mpc)'].dropna()  # Remove NaN values, if any
distances = distances[:1501]  # Keep only the first 1500 rows
distances=distances[distances<=4]   #keep only < 4Mpc
plt.hist(distances, bins=10, edgecolor='black')   # Step 3: Plot a histogram with 10 bins
plt.title('Histogram of Galaxy Distances (filtered)')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Number of Galaxies')
plt.savefig('images/10binhistogram.png')

#Calculate the estimated probabilities for each bin
hist, bin_edges = np.histogram(distances, bins=10)
n = len(distances)
probabilities = hist / n

print("Printing probability distributions for each bin=======================================")
for i, prob in enumerate(probabilities):
    print(f'Bin {i + 1} [{bin_edges[i]} - {bin_edges[i+1]}]: Probability = {prob:.4f}')
    

#Calculating cross validation scores
bin_counts = range(1, 1001)
cross_val_scores = []
bin_widths = []

for b in bin_counts:
    h = 4/ b      #calculating bin width as max-min/number_of_bins as max-min is 4 based on filtering of data
    bin_widths.append(h)  
    hist, bin_edges = np.histogram(distances, bins=b, density=True)
    probabilities = hist * np.diff(bin_edges) 
    sum_squared_probabilities = np.sum(probabilities**2)
    J_h = (2 / ((n - 1) * h)) - ((n + 1) / ((n - 1) * h)) * sum_squared_probabilities    #using above formula
    cross_val_scores.append(J_h)

plt.figure(figsize=(10, 6))
plt.plot(bin_widths, cross_val_scores, color='blue')
plt.title('Cross-Validation Score vs. Bin Width')
plt.xlabel('Bin Width (h)')
plt.ylabel('Cross-Validation Score J(h)')
plt.grid()
plt.savefig('images/crossvalidation.png')


#Finding optimal number of bins and plotting the optimal histogram
optimal_index = np.argmin(cross_val_scores)     #minimising cross validation scores
optimal_bin_width = bin_widths[optimal_index]
optimal_bins= int(4/optimal_bin_width)
print("Optimal Bin width: ",optimal_bin_width)
print("Optimal Number of bins: ",optimal_bins)
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=optimal_bins, density=True, alpha=0.6, color='blue', edgecolor='black')
plt.title('Histogram of Distances with Optimal Bin Width')
plt.xlabel('Distance (D in Mpc)')
plt.ylabel('Density')
plt.grid()
plt.savefig('images/optimalhistogram.png')