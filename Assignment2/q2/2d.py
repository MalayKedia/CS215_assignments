import numpy as np
import matplotlib.pyplot as plt

def simulate_galton_board(N, h):
    # Simulate N balls falling through h pegs, where each movement is either left (-1) or right (+1)
    steps =2*np.random.randint(0,2,(N,h)) -1  # N x h array of random -1 or 1
    final_positions=np.sum(steps, axis=1)   
    
    # Shift positions by h to make them non-negative
    final_positions_shifted=final_positions+h
    pockets=np.bincount(final_positions_shifted, minlength=2*h+1)
    
    return pockets

def plt_hist(pockets, h, fname, N):
    x_values=np.arange(-h, h+1)

    normalized_pockets =pockets /N
   
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, normalized_pockets, color='blue')
    plt.title(f"Galton Board of h = {h}")
    plt.xlabel("Pocket Position")
    plt.ylabel("Normalized Count")
    plt.savefig(fname)
    plt.close()

N=10**5
depths=[10, 50, 100]
fnames=["2d1.png", "2d2.png", "2d3.png"]

# Simulate for different depths and plot histograms
for h, fname in depths, fnames:
    pockets =simulate_galton_board(N, h)
    plt_hist(pockets, h, fname, N)
