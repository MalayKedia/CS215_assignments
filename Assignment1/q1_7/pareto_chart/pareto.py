import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv("AttendanceMarks.csv", usecols=['ESE'])

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Step 1: Create bins and calculate frequencies
bins = range(30, 81, 3)  # Create bins for scores 0-10, 10-20, ..., 90-100
data['Bins'] = pd.cut(data['ESE'], bins, right=False)
frequency_counts = data['Bins'].value_counts().sort_index()

# Step 2: Calculate cumulative percentage
cumulative_percentage = frequency_counts.cumsum() / frequency_counts.sum() * 100

# Step 3: Create the Pareto chart
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for frequencies
ax1.bar(frequency_counts.index.astype(str), frequency_counts, color='skyblue', alpha=0.6)
ax1.set_xlabel('Scores')
ax1.set_ylabel('Number of Students', color='b')
ax1.tick_params(axis='y', colors='b')

# Step 4: Plot cumulative percentage line
ax2 = ax1.twinx()
ax2.plot(frequency_counts.index.astype(str), cumulative_percentage, color='red', marker='D', linestyle='-', linewidth=2)
ax2.set_ylabel('Cumulative Percentage', color='r')
ax2.tick_params(axis='y', colors='r')

# Step 5: Title and layout
plt.title('Pareto Chart of Maths Scores')
plt.tight_layout()

# Show the plot
plt.show()
