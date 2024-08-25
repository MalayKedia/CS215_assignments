import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("World Marriage Dataset.csv", usecols=['Country', 'MaritalStatus'])


desired_statuses = ['Divorced', 'Married', 'Single']
df = df[df['MaritalStatus'].isin(desired_statuses)]

# Count the number of each MaritalStatus per Country
df_counts = df.groupby(['Country', 'MaritalStatus']).size().reset_index(name='Count')

df['sub_total'] = df.groupby('Country')['MaritalStatus'].transform('sum')

# Prepare data for plotting
countries = df_counts['Country'].unique()  # Get unique countries
statuses = df_counts['MaritalStatus'].unique()  # Get unique marital statuses
num_countries = len(countries)

# Set up the figure and axis for polar plot
fig, ax = plt.subplots(figsize=(8, 8), facecolor="#FFFFFF", subplot_kw=dict(polar=True))
fig.tight_layout(pad=3.0)

# Initialize bottom for stacking
bottom = np.zeros(num_countries)

# Plot bars for each marital status
for status in statuses:
    status_data = df_counts[df_counts['MaritalStatus'] == status]['Count'].reindex(countries).fillna(0).values
    width = 2 * np.pi / num_countries
    x_coords = np.linspace(0, 2 * np.pi, num_countries, endpoint=False)
    bars = ax.bar(x_coords, status_data, width=width, bottom=bottom, label=status)
    bottom += status_data  # Stack the bars

    # Add text labels
    for bar, count in zip(bars, status_data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            int(count),
            ha='center',
            va='center',
            size=8,
            color='w',
            weight='light'
        )

# Customize plot appearance
ax.set_axis_off()
ax.set_theta_zero_location("N")
plt.legend(title="Marital Status", bbox_to_anchor=(1.1, 1.05))
plt.show()


