import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("salaries-by-college-type.csv", usecols=['School Type', 'Starting Median Salary'])

data['Starting Median Salary'] = data['Starting Median Salary'].replace('[\$,]', '', regex=True).astype(float)

# Create a figure and axis
plt.figure(figsize=(10, 6))

school_types = data['School Type'].unique()


salary_data = [data[data['School Type'] == school_type]['Starting Median Salary'] for school_type in school_types]

plt.violinplot(salary_data)

# Set the x-ticks to be the school types
plt.xticks(ticks=range(1, len(school_types) + 1), labels=school_types)


plt.xlabel("College field")
plt.ylabel("Starting Median Salary")
plt.title("Violin Plot of Starting Median Salaries by School Type")


plt.tight_layout()
plt.show()
