import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/dcord/Downloads/consolidated-data.csv'
df = pd.read_csv(file_path)

# List of occupations corresponding to each block of 100 rows
occupations = [
    'Affectionate', 'Dedicated', 'Friendly', 'Organized', 'Corrupt', 
    'Dishonest', 'Distracted', 'Unethical', 'Corporate Meeting', 'Dancing', 
    'Gardening', 'Golf', 'Meditating', 'Shopping', 'Solo Traveling', 'Surfing', 
    'Video Games', 'Volunteering', 'Activist', 'Construction Worker', 
    'Criminal', 'Doctor', 'Dry Cleaner', 'Maid', 'Mechanic', 
    'Nurse', 'Waiter', 'Musician'
]

# Add a new column 'Occupation' to the DataFrame based on the block of rows
df['Occupation'] = pd.cut(df.index, bins=[i*100 for i in range(len(occupations) + 1)], labels=occupations, right=False)

# Create the contingency table: cross-tabulate Race and Occupation
contingency_table = pd.crosstab(df['Race'], df['Occupation'])

# Filter out any races with total counts less than 10
race_totals = contingency_table.sum(axis=1)
contingency_table_filtered = contingency_table[race_totals >= 10]

# Print the filtered contingency table
print(contingency_table_filtered)

# Create a heatmap for the filtered contingency table
plt.figure(figsize=(14, 10))
sns.heatmap(contingency_table_filtered, annot=True, fmt="d", cmap="YlGnBu", cbar=True)

# Customize the plot
plt.title('Race vs. Prompt Contingency Table')
plt.xlabel('Prompt')
plt.ylabel('Race')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.75)

# Show the heatmap
plt.tight_layout()
plt.show()

