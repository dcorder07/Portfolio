import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/dcord/Downloads/consolidated-data.csv'
df = pd.read_csv(file_path)

# Create the contingency table: cross-tabulate Race and Gender
contingency_table = pd.crosstab(df['Race'], df['Gender'])

# Print the full contingency table
print("Full Contingency Table:")
print(contingency_table)

# Drop the last race and the 'Nonwhite' category
contingency_table_filtered = contingency_table.drop(index=[contingency_table.index[-1], 'Nonwhtie'], errors='ignore')

# Select the first two columns (genders) from the contingency table
contingency_table_first_two_genders = contingency_table_filtered.iloc[:, :2]

# Print the selected subset
print("\nContingency Table for the First Two Genders:")
print(contingency_table_first_two_genders)

# Create a heatmap for the selected subset
plt.figure(figsize=(10, 8))
sns.heatmap(contingency_table_first_two_genders, annot=True, fmt="d", cmap="YlGnBu", cbar=True)

# Customize the plot
plt.title('Most Common Race and Gender Combinations')
plt.xlabel('Gender')
plt.ylabel('Race')

# Adjust the layout to ensure everything fits well
plt.tight_layout()
plt.show()
