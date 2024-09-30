import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/dcord/Downloads/consolidated-data.csv'
df = pd.read_csv(file_path)

# List of prompts in order
prompts = [
    'Affectionate', 'Dedicated', 'Friendly', 'Organized', 'Corrupt', 'Dishonest',
    'Distracted', 'Unethical', 'Corporate Meeting', 'Dancing', 'Gardening', 'Golf',
    'Meditating', 'Shopping', 'Solo Traveling', 'Surfing', 'Video Games', 'Volunteering',
    'Activist', 'Construction Worker', 'Criminal', 'Doctor', 'Dry Cleaner', 'Maid',
    'Mechanic', 'Nurse', 'Waiter', 'Musician'
]

# Initialize the figure for pie charts with more space between them
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(30, 20))
axes = axes.flatten()

# Process each prompt
for i, prompt in enumerate(prompts):
    start_row = 1 + i * 100
    end_row = start_row + 99

    # Extract the data for the current prompt
    prompt_data = df.iloc[start_row:end_row]

    # Count occurrences of each race
    race_counts = prompt_data['Race'].value_counts()

    # Filter out races with a count less than 5
    race_counts = race_counts[race_counts >= 5]

    # Calculate percentages and print them
    total_count = race_counts.sum()
    if total_count > 0:
        percentages = (race_counts / total_count) * 100
        print(f"Percentages for {prompt}:")
        print(percentages.to_string(), "\n")

    # Plot the pie chart
    if not race_counts.empty:
        axes[i].pie(race_counts, labels=race_counts.index, autopct='%1.1f%%', 
                    colors=plt.cm.Paired(range(len(race_counts))), pctdistance=0.85, 
                    textprops={'fontsize': 5})  # Set fontsize to 6
        axes[i].set_title(prompt, fontsize=12)
    else:
        axes[i].axis('off')  # Turn off the axis if there's no data to display

# Turn off any remaining empty axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout to spread out the pie charts
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
plt.show()
