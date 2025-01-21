import pandas as pd
import numpy as np


# Set random seed for reproducibility
np.random.seed(42)

# Define the size of the dataset
num_entries = 3000  # Change this to 2000 or any desired number

# Generate synthetic data
data = {
    "Machine_ID": np.arange(1, num_entries + 1),
    "Temperature": np.random.uniform(50, 100, num_entries),
    "Run_Time": np.random.uniform(200, 1000, num_entries),
    "Downtime_Flag": np.random.choice([0, 1], size=num_entries, p=[0.7, 0.3]),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
file_name = "synthetic_manufacturing_data_large.csv"
df.to_csv(file_name, index=False)
print(f"Synthetic dataset saved as {file_name}!")
