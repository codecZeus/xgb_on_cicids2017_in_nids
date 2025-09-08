import pandas as pd

# Define the file path for the dataset
file_path = "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Load the dataset, handling potential errors and setting the encoding
try:
    df = pd.read_csv(file_path, encoding='latin1')
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# The describe() function is used to generate descriptive statistics.
# We transpose the output (.T) to make it easier to read.
summary_stats = df.describe().T

# Print the resulting summary statistics
print("Summary Statistics of the Dataset:")
print(summary_stats)

# You can also save the output to a CSV or Excel file for easier use in your report
# summary_stats.to_csv('dataset_summary_statistics.csv')
