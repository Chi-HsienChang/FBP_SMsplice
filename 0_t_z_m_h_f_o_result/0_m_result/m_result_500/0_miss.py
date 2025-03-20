import os
import re

# Specify the directory containing the files
directory = "./"

# Get all filenames in the directory
files = os.listdir(directory)

# Extract numbers from filenames using regex
pattern = re.compile(r"000_mouse_g_(\d+)\.txt")
found_numbers = sorted(int(pattern.search(f).group(1)) for f in files if pattern.search(f))

# Define the expected range
expected_numbers = set(range(0, 1212))  # Adjust based on your full expected range

# Find missing numbers
missing_numbers = sorted(expected_numbers - set(found_numbers))

# Print missing numbers
print("Missing numbers:", missing_numbers)
print("not_yet = ", len(missing_numbers))
print("done = ", 1212-len(missing_numbers))


