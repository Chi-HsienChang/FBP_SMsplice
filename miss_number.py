import os
import re
import sys


my_seed = int(sys.argv[1])
dataset = sys.argv[2]   # "t", "z", "m", or "h" or "f" or "o"
top_k = int(sys.argv[3])

# Specify the directory containing the files
directory = f"{my_seed}_t_z_m_h_f_o_result/"

if dataset == "t":
    directory += f"{my_seed}_t_result/t_result_{top_k}"
elif dataset == "z":
    directory += f"{my_seed}_z_result/z_result_{top_k}"
elif dataset == "m":
    directory += f"{my_seed}_m_result/m_result_{top_k}"
elif dataset == "h":
    directory += f"{my_seed}_h_result/h_result_{top_k}"
elif dataset == "f":
    directory += f"{my_seed}_f_result/f_result_{top_k}"
elif dataset == "o":
    directory += f"{my_seed}_o_result/o_result_{top_k}"



# Define the expected range
if dataset == "t":
    expected_number = 1117
elif dataset == "z":
    expected_number = 825
elif dataset == "m":
    expected_number = 1212
elif dataset == "h":
    expected_number = 1629
elif dataset == "f":
    expected_number = 1938
elif dataset == "o":
    expected_number = 921


if dataset == "t":
    pattern = re.compile(r"000_arabidopsis_g_(\d+)\.txt")
elif dataset == "z":
    pattern = re.compile(r"000_zebrafish_g_(\d+)\.txt")
elif dataset == "m":
    pattern = re.compile(r"000_mouse_g_(\d+)\.txt")
elif dataset == "h":
    pattern = re.compile(r"000_human_g_(\d+)\.txt")
elif dataset == "f":
    pattern = re.compile(r"000_fly_g_(\d+)\.txt")
elif dataset == "o":
    pattern = re.compile(r"000_moth_g_(\d+)\.txt")




# Get all filenames in the directory
files = os.listdir(directory)

# Extract numbers from filenames using regex
# pattern = re.compile(r"000_arabidopsis_g_(\d+)\.txt")
found_numbers = sorted(int(pattern.search(f).group(1)) for f in files if pattern.search(f))

# Define the expected range
expected_numbers = set(range(0, expected_number))  # Adjust based on your full expected range

# Find missing numbers
missing_numbers = sorted(expected_numbers - set(found_numbers))

# Print missing numbers
print("Missing numbers:", missing_numbers)
print("not_yet = ", len(missing_numbers))


if dataset == "t":
    print("done = ", 1117-len(missing_numbers))
elif dataset == "z":
    print("done = ", 825-len(missing_numbers))
elif dataset == "m":
    print("done = ", 1212-len(missing_numbers))
elif dataset == "h":
    print("done = ", 1629-len(missing_numbers))
elif dataset == "f":
    print("done = ", 1938-len(missing_numbers))
elif dataset == "o":
    print("done = ", 921-len(missing_numbers))



