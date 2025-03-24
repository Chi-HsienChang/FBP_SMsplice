import os
import shutil

# Define directories
source_directory = "./t_result"
destination_directory = "./t_result/empty_files"

# Create destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# List all .txt files
txt_files = [f for f in os.listdir(source_directory) if f.endswith(".txt")]

# Find empty files and move them
empty_files = [f for f in txt_files if os.path.getsize(os.path.join(source_directory, f)) == 0]

if empty_files:
    print("Empty files found and moving to ../empty_files/:")
    for file in empty_files:
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(destination_directory, file)
        
        # Move file
        shutil.move(source_path, destination_path)
        print(f"✅ Moved: {file}")
else:
    print("No empty files found.")

print("📁 Operation complete.")






# import os

# # Define directory
# directory = "./t_result"

# # List all .txt files
# txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

# # Find empty files
# empty_files = [f for f in txt_files if os.path.getsize(os.path.join(directory, f)) == 0]

# # Print empty files
# if empty_files:
#     print("Empty files found:")
#     for file in empty_files:
#         print(file)
# else:
#     print("No empty files found.")
