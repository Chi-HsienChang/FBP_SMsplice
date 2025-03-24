import subprocess

# Remote server details
remote_user = "labadmin"
remote_host = "140.112.42.124"
remote_path = "/home/labadmin/chi-hsien/weak_analysis_system/t_result/"

# Local destination
local_path = "."

# Define the range of files  to transfer (317-400)
for i in [350, 351, 352, 353, 355, 356, 357, 358, 360, 361, 363, 366, 367, 368, 369, 370, 371]:
    filename = f"000_arabidopsis_g_{i}.txt"
    remote_file = f"{remote_user}@{remote_host}:{remote_path}{filename}"
    
    print(f"Transferring {filename}...")
    
    try:
        subprocess.run(["scp", "-P", "22", remote_file, local_path], check=True)
        print(f"✅ Successfully transferred {filename}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to transfer {filename}")

print("📁 Transfer complete.")



