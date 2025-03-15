import os
import re
import time
import subprocess
import sys

top_k = int(sys.argv[1])

################### 0_fba_t.py ###################
print("0_fba_t.py")
for i in range(0, 1117):  # 1 到 1000
    print(f"Running index {i}...")
    subprocess.run(["python3", f"0_fba_t.py", str(i), str(top_k)], check=True)


