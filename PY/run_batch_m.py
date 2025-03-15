import os
import re
import time
import subprocess
import sys


top_k = int(sys.argv[1])

# # 設定資料夾名稱
# output_folder = f"./t_result_{top_k}"

# # 如果資料夾不存在則建立
# os.makedirs(output_folder, exist_ok=True)


################### 0_fba_m.py ###################
# print("0_fba_m.py")X
# for i in range(0, 1212):  # 1 到 1000
for i in range(800, 1000):  # 1 到 1000
    print(f"Running index {i}...")
    subprocess.run(["python3", f"0_fba_m.py", str(i), str(top_k)], check=True)



