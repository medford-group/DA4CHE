import glob
import subprocess

logs = sorted(glob.glob("_build/html/reports/*/*.log"))
N_hash = 80

for log in logs:
    print('#'*N_hash)
    print(log)
    print('#'*N_hash)
    subprocess.run(['cat',log])
    print('#'*N_hash)
    input("Press Enter for the next log...")
