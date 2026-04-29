import subprocess

# 定義所有要執行的 Python 檔案路徑
scripts = [
    "experiment/totalframes_5_3-1/testscript_5_3-1.py",
    "experiment/totalframes_5_3-2/testscript_5_3-2.py",
    "experiment/totalframes_5_3-3/testscript_5_3-3.py",
    "experiment/totalframes_5_3-4/testscript_5_3-4.py",
    "experiment/totalframes_5_3-5/testscript_5_3-5.py",
    "experiment/totalframes_5_3-6/testscript_5_3-6.py",
    "experiment/totalframes_5_3-random/testscript_5_3-random.py"
]

# 逐一執行每個腳本
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script])
    
    # 輸出結果與錯誤訊息
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
