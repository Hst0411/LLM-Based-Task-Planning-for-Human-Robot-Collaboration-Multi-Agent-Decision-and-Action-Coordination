import subprocess

# 定義所有要執行的 Python 檔案路徑
scripts = [
    "experiment/totalframes_5/testscript_5.py",
    "experiment/totalframes_5_2-1/testscript_5_2-1.py",
    "experiment/totalframes_5_2-2/testscript_5_2-2.py",
    "experiment/totalframes_5_2-3/testscript_5_2-3.py",
    "experiment/totalframes_5_2-4/testscript_5_2-4.py",
    "experiment/totalframes_5_2-5/testscript_5_2-5.py",
    "experiment/totalframes_5_2-6/testscript_5_2-6.py",
    "experiment/totalframes_5_2-random/testscript_5_2-random.py"
]

# 逐一執行每個腳本
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script])
    
    # 輸出結果與錯誤訊息
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
