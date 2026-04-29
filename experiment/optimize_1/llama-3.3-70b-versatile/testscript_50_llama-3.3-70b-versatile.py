import subprocess
import time
import os

def clear_log_file(log_path):
    with open(log_path, "w") as f:
        f.truncate(0)

def get_last_frame_from_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    return int(lines[-1].strip())

def main():
    model_name = "llama-3.3-70b-versatile"
    script_path = "experiment/optimize_1/third_person_3-replan-anytime.py"
    log_path = f"experiment/optimize_1/{model_name}/frame_log-6.txt"

    # 確保資料夾存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    runs = 50
    total_frames = 0
    successful_runs = 0

    # clear_log_file(log_path)

    for i in range(runs):
        print(f"\n🚀 Running simulation {i+1}/{runs}...")
        # ✅ 傳 model_name 當參數給子腳本
        subprocess.run(["python", script_path, model_name])

        time.sleep(1)  # 確保檔案寫入完成

        frame_count = get_last_frame_from_log(log_path)
        if frame_count is not None:
            print(f"✅ Run {i+1} complete. Total frames used: {frame_count}")
            total_frames += frame_count
            successful_runs += 1
        else:
            print("❌ Frame count not found in log.")

    if successful_runs > 0:
        average = total_frames / successful_runs
        print("\n📊 ===========================")
        print(f"✅ Ran {successful_runs} successful simulations.")
        print(f"📉 Average Total Frames: {average:.2f}")
        print("📊 ===========================")
    else:
        print("❌ No successful runs to average.")

if __name__ == "__main__":
    main()
