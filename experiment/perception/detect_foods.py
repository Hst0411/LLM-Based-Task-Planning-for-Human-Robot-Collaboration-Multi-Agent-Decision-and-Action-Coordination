from ultralytics import YOLO
import os

# 載入 YOLOv8 模型 (COCO 預訓練)
# segmentation = "yolov8s"
# segmentation = "yolov8n-seg"
segmentation = "yolov8s-seg"
# segmentation = "yolov8m-seg"
# segmentation = "yolov8l-seg"
# segmentation = "yolov8x-seg"

model = YOLO(segmentation + ".pt")

# 你的圖片資料夾根目錄
image_folder = "experiment/detection_data"

# 設定儲存結果的資料夾
output_folder = "experiment/detection_data"
os.makedirs(output_folder, exist_ok=True)

# 最終的輸出檔案
output_file = os.path.join(output_folder, segmentation + ".txt")

# 只保留這些類別
target_classes = {
    "person": "a agent",
    "banana": "banana",
    "sandwich": "burger",       # 用 COCO sandwich 當 burger
    "apple": "apple",
    "cake": "bread",            # 先暫時用 cake 代替 bread
    "orange": "orange",
    "cell phone": "phone"
}

# 開啟輸出檔案進行寫入
with open(output_file, "w", encoding="utf-8") as f:
    # 遞歸處理所有資料夾內的圖片
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)

        # 確保是資料夾
        if os.path.isdir(folder_path):
            f.write(f"=== Results for folder: {folder_name} ===\n")
            
            # 只處理這些資料夾中的圖片
            image_files = [f_ for f_ in os.listdir(folder_path) if f_.lower().endswith((".png", ".jpg", ".jpeg"))]
            image_files.sort()

            # 處理每一張圖片
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                results = model(img_path)

                detected_labels = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        conf = float(box.conf[0])
                        if name in target_classes:
                            mapped = target_classes[name]
                            detected_labels.append(f"{mapped} ({conf:.2f})")

                # 寫入結果到 txt 檔案
                if detected_labels:
                    line = f"{img_file}: {', '.join(detected_labels)}\n"
                else:
                    line = f"{img_file}: (no target detected)\n"

                f.write(line)

            f.write("\n")  # 每個資料夾結束後換行

print(f"✅ 偵測完成，所有結果已存到 {output_file}")
