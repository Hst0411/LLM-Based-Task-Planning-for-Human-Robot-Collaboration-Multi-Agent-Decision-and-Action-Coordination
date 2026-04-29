from ultralytics import YOLO
import os

# 載入 YOLOv8 模型 (COCO 預訓練)
# segmentation = "yolov8s"
# segmentation = "yolov8n-seg"
segmentation = "yolov8s-seg"
# segmentation = "yolov8m-seg"
# segmentation = "yolov8l-seg"
# segmentation = "yolov8x-seg"
# segmentation = "experiment/perception/best.pt"

# model = YOLO(segmentation + ".pt")
model = YOLO(segmentation)

# 你的圖片資料夾
image_folder = "experiment/robot_first_person/human_robot_2/a"

# 設定儲存結果的資料夾
output_folder = "experiment/robot_first_person/human_robot_2/a"
os.makedirs(output_folder, exist_ok=True)

# 最終的輸出檔案
output_file = os.path.join(output_folder, "detections(" + segmentation + ").txt")

# 最終的輸出檔案
# output_file = os.path.join(output_folder, "detections(best).txt")

# 只保留這些類別
target_classes = {
    "person": "a agent",
    "banana": "banana",
    "sandwich": "burger",       # 用 COCO sandwich 當 burger
    "apple": "apple",
    "cake": "bread",            # 先暫時用 cake 代替 bread
    "donut": "loafbread",       # 先暫時用 donut 代替 loafbread
    "orange": "orange",
    "cell phone": "iphone"
}

# target_classes = {
#     "human": "human",
#     "banana": "banana",
#     "burger": "burger",
#     "apple": "apple",
#     "bread": "bread",
#     "loafbread": "loafbread",
#     "orange": "orange",
#     "phone": "phone",
#     "robot": "robot"
# }

with open(output_file, "w", encoding="utf-8") as f:
    # 逐張圖片處理
    image_files = [f_ for f_ in os.listdir(image_folder) if f_.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
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

        # 寫入 txt
        if detected_labels:
            line = f"{img_file}: {', '.join(detected_labels)}\n"
        else:
            line = f"{img_file}: (no target detected)\n"

        f.write(line)

print(f"✅ 偵測完成，結果已存到 {output_file}")
