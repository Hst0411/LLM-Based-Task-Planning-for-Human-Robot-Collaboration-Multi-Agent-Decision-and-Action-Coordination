import os, base64, json, cv2
from ultralytics import YOLO
from openai import OpenAI

# -----------------
# 可調參數
# -----------------
API_KEY = "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"

IMG_DIR        = r"experiment/robot_first_person/human_robot_2/a"
REF_ROBOT_DIR  = r"experiment/detection_data/replicant"   # robot 參考圖
REF_FOOD_DIR   = r"experiment/ref_food"    # 食物參考圖
YOLO_MODEL     = "yolov8s-seg.pt"
CONF_GATE      = 0.60
FRAME_GROUP    = 40
CROP_MARGIN    = 0.20
GPT_MODEL      = "gpt-4o"

OUT_FILE = os.path.join(IMG_DIR, "gpt_check_class.txt")

# -----------------
# 初始化
# -----------------
client   = OpenAI(api_key=API_KEY)
detector = YOLO(YOLO_MODEL)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def list_images_sorted(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def crop_person(image_path, margin=CROP_MARGIN):
    """裁切 YOLO 偵測到的 person"""
    try:
        res = detector(image_path, verbose=False)
        best = None
        best_conf = -1.0
        for r in res:
            if not hasattr(r, "boxes"):
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = detector.names.get(cls_id, str(cls_id))
                if name == "person":
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                        best = (x1,y1,x2,y2)
        img = cv2.imread(image_path)
        if img is None or best is None:
            return image_path
        H, W = img.shape[:2]
        x1,y1,x2,y2 = best
        w, h = x2-x1, y2-y1
        x1 = max(0, int(x1 - margin*w))
        y1 = max(0, int(y1 - margin*h))
        x2 = min(W, int(x2 + margin*w))
        y2 = min(H, int(y2 + margin*h))
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(IMG_DIR, f"__crop__{os.path.basename(image_path)}")
        cv2.imwrite(out_path, crop)
        return out_path
    except Exception:
        return image_path

SYSTEM_PROMPT = """You are a strict visual verifier.

Tasks:
1) Compare candidate with reference robot images.
   - If same, it's a replicant (robot).
   - If not, treat it as a human.
2) If human, decide whether they are holding ANY of the foods in the reference food images.

Return compact JSON only.
"""

USER_INSTRUCTION = """Compare candidate frames against reference robot and reference food images.
Return JSON ONLY in this schema:
{"replicant": true|false, "holding_food": true|false, "food_name": "filename or empty", "score": float (0-1), "notes": "1 short reason"}"""

def gpt_verify(ref_robot, ref_foods, candidate_imgs, food_names):
    """送候選圖片給 GPT 驗證 robot/human + food"""
    content = [{"type":"text", "text": USER_INSTRUCTION}]
    # reference robot
    for ref in ref_robot:
        content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encode_image(ref)}"}})
    # reference food
    for food in ref_foods:
        content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encode_image(food)}"}})
    # candidates
    for c in candidate_imgs:
        content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encode_image(c)}"}})
    
    # 提供 food 名稱讓 GPT 知道要怎麼回傳
    food_list_text = "Available food names (filenames): " + ", ".join(food_names)
    content.insert(0, {"type": "text", "text": food_list_text})

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": content}
        ]
    )
    text = resp.choices[0].message.content.strip()
    try:
        s, e = text.find("{"), text.rfind("}")
        obj = json.loads(text[s:e+1])
        obj.setdefault("replicant", False)
        obj.setdefault("holding_food", False)
        obj.setdefault("food_name", "")
        obj.setdefault("score", 0.0)
        return obj
    except Exception:
        return {"replicant": False, "holding_food": False, "food_name": "", "score": 0.0}

def detect_and_check(start_index=0):
    """增量檢查：從 start_index 開始處理新圖片"""
    ref_robot = list_images_sorted(REF_ROBOT_DIR)
    ref_foods = list_images_sorted(REF_FOOD_DIR)
    
    # food_names = ["apple", "bread", "banana", "burger", "orange", "loafbread"]
    food_names = [os.path.splitext(os.path.basename(f))[0] for f in ref_foods]

    if not ref_robot or not ref_foods:
        print("⚠️ 缺少參考 robot 或 food 圖片")
        return start_index, []

    all_imgs = list_images_sorted(IMG_DIR)
    if start_index >= len(all_imgs):
        return start_index, []

    events = []
    i = start_index
    while i < len(all_imgs):
        img_path = all_imgs[i]
        results = detector(img_path, verbose=False)

        found_person = any(
            detector.names[int(box.cls[0])] == "person" and float(box.conf[0]) >= CONF_GATE
            for r in results for box in r.boxes
        )

        if found_person:
            # 收集後續 FRAME_GROUP 張 (必須有 person)
            group = []
            j = i
            while j < len(all_imgs) and len(group) < FRAME_GROUP:
                img_j = all_imgs[j]
                res_j = detector(img_j, verbose=False)
                has_person = any(
                    detector.names[int(box.cls[0])] == "person" and float(box.conf[0]) >= CONF_GATE
                    for rj in res_j for box in rj.boxes
                )
                if has_person:
                    group.append(img_j)
                j += 1

            if group:
                cand_crops = [crop_person(p) for p in group]
                obj = gpt_verify(ref_robot, ref_foods, cand_crops, food_names)

                if obj.get("replicant"):
                    print(f"🤖 {os.path.basename(img_path)} -> Robot detected, skip")
                else:
                    if obj.get("holding_food"):
                        line = f"{os.path.basename(img_path)} detected a human carrying {obj.get('food_name')}"
                        print("✅", line)
                        events.append(line)
                    else:
                        line = f"{os.path.basename(img_path)} only detected a human, but not carrying anything"
                        print("ℹ️", line)
                        events.append(line)

            i = j
        else:
            i += 1

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for e in events:
            fout.write(e + "\n")

    return len(all_imgs), events

if __name__ == "__main__":
    detect_and_check(0)
