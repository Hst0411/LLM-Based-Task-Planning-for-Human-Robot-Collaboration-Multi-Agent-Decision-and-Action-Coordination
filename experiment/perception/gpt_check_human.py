import os, re, json, base64, cv2
from ultralytics import YOLO
from openai import OpenAI

# ------------ 可調參數 ------------

API_KEY = "API_KEY"
DETECTION_FILE = r"experiment/robot_first_person/human_robot_2/a/detections(yolov8s-seg).txt"
IMG_DIR        = r"experiment/robot_first_person/human_robot_2/a"
REF_DIR        = r"experiment/detection_data/replicant"   # 放 4 張清楚的 replicant
OUT_FILE       = os.path.join(IMG_DIR, "gpt_check_agents.txt")
YOLO_MODEL     = "yolov8s-seg.pt"                      # seg 或 det 皆可
CONF_GATE      = 0.80                                  # 只有偵測 > 0.8 才送 GPT
MAX_CAND_FRAMES= 5                                     # 取觸發張之後的最多 N 張
CROP_MARGIN    = 0.15                                  # 裁切時 bbox 旁邊加的邊界比例
GPT_MODEL      = "gpt-4o"                              # 視覺模型；可換 "gpt-4o"
GPT_SCORE_THRESH = 0.65                                # JSON score 達到才視為同類型
# ---------------------------------

client = OpenAI(api_key=API_KEY)
detector = YOLO(YOLO_MODEL)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def list_images_sorted(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    return files

def crop_person(image_path):
    """
    用 YOLO 找 'person'，取最高分的 bbox，裁切出來並加邊界；失敗就回傳原圖。
    """
    try:
        res = detector(image_path, verbose=False)
        h_person = None
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
                        h_person = (x1,y1,x2,y2)
        img = cv2.imread(image_path)
        if img is None:
            return image_path  # 讀不到圖，回原路徑
        if h_person is None:
            return image_path  # 找不到人，回原圖

        H,W = img.shape[:2]
        x1,y1,x2,y2 = h_person
        # 加邊界
        w = x2-x1; h = y2-y1
        x1 = max(0, int(x1 - CROP_MARGIN*w))
        y1 = max(0, int(y1 - CROP_MARGIN*h))
        x2 = min(W, int(x2 + CROP_MARGIN*w))
        y2 = min(H, int(y2 + CROP_MARGIN*h))
        crop = img[y1:y2, x1:x2]

        # 輸出到暫存檔（與原圖同資料夾）
        out_path = os.path.join(IMG_DIR, f"__crop__{os.path.basename(image_path)}")
        cv2.imwrite(out_path, crop)
        return out_path
    except Exception:
        return image_path

SYSTEM_PROMPT = """You are a strict visual verifier. Decide if the candidate person is the same target type as the reference agent.
Target definition: adult male wearing a BLACK JACKET (upper-body garment predominantly black).
Ignore pose, rotation, partial occlusion, distance, and lighting differences. Ignore background and objects being carried.
Deep navy/grey is NOT black unless it visually appears black under lighting. Focus on the upper garment color and type.
Return a compact JSON only.
"""

USER_INSTRUCTION = """Compare the candidate frames against ALL reference images of the target (black-jacket agent).
Return JSON with this schema (and nothing else):
{"same_agent": true|false, "score": float (0-1), "notes": "1 short reason"}"""

def gpt_verify(reference_imgs, candidate_imgs):
    """
    將多張參考 + 多張候選（已裁切）送進 GPT，要求回傳 JSON。
    """
    content = [
        {"type":"text", "text": USER_INSTRUCTION}
    ]
    # 先放 4 張 reference
    for ref in reference_imgs:
        content.append({"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{encode_image(ref)}"}})
    # 再放多張 candidate（已裁切）
    for c in candidate_imgs:
        content.append({"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{encode_image(c)}"}})

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user",   "content": content}
        ]
    )
    text = resp.choices[0].message.content.strip()
    # 盡力解析 JSON
    try:
        # 嘗試抓出第一個 JSON 區段
        start = text.find("{")
        end   = text.rfind("}")
        obj = json.loads(text[start:end+1])
        same = bool(obj.get("same_agent", False))
        score = float(obj.get("score", 0.0))
        notes = str(obj.get("notes",""))
        return same, score, notes, text
    except Exception:
        # 解析失敗，退回保守判定
        return False, 0.0, "parse_error", text

def parse_detection_conf(line):
    """
    解析像 'img_1567.jpg: a agent (0.44)' 這種行，回傳 (filename, conf or None)
    """
    m = re.match(r"\s*([^:]+):\s*(.*)", line)
    if not m:
        return None, None
    fname = m.group(1).strip()
    tail = m.group(2)
    if "a agent" not in tail:
        return fname, None
    m2 = re.search(r"a agent\s*\(([\d\.]+)\)", tail)
    if not m2:
        return fname, None
    try:
        conf = float(m2.group(1))
    except:
        conf = None
    return fname, conf

def main():
    # 讀參考圖
    ref_imgs = [os.path.join(REF_DIR,f) for f in list_images_sorted(REF_DIR)]
    if len(ref_imgs) == 0:
        print("⚠️ 找不到 reference 圖片，請放 4 張在", REF_DIR)
        return

    # 讀全部影格清單（用來找後續相鄰幾張）
    all_imgs = list_images_sorted(IMG_DIR)
    idx_map = {name:i for i,name in enumerate(all_imgs)}

    with open(DETECTION_FILE, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for ln in lines:
            fname, conf = parse_detection_conf(ln)
            if fname is None:
                continue
            if conf is None or conf < CONF_GATE:
                # 不是 agent 或信心分數太低，略過（也可寫入候選紀錄，看你需求）
                continue

            if fname not in idx_map:
                # 找不到原圖就略過
                continue

            # 蒐集候選影格：當前 + 之後最多 N-1 張
            start_idx = idx_map[fname]
            cand_names = all_imgs[start_idx : min(len(all_imgs), start_idx + MAX_CAND_FRAMES)]
            # 先裁切人再送 GPT
            cand_crops = [crop_person(os.path.join(IMG_DIR, n)) for n in cand_names]

            same, score, notes, raw = gpt_verify(ref_imgs, cand_crops)
            final_yes = (same and score >= GPT_SCORE_THRESH)

            fout.write(
                f"{fname}: agent_conf={conf:.2f}  gpt_same={same}  gpt_score={score:.2f}  decision={'YES' if final_yes else 'NO'}  notes={notes}\n"
            )
            print(f"🧪 {fname} | yolo={conf:.2f} | gpt={score:.2f} -> {'✅ YES' if final_yes else '❌ NO'}")

    print(f"\n✅ 完成，結果已寫入 {OUT_FILE}")

if __name__ == "__main__":
    main()
