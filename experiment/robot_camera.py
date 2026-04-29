import os, base64, glob
from ultralytics import YOLO
from openai import OpenAI

# ---------------------
# 初始化
# ---------------------
API_KEY = "API_KEY"
client = OpenAI(api_key=API_KEY)
detector = YOLO("yolov8s-seg.pt")

FOODS = {"apple", "banana", "bread", "burger", "loafbread", "orange"}
REF_DIR = r"experiment/detection_data/human"  # 黑衣 agent 的 reference 影像

# 載入 reference
reference_imgs = [os.path.join(REF_DIR, f) for f in os.listdir(REF_DIR) if f.lower().endswith((".jpg",".png"))]

# ---------------------
# 工具函數
# ---------------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def verify_black_agent(candidate_img, reference_imgs):
    """
    GPT 二次確認該候選影像的 agent 是否為黑衣 agent
    """
    SYSTEM_PROMPT = """You are a strict visual verifier.
Decide if the candidate person is the same target type as the reference agent.
Target definition: adult male wearing a BLACK JACKET (upper-body garment predominantly black).
Ignore pose, rotation, occlusion, lighting differences. Focus only on the upper garment.
Return only 'yes' or 'no'."""

    content = [{"type": "text", "text": "Compare this candidate with the reference agent."}]
    for ref in reference_imgs:
        content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encode_image(ref)}"}})
    content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{encode_image(candidate_img)}"}})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": content}
        ]
    )
    ans = resp.choices[0].message.content.strip().lower()
    return "yes" in ans

def detect_agent_with_food(image_path):
    """
    用 YOLO 偵測，回傳是否同張圖有 agent + 食物，以及類別列表
    """
    results = detector(image_path)
    classes = []
    for r in results:
        for c in r.boxes.cls:
            cls_name = detector.names[int(c)]
            classes.append(cls_name)

    found_agent = any("agent" in cls for cls in classes)
    found_food = any(cls in FOODS for cls in classes)
    return found_agent and found_food, classes

# ---------------------
# 在 robot camera loop 中呼叫
# ---------------------
def process_latest_image(image_folder):
    latest_imgs = sorted(glob.glob(os.path.join(image_folder, "*.jpg")), key=os.path.getmtime)
    if not latest_imgs:
        return
    latest_img = latest_imgs[-1]

    has_both, classes = detect_agent_with_food(latest_img)
    if has_both:
        # 二次確認是否為黑衣 agent
        if verify_black_agent(latest_img, reference_imgs):
            print(f"📸 Black agent WITH food detected in {os.path.basename(latest_img)}: {classes}")
        else:
            print(f"❌ Detected agent is not the black agent in {os.path.basename(latest_img)}")
