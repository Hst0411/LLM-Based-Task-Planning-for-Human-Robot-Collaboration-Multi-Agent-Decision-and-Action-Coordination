import json, os
from typing import List, Tuple, Dict
from openai import OpenAI
import math
import requests  # google gemini
from groq import Groq   # Groq llama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "API_KEY")

# === 🧠 LLaMa 呼叫函式 ===
def call_groq_llama(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    使用 Groq API 呼叫 Llama 模型。
    Args:
        prompt (str): 輸入的文字提示
        model (str): 模型名稱 (預設 llama-3.3-70b-versatile)
    Returns:
        str: 模型回覆的文字
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a professional multi-robot planner."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Groq Llama 呼叫失敗: {e}")
        return "ERROR: Failed to call Groq API."

# === 🧠 Gemini 呼叫函式 ===
def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Calls the Google Gemini API and returns the text output.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        print("❌ Unexpected Gemini API response format:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise

# === 🧠 OpenAI 呼叫函式 ===
def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """
    Calls the OpenAI API and returns the text output.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a professional multi-robot planner."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def language_grounding(task_state):
    """
    Convert task_list_all.json content into a natural language description.
    """
    desc = []

    for robot_name, robot_info in task_state.get("robots", {}).items():
        pos = robot_info["position"]
        status = robot_info["status"]
        task_id = robot_info["current_task_id"]
        if task_id is None:
            desc.append(f"{robot_name} is currently idle at position ({pos['x']}, {pos['y']}, {pos['z']}).")
        else:
            desc.append(f"{robot_name} is working on task {task_id} at position ({pos['x']}, {pos['y']}, {pos['z']}).")

    for task in task_state.get("tasks", []):
        name = task["name"]
        task_id = task["id"]
        pos = task["position"]
        target = task["target"]
        state = task["state"]
        assigned_to = task["assigned_to"]

        if assigned_to is None:
            assignment_text = "No robot is assigned to this task."
        else:
            assignment_text = f"This task is assigned to {assigned_to}."

        if state == "unfinished":
            desc.append(
                f"Task {task_id} is to move a '{name}' from position "
                f"({pos['x']}, {pos['y']}, {pos['z']}) to target position "
                f"({target['x']}, {target['y']}, {target['z']}). "
                f"Current state: {state}. {assignment_text}"
            )

    return "\n".join(desc)

def _distance(pos1, pos2):
    return math.sqrt((pos1["x"] - pos2["x"])**2 +
                     (pos1["y"] - pos2["y"])**2 +
                     (pos1["z"] - pos2["z"])**2)

def get_human_finished_tasks(task_list_path="experiment/optimize_1/task_list_new.json", model_name="gpt-4"):
    """
    Called when human finished one or more tasks.
    Reloads the unified task list and asks GPT-4 to replan tasks for both robots.
    Returns:
        new_robot_tasks, new_robot2_tasks (Lists of (action, target) tuples)
    """

    with open(task_list_path, "r") as f:
        data = json.load(f)

    tasks = data["tasks"]
    robots = data["robots"]

    # Filter tasks
    unfinished = [t for t in tasks if t["state"] != "finished"]
    finished = [t for t in tasks if t["state"] == "finished"]

    # If no unfinished tasks
    if not unfinished:
        print("✅ No unfinished tasks.")
        return [], []

    # If only one task left → assign to nearest robot directly
    if len(unfinished) == 1:
        task = unfinished[0]
        pickup = task["position"]
        dropoff = task["target"]
        d1 = _distance(robots["robot1"]["position"], pickup)
        d2 = _distance(robots["robot2"]["position"], pickup)

        task_dict = {"id": task["id"], "pickup": pickup, "dropoff": dropoff}
        if d1 <= d2:
            robot1_tasks = _convert([task_dict])
            robot2_tasks = []
            print(f"⚡ Only one unfinished task (ID={task['id']}) → assigned to Robot1 (closer).")
        else:
            robot1_tasks = []
            robot2_tasks = _convert([task_dict])
            print(f"⚡ Only one unfinished task (ID={task['id']}) → assigned to Robot2 (closer).")

        return robot1_tasks, robot2_tasks

    robot1_status = robots["robot1"]["status"]
    robot2_status = robots["robot2"]["status"]

    # 產生自然語言描述
    description = language_grounding(data)
    
    # --- Construct GPT prompt ---
    prompt = f"""
You are a task planner for a multi-robot system.
The environment has 2 robots delivering food items.
You have access to the *full* task list and the current status of both robots.

This is the current world state file:
- {description}

This file has two parts, including robots and tasks

In "robots" part, there are two robots(robot1 and robot2), each robot includes:
- position: current coordinates
- current_task_id: which task ID it is handling (or null)
- status: "idle", "executing"

In "tasks" part, there are six foods tasks, each task includes:
- name: food name
- id: unique identifier
- position: pickup location
- target: delivery location(dropoff position)
- state: "unfinished" or "finished"
- assigned_to: which robot is responsible, or null if none

Your job:
- Decide how to reassign unfinished tasks (state="unfinished") to Robot1 and Robot2.
- Optimize overall efficiency by minimizing total travel distance.
  * Consider both the distance between robots and pickup points,
    and the delivery (pickup → dropoff) distance.
- Order tasks from the most efficient (closest / fastest) to least efficient.
- If a robot is currently idle, prioritize assigning it new tasks.
- Avoid assigning the same task to both robots.
- Do not modify finished tasks.
- Return ONLY valid JSON output.

Output JSON ONLY in this exact format:
{{
  "Robot1": [{{"id": ..., "pickup": ..., "dropoff": ...}}, ...],
  "Robot2": [{{"id": ..., "pickup": ..., "dropoff": ...}}, ...]
}}

All earlier tasks in each list should represent the ones that should be executed first.
No explanation, text, or comments outside JSON.
    """

    # === OpenAI GPT ===
    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4.1", "gpt-4-turbo"]:
        raw = call_openai(prompt, model=model_name)
    # === Google Gemini ===
    elif model_name == "gemini-2.5-flash":
        raw = call_gemini(prompt, model=model_name)
    elif model_name == "llama-4":
        raw = call_groq_llama(prompt, model = "meta-llama/llama-4-scout-17b-16e-instruct")

    # === 🧹 Clean Markdown Blocks ===
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ LLM output is not valid JSON. Skipping replan.")
        print(raw)
        return [], []

    # --- Convert and display ---
    robot1_tasks = _convert(plan["Robot1"])
    robot2_tasks = _convert(plan["Robot2"])

    print("🤖 Robot1 新任務規劃：")
    for i, t in enumerate(plan["Robot1"]):
        print(f"{i+1}. {t['id']} → from {t['pickup']} to {t['dropoff']}")
    print("======================================================================")
    print("🦾 Robot2 新任務規劃：")
    for i, t in enumerate(plan["Robot2"]):
        print(f"{i+1}. {t['id']} → from {t['pickup']} to {t['dropoff']}")
    print("======================================================================")

    # --- Update JSON task list with new assignment ---
    id_to_robot = {t["id"]: "robot1" for t in plan["Robot1"]}
    id_to_robot.update({t["id"]: "robot2" for t in plan["Robot2"]})

    for t in tasks:
        if t["state"] != "finished":
            t["assigned_to"] = id_to_robot.get(t["id"], None)

    with open(task_list_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Updated task list after replan.")

    return robot1_tasks, robot2_tasks


def _convert(robot_plan):
    def to_dict(pos):
        # 如果已經是 dict，就直接回傳
        if isinstance(pos, dict):
            return pos
        # 如果是 list，就轉成 dict
        elif isinstance(pos, list) and len(pos) == 3:
            return {"x": pos[0], "y": pos[1], "z": pos[2]}
        else:
            raise ValueError(f"Invalid position format: {pos}")

    seq = []
    for task in robot_plan:
        seq.append(("navigate_to", to_dict(task["pickup"])))
        seq.append(("pick_up", task["id"]))
        seq.append(("navigate_to", to_dict(task["dropoff"])))
        seq.append(("drop", None))
    return seq
