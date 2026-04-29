import json, os
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

# === 5️⃣ 轉換 LLM 的任務格式 → 行動序列 (navigate_to, pick_up, drop, etc.) ===
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

def report_progress_to_LLM(task_list_path="experiment/task_list_new.json", model_name="gpt-4"):
    """
    Let LLM decide whether to replan based on the entire task list.
    If replan == True, return the new assignments for both robots as action lists.
    """

    # === 1️⃣ Load current task list ===
    with open(task_list_path, "r") as f:
        data = json.load(f)

    robots = data["robots"]
    tasks = data["tasks"]

    # Filter tasks
    unfinished = [t for t in tasks if t["state"] != "finished"]
    finished = [t for t in tasks if t["state"] == "finished"]

    # If no unfinished tasks
    if not unfinished:
        print("✅ No unfinished tasks.")
        return {"replan": False}, [], []

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

        return {"replan": True}, robot1_tasks, robot2_tasks

    # 產生自然語言描述
    description = language_grounding(data)
    
    # === 2️⃣ Build prompt for LLM ===
    prompt = f"""
You are an expert multi-robot planner.
The environment has 2 robots delivering food items.

You have access to the *full* task list and the current status of both robots.

This is the current world state file:
- {description}

Your job:
Decide whether tasks need to be replanned between Robot1 and Robot2.

A replan **must** occur if **any** of the following conditions are true:
1. **Any robot is idle** (status = "idle").
2. **The current task allocation is inefficient.**  
   Inefficiency means that **at least one task** is currently assigned to a robot that is **not the closest or most efficient choice**.  
   You must check whether **reassigning any unfinished task** to a different robot would reduce the **total travel distance**, defined as:
   - Distance from robot's **current position** to the task's **pickup location**.
   - Plus the distance from **pickup location** to **delivery location**.
Replanning is required **even if all tasks are currently assigned**, as long as reassigning tasks to different robots **would reduce the combined total travel cost**.

Task Assignment Efficiency Criteria:
The most efficient task assignment is the one that minimizes the total travel distance for both robots combined.
For each task, calculate the total cost as:
1. Distance from the robot's current position to the task's pickup location.
2. Distance from the pickup location to the delivery (dropoff) location.
The goal is to assign and order the tasks so that the sum of these costs across all assigned tasks and both robots is minimized.

If no replan is needed, output "replan": false.

When replanning:
- Only tasks where state = "unfinished" should be considered during planning and assignment.
- Tasks marked as state = "finished" must be ignored and excluded from all planning, assignment, or output.
- Every task in the output must be an unfinished task.
- Each unfinished task must be assigned to exactly one robot.
- Each robot's assigned task list must be ordered in the **most efficient execution order**, minimizing its own total travel distance.
- Finished tasks must not appear.
- Ensure no task duplication between robots.

Output JSON ONLY in this exact format:
{{
  "replan": true/false,
  "Robot1": [
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    ...
  ],
  "Robot2": [
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    ...
  ]
}}

The order of tasks inside each robot's list MUST represent the actual execution order:
- The first item is the task that the robot should perform FIRST.
- The last item is the task that the robot should perform LAST.
Tasks must be sorted by efficiency (shortest total path cost first).

Respond with ONLY a valid JSON object.
Do not include any explanations, reasoning, or commentary outside the JSON.
If you include anything else, it will cause a system error.
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
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ LLM output not valid JSON. Skipping replan.")
        print(raw)
        return None, [], []
    # print(result)
    # === 4️⃣ If replan needed ===
    if result["replan"]:
        print("🔁 LLM decided to replan tasks.")

        # 更新任務分配狀態
        for robot_name, plan_key in [("robot1", "Robot1"), ("robot2", "Robot2")]:
            assigned_ids = [p["id"] for p in result.get(plan_key, [])]
            for t in tasks:
                if t["id"] in assigned_ids:
                    t["assigned_to"] = robot_name           # ✅ 更新任務所屬 robot
            robots[robot_name]["status"] = "executing"
            if assigned_ids:
                robots[robot_name]["current_task_id"] = assigned_ids[0]

        # 寫回 task_list.json
        with open(task_list_path, "w") as f:
            json.dump({"robots": robots, "tasks": tasks}, f, indent=2)
        print("✅ Updated task list after replan.")

        

        new_robot_tasks = _convert(result.get("Robot1", []))
        new_robot2_tasks = _convert(result.get("Robot2", []))

        # ✅ 印出任務結果供檢查
        print("🤖 Robot1 新任務規劃：")
        for i, task in enumerate(result.get("Robot1", [])):
            print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
        print("======================================================================")
        print("🦾 Robot2 新任務規劃：")
        for i, task in enumerate(result.get("Robot2", [])):
            print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
        print("======================================================================")

        # 回傳給主程式
        return result, new_robot_tasks, new_robot2_tasks

    # === 6️⃣ 若不需重新分配 ===
    else:
        print("🟢 LLM decided no replan is needed.")
        return result, [], []

