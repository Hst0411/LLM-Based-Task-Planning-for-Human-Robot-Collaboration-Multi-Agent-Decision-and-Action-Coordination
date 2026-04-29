import json, time, os
from openai import OpenAI
from typing import List, Tuple, Dict
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

        desc.append(
            f"Task {task_id} is to move a '{name}' from position "
            f"({pos['x']}, {pos['y']}, {pos['z']}) to target position "
            f"({target['x']}, {target['y']}, {target['z']}). "
            f"Current state: {state}. {assignment_text}"
        )

    return "\n".join(desc)

def get_two_robot_tasks_from_llm(task_list_path="experiment/optimize_1/task_list_new.json",
                                 robot1_position=None,
                                 robot2_position=None,
                                 model_name="gpt-4"):
    """
    Reads the full task_list.json (with 'robots' + 'tasks') and uses GPT-4 to
    assign unfinished tasks to both robots efficiently.

    Returns two lists of (action, target) tuples for each robot.
    """

    # === 1️⃣ Load data ===
    with open(task_list_path, "r") as f:
        data = json.load(f)

    robots = data["robots"]
    tasks = data["tasks"]

    unfinished = [t for t in tasks if t["state"] != "finished"]
    if not unfinished:
        print("✅ No unfinished tasks.")
        return [], []

    # 產生自然語言描述
    description = language_grounding(data)
    # print(description)
    # === 2️⃣ Construct LLM prompt ===
    prompt = f"""
You are an expert multi-robot planner for household delivery.

You have access to the *full* task list and the current status of both robots.

This is the current world state file:
- {description}

Your goals:
1. Assign all *unfinished* tasks to Robot1 and Robot2 efficiently (minimize total travel distance).
2. Distribute workload fairly; both robots should have tasks if possible.
3. Do not assign already-finished tasks.
4. Do not assign the same task to both robots.
5. Only output valid JSON.

Task Assignment Efficiency Criteria:
The most efficient task assignment is the one that minimizes the total travel distance for both robots combined.
For each task, calculate the total cost as:
1. Distance from the robot's current position to the task's pickup location.
2. Distance from the pickup location to the delivery (dropoff) location.
The goal is to assign and order the tasks so that the sum of these costs across all assigned tasks and both robots is minimized.

Output JSON format:
{{
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

""".strip()
    
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
    
    plan = json.loads(raw)

    # === 4️⃣ Convert to robot action sequences ===
    def convert_to_actions(robot_plan):
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


    robot1_tasks = convert_to_actions(plan["Robot1"])
    robot2_tasks = convert_to_actions(plan["Robot2"])

    # === 5️⃣ Update assignments in task_list.json ===
    for t in tasks:
        if any(t["id"] == p["id"] for p in plan["Robot1"]):
            t["assigned_to"] = "robot1"
        elif any(t["id"] == p["id"] for p in plan["Robot2"]):
            t["assigned_to"] = "robot2"

    data["robots"]["robot1"]["current_task_id"] = plan["Robot1"][0]["id"]
    data["robots"]["robot2"]["current_task_id"] = plan["Robot2"][0]["id"]
    data["robots"]["robot1"]["status"] = "executing"
    data["robots"]["robot2"]["status"] = "executing"

    with open(task_list_path, "w") as f:
        json.dump(data, f, indent=2)

    # === 6️⃣ Pretty print results ===
    print("🤖 Robot1 任務規劃：")
    for i, task in enumerate(plan["Robot1"]):
        print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
    print("======================================================================")
    print("🦾 Robot2 任務規劃：")
    for i, task in enumerate(plan["Robot2"]):
        print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
    # print(robot1_tasks)
    # print(robot2_tasks)
    return robot1_tasks, robot2_tasks

