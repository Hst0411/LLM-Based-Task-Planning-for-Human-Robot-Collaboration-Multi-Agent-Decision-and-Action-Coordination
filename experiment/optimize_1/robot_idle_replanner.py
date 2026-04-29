import json
from typing import List, Tuple, Dict
from openai import OpenAI
import math

api_key = "API_KEY"

def _distance(pos1, pos2):
    return math.sqrt((pos1["x"] - pos2["x"])**2 +
                     (pos1["y"] - pos2["y"])**2 +
                     (pos1["z"] - pos2["z"])**2)

def get_global_plan_from_llm(robot1_pos, robot2_pos,
                             task_list_path="experiment/optimize_1/task_list_new.json"):
    """
    Reads all unfinished tasks and asks LLM to assign them dynamically to both robots.
    Includes history + current progress in the prompt.
    Returns two lists of (action, target) for robot1 and robot2.
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

    # history_messages = ""
    # if history:
    #     history_messages = "Previous plans:\n" + "\n".join([str(h) for h in history])
    # history_messages = history

    prompt = f"""
Two robots are cooperating to deliver food.
Robot1 is at {robots["robot1"]["position"]}, Robot2 is at {robots["robot2"]["position"]}.

Finished items: {finished}
Unfinished items:
{unfinished}

Please assign the unfinished tasks to robot1 and robot2 in efficient order to minimize travel and avoid duplication.

IMPORTANT:
- Only output a valid JSON object.
- Do not include any explanation, text, or comments.
- The JSON must be directly parseable by Python json.loads().

JSON Output format:
{{
  "Robot1": [
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    ...
  ],
  "Robot2": [
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    ...
  ]
}}

Respond with ONLY the JSON object. Do not include any explanation or extra text.

Each is a list of tasks:
- "id": object id
- "pickup": pickup position
- "dropoff": dropoff position
    """

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a professional multi-robot planner."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    raw = response.choices[0].message.content.strip()
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


def _convert(seq):
    result = []
    for task in seq:
        result.append(("navigate_to", task["pickup"]))
        result.append(("pick_up", task["id"]))
        result.append(("navigate_to", task["dropoff"]))
        result.append(("drop", None))
    return result
