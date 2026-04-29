import json
from openai import OpenAI
from typing import List, Tuple, Dict

def get_robot_tasks_from_llm(task_list_path="experiment/robot_task_list.json", 
                             robot_position: Dict[str, float] = None) -> List[Tuple[str, object]]:
    """
    Reads the task list and uses GPT-4 to determine an efficient pickup/delivery sequence.
    Returns a list of (action, target) tuples for the robot to execute.
    """
    # Load all tasks
    with open(task_list_path, "r") as f:
        task_list = json.load(f)

    # Filter unfinished tasks
    unfinished = [task for task in task_list if task["state"] != "finished"]
    if not unfinished:
        print("✅ No unfinished tasks.")
        return []

    # Format robot start position
    robot_pos_str = "(unknown)"
    if robot_position is not None:
        robot_pos_str = f"({robot_position['x']:.2f}, {robot_position['y']:.2f}, {robot_position['z']:.2f})"

    # Build prompt: summarize food items
    item_descriptions = "\n".join([
        f"- {item['name']} (id={item['id']}): pickup={item['position']}, target={item['target']}"
        for item in unfinished
    ])

    prompt = f"""
You are a robot task planner. A robot starts at position {robot_pos_str}. It must deliver the following food items, each from a pickup position to a specified target location.

Your goal is to output a JSON array of tasks sorted in an efficient delivery order.
Try to minimize total travel distance: from the robot's current position to the first pickup,
then from each pickup to its dropoff, and between each task.

Each item in the output must contain:
- "id": (integer ID of the object)
- "pickup": (pickup position)
- "dropoff": (must match the original 'target' position of the object)

IMPORTANT:
- Use only the provided "target" field from each item as the dropoff location.
- Must ensure that all tasks are completed.
- Do not add any explanation or comments. Only return a valid JSON array.

Unfinished items:
{item_descriptions}
    """.strip()

    # Query GPT-4
    api_key = "API_KEY"
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.1,  # 增加隨機性
        messages=[
            {"role": "system", "content": "You are a robot planning assistant."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    raw = response.choices[0].message.content.strip()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ Failed to parse GPT output. Raw content:")
        print(raw)
        raise

    # Convert to action list
    task_sequence: List[Tuple[str, object]] = []
    print("🤖 LLM-generated robot task sequence:")
    for i, task in enumerate(plan):
        print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
        task_sequence.append(("navigate_to", task["pickup"]))
        task_sequence.append(("pick_up", task["id"]))
        task_sequence.append(("navigate_to", task["dropoff"]))
        task_sequence.append(("drop", None))

    return task_sequence