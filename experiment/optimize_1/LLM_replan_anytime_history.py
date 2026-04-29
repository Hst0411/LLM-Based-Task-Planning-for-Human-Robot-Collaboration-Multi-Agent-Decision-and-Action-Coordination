import json, time, os, math, re
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

API_KEY = "API_KEY"

client = OpenAI(api_key=API_KEY)

HISTORY_PATH = "experiment/optimize_1/history.json"  # persistent history

def _load_history(path=HISTORY_PATH) -> List[Dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def _save_history(history: List[Dict], path=HISTORY_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def _dedupe_tasks_from_paths(paths: List[str]) -> List[Dict]:
    all_tasks = []
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                all_tasks.extend(json.load(f))
    # dedupe by id, prefer "finished" if any copy says finished
    task_map = {}
    for t in all_tasks:
        tid = str(t["id"])
        if tid not in task_map:
            task_map[tid] = t
        else:
            if task_map[tid].get("state") == "finished" or t.get("state") == "finished":
                task_map[tid]["state"] = "finished"
            # optionally update other fields
    return [v for v in task_map.values()]

def _extract_json(raw: str) -> str:
    # try to find first {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw

def _parse_llm_output(raw: str) -> Dict:
    raw = raw.strip()
    block = _extract_json(raw)
    return json.loads(block)

def _distance(a: Dict, b: Dict) -> float:
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)

def report_progress_to_LLM(
    reporter: str,                        # "robot1" or "robot2"
    robot_positions: Dict[str, Dict],    # {"robot1": {x,y,z}, "robot2": {...}}
    completed_task: Optional[Dict],      # task dict if completed, else None
    observed_human_ids: Optional[List[int]] = None,
    task_list_paths: List[str] = ["experiment/optimize_1/robot_task_list.json",
                                  "experiment/optimize_1/robot2_task_list.json"],
    history_path: str = HISTORY_PATH,
    model: str = "gpt-4"                 # or gpt-4o etc.
) -> Tuple[bool, List[Dict], List[Dict], List[Dict]]:
    """
    Returns: (replan_bool, robot1_plan, robot2_plan, updated_history)
    robot1_plan / robot2_plan are lists of dicts: {"id":..., "pickup": {...}, "dropoff": {...}}
    """

    # 1) load and append history event
    history = _load_history(history_path)
    event = {
        "ts": time.time(),
        "reporter": reporter,
        "event_type": "completed" if completed_task else "observation",
        "position": robot_positions.get(reporter),
        "completed_task": completed_task if completed_task else None,
        "observed_human": observed_human_ids or []
    }
    history.append(event)
    _save_history(history, history_path)

    # 2) compute unfinished tasks (dedup)
    deduped = _dedupe_tasks_from_paths(task_list_paths)
    unfinished = [t for t in deduped if t.get("state") != "finished"]

    # 3) compute in-progress ids (ask caller to supply them if possible; here infer from task lists)
    # you can pass explicit in-progress list from caller; for now create empty
    in_progress_ids = []  # ideally caller provides real in-progress list

    # 4) Build prompt (English)
    prompt = f"""
You are an expert multi-robot task planner.
Two robots (robot1, robot2) operate in a house and transport items.

This event was just reported:
{json.dumps(event, ensure_ascii=False, indent=2)}

Current robot positions:
- robot1: {robot_positions.get('robot1')}
- robot2: {robot_positions.get('robot2')}

In-progress task ids (do NOT reassign these):
{in_progress_ids}

Remaining unfinished tasks (deduplicated):
{json.dumps(unfinished, ensure_ascii=False, indent=2)}

History (chronological, recent last): show last 20 events:
{json.dumps(history[-20:], ensure_ascii=False, indent=2)}

Your task:
Decide whether to re-plan remaining tasks. IF you choose to re-plan, return a JSON object:
{{ "replan": true, "robot1": [{{"id":.., "pickup":{{...}}, "dropoff":{{...}}}}, ...], "robot2": [...] }}
If you choose NOT to re-plan, return:
{{ "replan": false, "robot1": [], "robot2": [] }}

Only return the JSON object (no extra text).
"""

    # 5) Call LLM
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content": "You are an expert in multi-robot coordination and task planning."},
            {"role":"user", "content": prompt}
        ],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = _parse_llm_output(raw)
    except Exception as e:
        print("⚠️ Failed to parse LLM response:", e)
        print("RAW:", raw)
        # fail safe: no replan
        return False, [], [], history

    # normalize keys to lower
    parsed_normalized = {k.lower(): v for k,v in parsed.items()}
    replan = bool(parsed_normalized.get("replan", False))
    r1 = parsed_normalized.get("robot1", [])
    r2 = parsed_normalized.get("robot2", [])

    # Validate tasks: ensure they have id/pickup/dropoff and are not finished
    def _valid_task(t):
        return ("id" in t) and (("pickup" in t) or ("position" in t)) and (("dropoff" in t) or ("target" in t))
    # Normalize field names for each task
    def _norm_task(t):
        pickup = t.get("pickup", t.get("position"))
        dropoff = t.get("dropoff", t.get("target"))
        return {"id": int(t["id"]), "pickup": pickup, "dropoff": dropoff}

    r1_clean = [_norm_task(t) for t in r1 if _valid_task(t)]
    r2_clean = [_norm_task(t) for t in r2 if _valid_task(t)]

    # final safety: remove any tasks that are already finished or in in_progress
    finished_ids = [int(t["id"]) for t in deduped if t.get("state") == "finished"]
    r1_final = [t for t in r1_clean if t["id"] not in finished_ids and str(t["id"]) not in in_progress_ids]
    r2_final = [t for t in r2_clean if t["id"] not in finished_ids and str(t["id"]) not in in_progress_ids]

    return replan, r1_final, r2_final, history


# ✅ Print formatted plan
    # print("🤖 Robot1 plan:")
    # if robot1_tasks:
    #     for i, task in enumerate(robot1_tasks):
    #         print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
    # else:
    #     print("(no changes)")
    # print("======================================================================")
    # print("🦾 Robot2 plan:")
    # if robot2_tasks:
    #     for i, task in enumerate(robot2_tasks):
    #         print(f"{i+1}. {task['id']} → from {task['pickup']} to {task['dropoff']}")
    # else:
    #     print("(no changes)")
    # print("======================================================================")
