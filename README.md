# 🧠 Prompt & Semantic State Design

This project adopts a centralized LLM-based planner with a structured semantic state abstraction to enable efficient and consistent multi-agent coordination under partial observability.

### 📦 Semantic State Representation

Instead of feeding raw perception (e.g., RGB-D images) into the LLM, we convert observations into a task-oriented semantic state.

#### Format
```
{
  "object": "string",
  "status": "string",
  "confidence": float,
  "position": [x, y, z]
}
```

#### Field Description
* object : Name of the task-related object (e.g., "bread", "apple")
* status : Current task-related state of the object
* confidence : Confidence score from perception module
* position : 3D world coordinates of the object

#### Status Definition
```
unfinished            # task unfinished
picked_by_robot       # currently held by robot
picked_by_human       # currently held by human
finished              # task finished (irreversible)
```

#### Example
```
{
  "object": "bread",
  "status": "picked_by_human",
  "confidence": 0.82,
  "position": [1.25, 0.93, -2.10]
}
```

### 🤖 LLM Prompt Template

The centralized LLM acts as a global task planner. The prompt is carefully designed to enforce constraints and ensure consistent outputs.

```
### Tasks:
- You are an expert multi-robot planner for household delivery.
- The environment has N robots delivering objects.
- You have access to the current world state information.

### This is the current world state file:
{language_state_description}

### Your Job:
- [Job 1] Assign all unfinished tasks to robots efficiently (minimize total travel distance)
OR
- [Job 2] Decide whether tasks need to be replanned

### Task Assignment Efficiency Criteria:
- Minimize distance from robot to pickup location
- Minimize distance from pickup location to delivery location

### Rules:
- Only assign unfinished tasks
- Each task must be assigned to exactly ONE robot
- No duplicate assignments
- Try to balance workload across robots

### Output JSON format:
{
  "Robot1": [["Task 1"], ["Task 2"]],
  "Robot2": [["Task 3"]]
}
```

### 📤 Output Constraint

The LLM must output a valid JSON object only (no explanations or extra text):
```
{{
  "Robot1": [
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    {{"id": ..., "pickup": ..., "dropoff": ...}},
    ...
  ],
  "Robot2": [
    {{"id": 01234, "pickup": ..., "dropoff": ...}},
    {{"id": 56789, "pickup": ..., "dropoff": ...}},
    ...
  ]
}}
```
