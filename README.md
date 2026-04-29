# LLM-Based-Task-Planning-for-Human-Robot-Collaboration-Multi-Agent-Decision-and-Action-Coordination

### 🧠 Prompt & Semantic State Design

This project adopts a centralized LLM-based planner with a structured semantic state abstraction to enable efficient and consistent multi-agent coordination under partial observability.

### 📦 Semantic State Representation

Instead of feeding raw perception (e.g., RGB-D images) into the LLM, we convert observations into a compact, task-oriented semantic state.

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

### 🧾 Language State (LLM Input)

All semantic states, task information, and robot states are converted into a structured natural language description before being sent to the LLM.

#### Example
```
[Task List]
- Task 1: Move bread to kitchen table (status: unfinished)
- Task 2: Move apple to fridge (status: completed)
- Task 3: Move bowl to sink (status: picked_by_human)

[Robot States]
- Robot1: position=(1.0, 0.0, 2.0), status=idle
- Robot2: position=(-2.0, 0.0, 1.5), status=executing Task 3

[Environment Notes]
- Human is holding bowl
- Bread is on the counter
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
{
  "Robot1": [["bread"], ["apple"]],
  "Robot2": [["bowl"]]
}
```
