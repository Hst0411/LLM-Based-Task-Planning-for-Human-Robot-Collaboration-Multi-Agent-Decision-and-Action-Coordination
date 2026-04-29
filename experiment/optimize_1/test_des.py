# from experiment.optimize_1.LLM_task_planner import get_two_robot_tasks_from_llm
# from experiment.optimize_1.robot_idle_replanner import get_global_plan_from_llm

# replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
# replicant2_position = {"x": 4.0, "y": 0, "z": 2.82}

# print("GPT-4 開始規劃兩 robots 初始運輸順序")
# robot_tasks, robot2_tasks = get_two_robot_tasks_from_llm(
#     task_list_path="experiment/optimize_1/task_list.json",
#     robot1_position=replicant_position,
#     robot2_position=replicant2_position
# )

# new_robot_tasks, new_robot2_tasks = get_global_plan_from_llm(replicant_position, replicant2_position)

# 假設新的工作
new_tasks = [
    ('navigate_to', {'x': 1.5, 'y': 0, 'z': 2.5}),
    ('pick_up', 1234567),
    ('navigate_to', {'x': 2.5, 'y': 0, 'z': 3.5}),
    ('drop', None),
    ('navigate_to', {'x': 3.5, 'y': 0, 'z': 4.5}),
    ('pick_up', 7654321),
    ('navigate_to', {'x': 4.5, 'y': 0, 'z': 5.5}),
    ('drop', None)
]

# 原始 robot2_tasks
robot2_tasks = [
    ('navigate_to', {'x': -1.6217177549247004, 'y': 0, 'z': -3.863255628194672}),
    ('pick_up', 10724664),
    ('navigate_to', {'x': 6.0, 'y': 0, 'z': -1.4}),
    ('drop', None),
    ('navigate_to', {'x': -1.6637, 'y': 0, 'z': 2.6118}),
    ('pick_up', 6871504),
    ('navigate_to', {'x': 8.796, 'y': 0, 'z': 1.925}),
    ('drop', None),
    ('navigate_to', {'x': 5.054031081339902, 'y': 0, 'z': -3.5}),
    ('pick_up', 1051562),
    ('navigate_to', {'x': -3.51, 'y': 0, 'z': 2.46319}),
    ('drop', None)
]

# 步驟 1: 找到需要覆蓋的位置（10724664）
index_to_replace = None
for i in range(len(robot2_tasks)):
    if robot2_tasks[i] == ('pick_up', 10724664):
        index_to_replace = i
        break

# 確保找到位置
if index_to_replace is not None:
    # 步驟 2: 從該位置開始，刪除舊的任務並插入新任務
    robot2_tasks = robot2_tasks[:index_to_replace + 1] + new_tasks

# 顯示更新後的 robot2_tasks
for action, details in robot2_tasks:
    print(f"動作: {action}, 參數: {details}")

print(robot2_tasks)
